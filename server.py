"""
Servidor de micrófono remoto con DSP, VAD y selector de dispositivo.
"""
from __future__ import annotations

import argparse
import datetime
import ipaddress
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

try:
    import webrtcvad
    _HAS_WEBRTCVAD = True
except ImportError:
    _HAS_WEBRTCVAD = False

# ---- Config ----
HOST = "0.0.0.0"
PORT = 8443
DEFAULT_RATE = 48_000
CHANNELS = 1
BLOCK_SIZE = 960
MAX_BUFFER_SECONDS = 2.0
PREBUFFER_SECONDS = 0.10

HPF_ENABLED       = True
HPF_CUTOFF_HZ     = 100.0
LIMITER_ENABLED   = True
GATE_ENABLED      = False
GATE_THRESHOLD_DB = -50.0
AGC_ENABLED       = False

BASE_DIR  = Path(__file__).parent
CERT_FILE = BASE_DIR / "cert.pem"
KEY_FILE  = BASE_DIR / "key.pem"


# ================= DSP =================
class AudioProcessor:
    def __init__(self, sample_rate: int, channels: int):
        self.sr = sample_rate
        self.ch = channels
        if HPF_ENABLED:
            rc = 1.0 / (2.0 * np.pi * HPF_CUTOFF_HZ)
            dt = 1.0 / sample_rate
            self.hp_alpha = rc / (rc + dt)
        else:
            self.hp_alpha = 1.0
        self.hp_x_prev = np.zeros(channels, dtype=np.float32)
        self.hp_y_prev = np.zeros(channels, dtype=np.float32)
        self.gate_thr = 10.0 ** (GATE_THRESHOLD_DB / 20.0)
        self.gate_state = 1.0
        self.gate_att = 1.0 - np.exp(-1.0 / (sample_rate * 0.005))
        self.gate_rel = 1.0 - np.exp(-1.0 / (sample_rate * 0.100))
        self.agc_target = 0.15
        self.agc_gain = 1.0
        self.agc_att = 1.0 - np.exp(-1.0 / (sample_rate * 0.010))
        self.agc_rel = 1.0 - np.exp(-1.0 / (sample_rate * 0.500))
        self.limiter_thr = 0.97

    def process(self, samples: np.ndarray) -> np.ndarray:
        x = samples.astype(np.float32) / 32768.0
        if HPF_ENABLED:
            y = np.empty_like(x)
            a = self.hp_alpha
            x_prev = self.hp_x_prev.copy()
            y_prev = self.hp_y_prev.copy()
            for i in range(len(x)):
                y_prev = y_prev + a * (x[i] - x_prev)
                x_prev = x[i]
                y[i] = y_prev
            self.hp_x_prev = x_prev
            self.hp_y_prev = y_prev
            x = y
        if GATE_ENABLED:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            target = 1.0 if rms > self.gate_thr else 0.0
            rate = self.gate_att if target > self.gate_state else self.gate_rel
            self.gate_state += rate * (target - self.gate_state)
            x = x * self.gate_state
        if AGC_ENABLED:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            if rms > 1e-4:
                desired = self.agc_target / rms
                rate = self.agc_att if desired < self.agc_gain else self.agc_rel
                self.agc_gain += rate * (desired - self.agc_gain)
                self.agc_gain = float(np.clip(self.agc_gain, 0.1, 20.0))
            x = x * self.agc_gain
        if LIMITER_ENABLED:
            x = np.clip(x, -self.limiter_thr, self.limiter_thr)
        return (x * 32768.0).astype(np.int16)


# ================= VAD =================
class VADProcessor:
    """
    VAD con hangover. Usa webrtcvad si está disponible; si no, energía (RMS).
    """
    def __init__(self, sample_rate: int = 48000, aggressiveness: int = 2,
                 hangover_ms: int = 300, energy_threshold_db: float = -40.0):
        self.sr = sample_rate
        self.frame_samples = int(sample_rate * 0.02)   # 20 ms
        self.frame_bytes = self.frame_samples * 2
        self.hangover_frames = max(1, int(hangover_ms / 20))
        self.energy_threshold = 10.0 ** (energy_threshold_db / 20.0)

        if _HAS_WEBRTCVAD and sample_rate in (8000, 16000, 32000, 48000):
            self.vad = webrtcvad.Vad(aggressiveness)
            self.mode = "webrtcvad"
        else:
            self.vad = None
            self.mode = "energy"

        self._remainder = b""
        self._hangover = 0
        self.frames_speech = 0
        self.frames_silence = 0

    def _is_speech(self, frame: bytes) -> bool:
        if self.vad is not None:
            try:
                return self.vad.is_speech(frame, self.sr)
            except Exception:
                pass
        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        return rms > self.energy_threshold

    def process(self, data: bytes) -> bytes:
        self._remainder += data
        out = bytearray()
        while len(self._remainder) >= self.frame_bytes:
            frame = self._remainder[:self.frame_bytes]
            self._remainder = self._remainder[self.frame_bytes:]
            if self._is_speech(frame):
                self._hangover = self.hangover_frames
                self.frames_speech += 1
                out.extend(frame)
            elif self._hangover > 0:
                self._hangover -= 1
                out.extend(frame)
            else:
                self.frames_silence += 1
                out.extend(b"\x00" * self.frame_bytes)
        return bytes(out)


# ================= Buffer =================
class AudioBuffer:
    def __init__(self, sample_rate: int, channels: int,
                 max_seconds: float = MAX_BUFFER_SECONDS,
                 prebuffer_seconds: float = PREBUFFER_SECONDS):
        bytes_per_sample = 2 * channels
        self.max_bytes = int(sample_rate * bytes_per_sample * max_seconds)
        self.prebuffer_bytes = int(sample_rate * bytes_per_sample * prebuffer_seconds)
        self._buf = bytearray()
        self._lock = threading.Lock()
        self._primed = False
        self.underruns = 0
        self.overruns = 0

    def push(self, data: bytes) -> None:
        if not data:
            return
        with self._lock:
            self._buf.extend(data)
            if len(self._buf) > self.max_bytes:
                self.overruns += 1
                del self._buf[:len(self._buf) - self.max_bytes // 2]
            if not self._primed and len(self._buf) >= self.prebuffer_bytes:
                self._primed = True

    def pull(self, n_bytes: int) -> bytes:
        with self._lock:
            if not self._primed:
                return b"\x00" * n_bytes
            if len(self._buf) >= n_bytes:
                chunk = bytes(self._buf[:n_bytes])
                del self._buf[:n_bytes]
                return chunk
            self.underruns += 1
            avail = len(self._buf)
            chunk = bytes(self._buf) + b"\x00" * (n_bytes - avail)
            self._buf.clear()
            return chunk


# ================= Estado global =================
_conn_lock = threading.Lock()
_conn_counter = 0
active_connections: dict[int, dict] = {}

audio_buf: AudioBuffer | None = None
audio_proc: AudioProcessor | None = None
_current_rate: int = DEFAULT_RATE


# ================= Utilidades =================
def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


def _fmt_duration(seconds: float) -> str:
    s = int(seconds); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    if h: return f"{h}h {m:02d}m {s:02d}s"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"


def _client_ip(ws: WebSocket) -> str:
    xff = ws.headers.get("x-forwarded-for")
    if xff: return xff.split(",")[0].strip()
    return ws.client.host if ws.client else "desconocido"


def _short_ua(ws: WebSocket) -> str:
    ua = ws.headers.get("user-agent", "desconocido")
    return ua if len(ua) <= 70 else ua[:67] + "..."


def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    finally:
        s.close()


# ================= Certificado autofirmado =================
def ensure_cert() -> tuple[Path, Path]:
    if CERT_FILE.exists() and KEY_FILE.exists():
        return CERT_FILE, KEY_FILE

    print("[tls] generando certificado autofirmado...")
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    lan_ip = _lan_ip()
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, lan_ip)])
    now = datetime.datetime.utcnow()
    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.IPAddress(ipaddress.ip_address(lan_ip)),
                x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
                x509.DNSName("localhost"),
            ]), critical=False,
        )
        .sign(key, hashes.SHA256())
    )
    KEY_FILE.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    CERT_FILE.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    return CERT_FILE, KEY_FILE


# ================= Audio stream =================
_stream: sd.OutputStream | None = None
_stream_lock = threading.Lock()


def _audio_callback(outdata, frames, time_info, status):
    needed = frames * CHANNELS * 2
    raw = audio_buf.pull(needed)
    if len(raw) < needed:
        raw += b"\x00" * (needed - len(raw))
    samples = np.frombuffer(raw, dtype=np.int16).reshape(-1, CHANNELS)
    if audio_proc is not None:
        samples = audio_proc.process(samples)
    outdata[:] = samples


def list_devices() -> None:
    print("\nDispositivos de audio disponibles:\n")
    print(f"{'Idx':<4} {'In':<4} {'Out':<4} Nombre")
    print("-" * 70)
    for idx, dev in enumerate(sd.query_devices()):
        in_ch = dev["max_input_channels"]
        out_ch = dev["max_output_channels"]
        marker = "  ← probable VB-Cable" if "CABLE Input" in dev["name"] else ""
        print(f"{idx:<4} {in_ch:<4} {out_ch:<4} {dev['name']}{marker}")
    print()


def resolve_device(spec: str | None) -> int:
    if spec is None:
        for idx, dev in enumerate(sd.query_devices()):
            if "CABLE Input" in dev["name"] and dev["max_output_channels"] > 0:
                print(f"[audio] Auto-seleccionado: {idx} — {dev['name']}")
                return idx
        try:
            idx = sd.default.device[1]
            print(f"[audio] AVISO: 'CABLE Input' no encontrado. Usando dispositivo "
                  f"por defecto: {idx} — {sd.query_devices(idx)['name']}")
            return idx
        except Exception:
            raise RuntimeError(
                "No hay dispositivo de salida disponible. "
                "Instala VB-Cable o usa --device <nombre|índice>."
            )
    try:
        idx = int(spec)
        dev = sd.query_devices(idx)
        if dev["max_output_channels"] < 1:
            raise ValueError(f"El dispositivo {idx} no tiene canales de salida")
        print(f"[audio] Seleccionado por índice: {idx} — {dev['name']}")
        return idx
    except ValueError:
        pass
    matches = [
        idx for idx, dev in enumerate(sd.query_devices())
        if dev["max_output_channels"] >= 1 and spec.lower() in dev["name"].lower()
    ]
    if not matches:
        raise RuntimeError(f"No se encontró dispositivo con nombre que contenga '{spec}'")
    if len(matches) > 1:
        print(f"[audio] Varios dispositivos coinciden con '{spec}':")
        for m in matches:
            print(f"  {m}: {sd.query_devices(m)['name']}")
        print(f"[audio] Usando el primero: {matches[0]}")
    print(f"[audio] Seleccionado por nombre: {matches[0]} — {sd.query_devices(matches[0])['name']}")
    return matches[0]


def start_stream(rate: int, device_idx: int) -> None:
    global _stream, audio_buf, audio_proc, _current_rate
    with _stream_lock:
        if _stream is not None:
            return
        audio_buf  = AudioBuffer(rate, CHANNELS)
        audio_proc = AudioProcessor(rate, CHANNELS)
        _current_rate = rate
        _stream = sd.OutputStream(
            samplerate=rate, channels=CHANNELS, dtype="int16",
            blocksize=BLOCK_SIZE, device=device_idx, callback=_audio_callback,
        )
        _stream.start()
        print(f"[audio] Stream @ {rate} Hz en '{sd.query_devices(device_idx)['name']}'")
        print(f"[audio] DSP: HPF={'ON' if HPF_ENABLED else 'OFF'} "
              f"gate={'ON' if GATE_ENABLED else 'OFF'} "
              f"agc={'ON' if AGC_ENABLED else 'OFF'} "
              f"limiter={'ON' if LIMITER_ENABLED else 'OFF'}")
        print(f"[audio] prebuffer: {int(PREBUFFER_SECONDS*1000)} ms")


# ================= FastAPI =================
app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _conn_counter
    await ws.accept()

    try:
        client_rate = int(ws.query_params.get("rate", DEFAULT_RATE))
    except ValueError:
        client_rate = DEFAULT_RATE
    if client_rate < 8000 or client_rate > 192000:
        client_rate = DEFAULT_RATE

    vad_enabled = ws.query_params.get("vad", "0") == "1"
    vad_aggr    = max(0, min(3, int(ws.query_params.get("aggr", "2") or "2")))

    if _stream is not None and client_rate != _current_rate:
        print(f"[ws] ⚠️ cliente @ {client_rate} Hz, servidor @ {_current_rate} Hz")

    vad_proc = VADProcessor(client_rate, aggressiveness=vad_aggr) if vad_enabled else None

    with _conn_lock:
        _conn_counter += 1
        conn_id = _conn_counter
        active_connections[conn_id] = {"ip": _client_ip(ws)}

    ip = _client_ip(ws); ua = _short_ua(ws)
    t_start = time.monotonic()
    n_active = len(active_connections)

    print(f"[{_now()}] [+] Conectado #{conn_id}  IP={ip}")
    print(f"           UA: {ua}")
    print(f"           Rate: {client_rate} Hz  |  VAD: "
          f"{'ON ('+vad_proc.mode+')' if vad_proc else 'OFF'}  |  Activas: {n_active}")

    bytes_received = 0
    reason = "cierre normal"

    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) % 2:
                data = data[:-1]
            bytes_received += len(data)
            if vad_proc is not None:
                data = vad_proc.process(data)
            if audio_buf is not None:
                audio_buf.push(data)
    except WebSocketDisconnect:
        reason = "WebSocket desconectado"
    except Exception as e:
        reason = f"error: {e}"
    finally:
        with _conn_lock:
            active_connections.pop(conn_id, None)
            n_active = len(active_connections)

        duration = time.monotonic() - t_start
        kbps = (bytes_received / 1024 / duration) if duration > 0 else 0.0
        under = audio_buf.underruns if audio_buf else 0
        over  = audio_buf.overruns  if audio_buf else 0

        print(f"[{_now()}] [-] Desconectado #{conn_id}")
        print(f"           Motivo: {reason}  |  Duración: {_fmt_duration(duration)}")
        print(f"           Datos: {bytes_received/1024:.1f} KB recibidos  "
              f"({kbps:.1f} KB/s)")
        if vad_proc is not None:
            total = vad_proc.frames_speech + vad_proc.frames_silence
            ratio = (vad_proc.frames_speech / total * 100) if total else 0.0
            print(f"           VAD: {vad_proc.frames_speech} frames voz / "
                  f"{total} ({ratio:.0f}%)")
        print(f"           Buffer: underruns={under}  overruns={over}")
        print(f"           Activas: {n_active}")


# ================= CLI / main =================
def parse_args():
    p = argparse.ArgumentParser(description="Micrófono remoto (WS + DSP + VAD)")
    p.add_argument("--device", "-d", type=str, default=None,
                   help="Dispositivo de salida (nombre parcial o índice). "
                        "Por defecto busca 'CABLE Input'.")
    p.add_argument("--list-devices", "-l", action="store_true",
                   help="Lista dispositivos de audio y sale.")
    p.add_argument("--rate", "-r", type=int, default=DEFAULT_RATE,
                   help=f"Sample rate (default: {DEFAULT_RATE})")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.list_devices:
        list_devices(); sys.exit(0)

    try:
        device_idx = resolve_device(args.device)
    except RuntimeError as e:
        print(f"[ERROR] {e}"); sys.exit(1)

    start_stream(args.rate, device_idx)
    cert, key = ensure_cert()
    ip = _lan_ip()

    print(f"\n  📱  Abre en tu teléfono:  https://{ip}:{PORT}")
    print(f"  🕒  Servidor iniciado a las {_now()}")
    print(f"  👀  Esperando conexiones... (Ctrl+C para salir)\n")

    uvicorn.run(
        app, host=HOST, port=PORT,
        ssl_certfile=str(cert), ssl_keyfile=str(key),
        log_level="warning",
    )