"""
Servidor de micrófono remoto con DSP y jitter buffer.
"""
from __future__ import annotations

import datetime
import ipaddress
import socket
import threading
import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# ---------------------- Configuración ----------------------
HOST = "0.0.0.0"
PORT = 8443
DEFAULT_RATE = 48_000
CHANNELS = 1
BLOCK_SIZE = 960                    # 20 ms @ 48 kHz
MAX_BUFFER_SECONDS = 2.0            # buffer máximo (absorbe picos de red)
PREBUFFER_SECONDS = 0.20            # arranca a reproducir tras 100 ms acumulados
                                    # súbelo a 0.2 si tu Wi-Fi es inestable

# ----- DSP (procesado de audio) -----
HPF_ENABLED       = True            # filtro paso-alto (quita retumbe/ruido grave)
HPF_CUTOFF_HZ     = 100.0
LIMITER_ENABLED   = True            # limita picos para evitar saturación
GATE_ENABLED      = False           # puerta de ruido (off por defecto)
GATE_THRESHOLD_DB = -50.0
AGC_ENABLED       = False           # ganancia automática (off por defecto)

BASE_DIR = Path(__file__).parent
CERT_FILE = BASE_DIR / "cert.pem"
KEY_FILE = BASE_DIR / "key.pem"

# ---------------------- DSP ----------------------
class AudioProcessor:
    """Filtro paso-alto + limiter + (opcional) gate y AGC. Todo int16 in/out."""

    def __init__(self, sample_rate: int, channels: int):
        self.sr = sample_rate
        self.ch = channels

        # High-pass one-pole (vectorizado por bloques)
        if HPF_ENABLED:
            rc = 1.0 / (2.0 * np.pi * HPF_CUTOFF_HZ)
            dt = 1.0 / sample_rate
            self.hp_alpha = rc / (rc + dt)
        else:
            self.hp_alpha = 1.0
        self.hp_x_prev = np.zeros(channels, dtype=np.float32)
        self.hp_y_prev = np.zeros(channels, dtype=np.float32)

        # Gate
        self.gate_thr = 10.0 ** (GATE_THRESHOLD_DB / 20.0)
        self.gate_state = 1.0
        self.gate_att = 1.0 - np.exp(-1.0 / (sample_rate * 0.005))
        self.gate_rel = 1.0 - np.exp(-1.0 / (sample_rate * 0.100))

        # AGC
        self.agc_target = 0.15
        self.agc_gain = 1.0
        self.agc_att = 1.0 - np.exp(-1.0 / (sample_rate * 0.010))
        self.agc_rel = 1.0 - np.exp(-1.0 / (sample_rate * 0.500))

        self.limiter_thr = 0.97

    def process(self, samples: np.ndarray) -> np.ndarray:
        # samples int16 (N, ch) -> float32 (N, ch) en [-1, 1]
        x = samples.astype(np.float32) / 32768.0

        # High-pass por muestras
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

        # Gate (por bloque, decisión por RMS)
        if GATE_ENABLED:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            target = 1.0 if rms > self.gate_thr else 0.0
            rate = self.gate_att if target > self.gate_state else self.gate_rel
            self.gate_state += rate * (target - self.gate_state)
            x = x * self.gate_state

        # AGC
        if AGC_ENABLED:
            rms = float(np.sqrt(np.mean(x * x) + 1e-12))
            if rms > 1e-4:
                desired = self.agc_target / rms
                rate = self.agc_att if desired < self.agc_gain else self.agc_rel
                self.agc_gain += rate * (desired - self.agc_gain)
                self.agc_gain = float(np.clip(self.agc_gain, 0.1, 20.0))
            x = x * self.agc_gain

        # Limiter
        if LIMITER_ENABLED:
            x = np.clip(x, -self.limiter_thr, self.limiter_thr)

        return (x * 32768.0).astype(np.int16)


# ---------------------- Jitter buffer ----------------------
class AudioBuffer:
    """
    Buffer con prebuffering. No reproduce hasta tener >= prebuffer_bytes
    acumulados; esto elimina los micro-cortes que suenan a 'granulado'.
    """

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
                # overrun serio: descartamos la mitad para dejar margen
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
            # underrun: lo que haya + silencio
            self.underruns += 1
            avail = len(self._buf)
            chunk = bytes(self._buf) + b"\x00" * (n_bytes - avail)
            self._buf.clear()
            return chunk


# ---------------------- Seguimiento de conexiones ----------------------
_conn_lock = threading.Lock()
_conn_counter = 0
active_connections: dict[int, dict] = {}

# buffer y processor (se crean al arrancar con el rate real)
audio_buf: AudioBuffer | None = None
audio_proc: AudioProcessor | None = None
_current_rate: int = DEFAULT_RATE


def _client_ip(ws: WebSocket) -> str:
    xff = ws.headers.get("x-forwarded-for")
    if xff: return xff.split(",")[0].strip()
    return ws.client.host if ws.client else "desconocido"

def _short_ua(ws: WebSocket) -> str:
    ua = ws.headers.get("user-agent", "desconocido")
    return ua if len(ua) <= 70 else ua[:67] + "..."

def _fmt_duration(seconds: float) -> str:
    s = int(seconds); h, r = divmod(s, 3600); m, s = divmod(r, 60)
    if h: return f"{h}h {m:02d}m {s:02d}s"
    if m: return f"{m}m {s:02d}s"
    return f"{s}s"

def _now() -> str:
    return datetime.datetime.now().strftime("%H:%M:%S")


# ---------------------- Stream hacia VB-Cable ----------------------
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


def find_cable_input() -> int:
    for idx, dev in enumerate(sd.query_devices()):
        if "CABLE Input" in dev["name"] and dev["max_output_channels"] > 0:
            return idx
    raise RuntimeError(
        "No se encontró 'CABLE Input'. Instala VB-Audio Virtual Cable."
    )


def start_stream(rate: int) -> None:
    global _stream, audio_buf, audio_proc, _current_rate
    with _stream_lock:
        if _stream is not None:
            return
        device = find_cable_input()
        audio_buf = AudioBuffer(rate, CHANNELS)
        audio_proc = AudioProcessor(rate, CHANNELS)
        _current_rate = rate

        _stream = sd.OutputStream(
            samplerate=rate,
            channels=CHANNELS,
            dtype="int16",
            blocksize=BLOCK_SIZE,
            device=device,
            callback=_audio_callback,
        )
        _stream.start()

        print(f"[audio] escribiendo en '{sd.query_devices(device)['name']}' @ {rate} Hz")
        print(f"[audio] DSP: HPF={'ON' if HPF_ENABLED else 'OFF'} "
              f"gate={'ON' if GATE_ENABLED else 'OFF'} "
              f"agc={'ON' if AGC_ENABLED else 'OFF'} "
              f"limiter={'ON' if LIMITER_ENABLED else 'OFF'}")
        print(f"[audio] prebuffer: {int(PREBUFFER_SECONDS*1000)} ms")


# ---------------------- Certificado SSL ----------------------
def _lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80)); return s.getsockname()[0]
    finally:
        s.close()


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


# ---------------------- FastAPI ----------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global _conn_counter
    await ws.accept()

    # rate desde query param
    try:
        client_rate = int(ws.query_params.get("rate", DEFAULT_RATE))
    except ValueError:
        client_rate = DEFAULT_RATE
    if client_rate < 8000 or client_rate > 192000:
        client_rate = DEFAULT_RATE

    # Si no coincide con el stream abierto, avisamos (no cambiamos en caliente)
    if _stream is not None and client_rate != _current_rate:
        print(f"[ws] ⚠️ cliente @ {client_rate} Hz, servidor @ {_current_rate} Hz "
              f"(posible pitch alterado)")

    with _conn_lock:
        _conn_counter += 1
        conn_id = _conn_counter
        active_connections[conn_id] = {"ip": _client_ip(ws)}

    ip = _client_ip(ws)
    ua = _short_ua(ws)
    t_start = time.monotonic()
    n_active = len(active_connections)

    print(f"[{_now()}] [+] Dispositivo conectado   #{conn_id}")
    print(f"           IP: {ip}")
    print(f"           UA: {ua}")
    print(f"           Rate: {client_rate} Hz")
    print(f"           Activas: {n_active}")

    bytes_received = 0
    reason = "cierre normal"

    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) % 2:
                data = data[:-1]
            bytes_received += len(data)
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

        print(f"[{_now()}] [-] Dispositivo desconectado #{conn_id}")
        print(f"           IP: {ip}")
        print(f"           Motivo: {reason}")
        print(f"           Duración: {_fmt_duration(duration)}")
        print(f"           Datos: {bytes_received / 1024:.1f} KB  ({kbps:.1f} KB/s)")
        print(f"           Buffer: underruns={under}  overruns={over}")
        print(f"           Activas: {n_active}")


# ---------------------- main ----------------------
if __name__ == "__main__":
    start_stream(DEFAULT_RATE)
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