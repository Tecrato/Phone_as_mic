class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.buf = new Float32Array(1024);   // ~21 ms @ 48 kHz
    this.idx = 0;
  }
  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;
    const ch = input[0];
    for (let i = 0; i < ch.length; i++) {
      this.buf[this.idx++] = ch[i];
      if (this.idx >= this.buf.length) {
        this.port.postMessage(this.buf.slice());
        this.idx = 0;
      }
    }
    return true;
  }
}
registerProcessor('mic-processor', MicProcessor);