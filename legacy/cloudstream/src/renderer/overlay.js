/* Overlay: owns the microphone while an utterance is live.
 *
 * Everything here is deliberately provider-agnostic. It produces 16 kHz mono
 * pcm_s16le frames of 20 ms (320 samples, 640 bytes) no matter what the device
 * reports, so every backend receives identical audio.
 */

const WORKLET_SRC = `
class Framer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.frame = new Int16Array(320);
    this.n = 0;
    this.pos = 0;
  }
  process(inputs) {
    const input = inputs[0];
    const channel = input && input[0];
    if (!channel) return true;
    const step = sampleRate / 16000;
    let pos = this.pos;
    while (pos < channel.length) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const a = channel[i] !== undefined ? channel[i] : 0;
      const b = channel[i + 1] !== undefined ? channel[i + 1] : a;
      let s = a + (b - a) * frac;
      if (s > 1) s = 1;
      else if (s < -1) s = -1;
      this.frame[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
      if (this.n === this.frame.length) {
        this.port.postMessage(this.frame.slice());
        this.n = 0;
      }
      pos += step;
    }
    this.pos = pos - channel.length;
    return true;
  }
}
registerProcessor('framer', Framer);
`;

const els = {
  state: document.getElementById('state'),
  text: document.getElementById('text'),
  level: document.getElementById('level'),
  dot: document.getElementById('dot'),
};

let ctx = null;
let stream = null;
let source = null;
let node = null;
let sink = null;
let silenceTimer = null;
let lastVoiceAt = 0;
let silenceMs = 15000;
let committing = false;
let live = false;

function setState(label) {
  els.state.textContent = label;
}

function setText(text, kind) {
  els.text.textContent = text;
  els.text.className = kind || '';
}

function rms(chunk) {
  const samples = new Int16Array(chunk);
  let sum = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const v = samples[i] / 32768;
    sum += v * v;
  }
  return Math.sqrt(sum / samples.length);
}

async function startCapture() {
  if (live) return;
  committing = false;
  setText('', '');
  setState('listening');

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (err) {
    setText('Microphone unavailable: ' + err.message, 'error');
    setState('blocked');
    window.cs.send('cs:cancel');
    return;
  }

  ctx = new AudioContext({ sampleRate: 16000 });
  const url = URL.createObjectURL(new Blob([WORKLET_SRC], { type: 'application/javascript' }));
  await ctx.audioWorklet.addModule(url);
  URL.revokeObjectURL(url);

  source = ctx.createMediaStreamSource(stream);
  node = new AudioWorkletNode(ctx, 'framer');

  // A worklet only gets pulled when it reaches the graph; gain 0 keeps it silent.
  sink = ctx.createGain();
  sink.gain.value = 0;

  node.port.onmessage = (event) => {
    if (committing) return;
    const chunk = event.data;
    const level = rms(chunk);
    els.level.style.width = Math.min(100, level * 900) + '%';
    if (level > 0.01) lastVoiceAt = Date.now();
    window.cs.send('cs:audio', chunk.buffer);
  };

  source.connect(node);
  node.connect(sink);
  sink.connect(ctx.destination);
  await ctx.resume();

  lastVoiceAt = Date.now();
  live = true;

  if (silenceTimer) clearInterval(silenceTimer);
  silenceTimer = setInterval(() => {
    if (!live || committing) return;
    if (Date.now() - lastVoiceAt >= silenceMs) {
      void commit();
    }
  }, 250);
}

async function commit() {
  if (!live || committing) return;
  committing = true;
  setState('transcribing');
  await stopCapture();
  window.cs.send('cs:commit');
}

async function stopCapture() {
  live = false;
  if (silenceTimer) clearInterval(silenceTimer);
  silenceTimer = null;
  try {
    if (node) {
      node.port.onmessage = null;
      node.disconnect();
    }
    if (source) source.disconnect();
    if (sink) sink.disconnect();
  } catch {
    /* already torn down */
  }
  node = null;
  source = null;
  sink = null;
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  if (ctx) {
    try {
      await ctx.close();
    } catch {
      /* ignore */
    }
    ctx = null;
  }
  els.level.style.width = '0%';
}

window.cs.on('cs:begin', (payload) => {
  silenceMs = (payload && payload.silenceMs) || 15000;
  committing = false;
  live = false;
  void startCapture();
});

window.cs.on('cs:interim', (payload) => {
  if (!payload) return;
  setText(payload.text, '');
});

window.cs.on('cs:final', (payload) => {
  committing = true;
  setText(payload.text, 'final');
  setState(payload.injected === false ? 'copied to clipboard' : 'inserted');
  void stopCapture();
});

window.cs.on('cs:error', (payload) => {
  committing = true;
  setText(payload.message, 'error');
  setState('error');
  void stopCapture();
});

window.cs.on('cs:notice', (payload) => {
  if (!payload || !payload.message) return;
  if (!els.text.textContent) setText(payload.message, '');
});

window.cs.on('cs:state', (payload) => {
  if (payload && payload.phase === 'idle') {
    committing = true;
    void stopCapture();
  }
});
