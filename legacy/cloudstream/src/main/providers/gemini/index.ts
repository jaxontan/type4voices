/**
 * Gemini API (Gemini 3.5 Transcribe Live) over the public Live API WebSocket.
 *
 * Official, documented path:
 *   wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=<api key>
 *
 * Frames arrive from the renderer as 20 ms 16 kHz mono pcm_s16le buffers (640 B).
 * We coalesce them into 100 ms chunks (3200 B) and stream them as
 * `realtimeInput.audio` with `mimeType: "audio/pcm;rate=16000"`.
 */
import WebSocket from 'ws';
import {
  AUDIO,
  NotConfiguredError,
  type SttEvent,
  type SttSession,
  type StartOptions,
} from '../types';

const MODEL = 'gemini-3.5-transcribe-live';
const WS_HOST = 'generativelanguage.googleapis.com';
const WS_PATH = '/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent';

const CONNECT_TIMEOUT_MS = 10_000;
const FINAL_TIMEOUT_SPEECH_MS = 6_000;
const FINAL_TIMEOUT_SILENCE_MS = 2_000;
/** 100 ms of 16 kHz mono PCM16 (5 x 20 ms frames). */
const CHUNK_BYTES = (AUDIO.sampleRate / 1000) * 100 * 2;

const LANGUAGE_MAP: Record<string, string[]> = {
  auto: [],
  en: ['en-US'],
  english: ['en-US'],
  zh: ['zh-CN'],
  chinese: ['zh-CN'],
  cantonese: ['zh-HK'],
  ja: ['ja-JP'],
  japanese: ['ja-JP'],
  ko: ['ko-KR'],
  korean: ['ko-KR'],
  es: ['es-ES'],
  spanish: ['es-ES'],
  fr: ['fr-FR'],
  french: ['fr-FR'],
  de: ['de-DE'],
  german: ['de-DE'],
  ru: ['ru-RU'],
  russian: ['ru-RU'],
};

export function resolveGeminiApiKey(configured?: string): string {
  const direct = (configured ?? '').trim();
  if (direct) return direct;
  for (const name of ['GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_API_KEY']) {
    const envVal = process.env[name]?.trim();
    if (envVal) return envVal;
  }
  return '';
}

function languageCodesFor(language?: string): string[] {
  const raw = (language ?? '').trim();
  if (!raw) return [];
  const mapped = LANGUAGE_MAP[raw.toLowerCase()];
  if (mapped !== undefined) return mapped;
  // Pass through explicit BCP-47 tags like en-SG, en-GB, zh-TW.
  if (/^[a-z]{2,3}(-[A-Za-z0-9]{2,8})+$/.test(raw)) return [raw];
  return [];
}

function cleanGlossary(glossary?: string[]): string[] {
  if (!glossary?.length) return [];
  const out: string[] = [];
  for (const term of glossary) {
    if (out.length >= 100) break;
    const clean = term.trim().replace(/\s+/g, ' ').slice(0, 64);
    if (clean) out.push(clean);
  }
  return out;
}

function extractServerError(raw: string): string {
  try {
    const payload = JSON.parse(raw) as { error?: { message?: string; status?: string } | string };
    if (typeof payload.error === 'string') return payload.error;
    if (payload.error && typeof payload.error === 'object') {
      return payload.error.message || payload.error.status || '';
    }
  } catch {
    /* non-JSON body */
  }
  return '';
}

interface GeminiServerMessage {
  setupComplete?: unknown;
  error?: { message?: string; status?: string } | string;
  serverContent?: {
    interimInputTranscription?: { text?: string };
    inputTranscription?: { text?: string };
    generationComplete?: boolean;
  };
  voiceActivity?: {
    type?: string;
  };
}

export class GeminiLiveAdapter implements SttSession {
  readonly id = 'gemini-api';
  readonly label = 'Gemini API (Transcribe Live)';

  private listeners = new Set<(e: SttEvent) => void>();
  private ws: WebSocket | null = null;
  private pendingPcm: Buffer[] = [];
  private pendingBytes = 0;
  private finals: string[] = [];
  private interim = '';
  private sawSpeech = false;
  private inputFinalSeen = false;
  private generationComplete = false;
  private activityEnd = false;
  private committing = false;
  private settle: ((text: string) => void) | null = null;
  private failure: Error | null = null;
  private framesSent = 0;

  constructor(private readonly resolveKey: () => string) {}

  async start(opts: StartOptions): Promise<void> {
    const key = this.resolveKey().trim();
    if (!key) {
      throw new NotConfiguredError(
        'Gemini API needs an API key. Paste it in settings or set GEMINI_API_KEY.',
      );
    }

    this.reset();

    const url = `wss://${WS_HOST}${WS_PATH}?key=${encodeURIComponent(key)}`;
    const ws = new WebSocket(url);
    this.ws = ws;

    const transcription: Record<string, unknown> = {
      languageCodes: languageCodesFor(opts.language),
      mode: 'VERBATIM',
    };
    const vocab = cleanGlossary(opts.glossary);
    if (vocab.length) transcription.customVocabulary = vocab;

    const setupPayload = JSON.stringify({
      setup: {
        model: `models/${MODEL}`,
        generationConfig: { responseModalities: ['TEXT'] },
        inputAudioTranscription: transcription,
      },
    });

    await new Promise<void>((resolve, reject) => {
      let settled = false;
      const done = (err?: Error) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (err) reject(err);
        else resolve();
      };

      const timer = setTimeout(() => {
        done(new Error(`Gemini API: no setupComplete within ${CONNECT_TIMEOUT_MS} ms`));
      }, CONNECT_TIMEOUT_MS);

      ws.once('open', () => {
        ws.send(setupPayload);
      });

      ws.once('unexpected-response', (_req, res) => {
        const chunks: Buffer[] = [];
        res.on('data', (c: Buffer) => chunks.push(c));
        res.on('end', () => {
          const detail = extractServerError(Buffer.concat(chunks).toString('utf8'));
          const status = res.statusCode ?? 0;
          if (status === 400 || status === 401 || status === 403) {
            done(
              new Error(
                `Gemini API key was rejected (HTTP ${status}${detail ? `: ${detail}` : ''}). Check your key in settings.`,
              ),
            );
          } else if (status === 429) {
            done(new Error('Gemini API quota exceeded (HTTP 429). Try again later or check billing.'));
          } else {
            done(new Error(`Gemini API: handshake rejected with HTTP ${status}${detail ? `: ${detail}` : ''}`));
          }
        });
      });

      ws.on('message', (data) => {
        const raw = data.toString();
        if (!settled) {
          const errMsg = extractServerError(raw);
          if (errMsg) {
            done(new Error(`Gemini API: ${errMsg}`));
            return;
          }
          try {
            const parsed = JSON.parse(raw) as GeminiServerMessage;
            if ('setupComplete' in parsed) {
              done();
              return;
            }
          } catch {
            /* ignore non-JSON */
          }
        }
        this.handleMessage(raw);
      });

      ws.on('close', (code, reason) => {
        const reasonStr = reason.toString();
        if (!settled) {
          done(
            new Error(
              `Gemini API: socket closed during setup (code ${code}${reasonStr ? `: ${reasonStr}` : ''})`,
            ),
          );
          return;
        }
        this.handleClose(code, reasonStr);
      });

      ws.on('error', (err) => {
        if (!settled) {
          done(err);
          return;
        }
        this.failure = err;
        this.emit({ type: 'error', error: err, fatal: true });
        this.finishCommit(this.buildTranscript());
      });
    });
  }

  push(frame: Buffer): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN || !frame.length) return;
    this.pendingPcm.push(frame);
    this.pendingBytes += frame.length;
    if (this.pendingBytes >= CHUNK_BYTES) {
      this.flushAudio();
    }
  }

  async commit(): Promise<string> {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.close();
      if (this.finals.length || this.interim) return this.buildTranscript();
      throw this.failure ?? new NotConfiguredError('Gemini API: socket closed before commit');
    }

    this.committing = true;
    this.flushAudio();

    try {
      ws.send(JSON.stringify({ realtimeInput: { audioStreamEnd: true } }));
    } catch (err) {
      this.close();
      if (this.finals.length || this.interim) return this.buildTranscript();
      throw err;
    }

    if (this.inputFinalSeen && (this.generationComplete || this.activityEnd)) {
      const text = this.buildTranscript();
      if (text) this.emit({ type: 'final', text });
      this.close();
      return text;
    }

    const waitMs = this.sawSpeech || this.interim || this.finals.length
      ? FINAL_TIMEOUT_SPEECH_MS
      : FINAL_TIMEOUT_SILENCE_MS;

    const text = await new Promise<string>((resolve) => {
      this.settle = resolve;
      setTimeout(() => {
        if (this.settle === resolve) {
          this.settle = null;
          resolve(this.buildTranscript());
        }
      }, waitMs);
    });

    if (text) this.emit({ type: 'final', text });
    this.close();
    return text;
  }

  abort(): void {
    this.settle?.('');
    this.settle = null;
    this.close();
  }

  stats(): { framesSent: number; connected: boolean } {
    return { framesSent: this.framesSent, connected: this.ws?.readyState === WebSocket.OPEN };
  }

  on(listener: (e: SttEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private flushAudio(): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN || this.pendingBytes === 0) return;
    const pcm = Buffer.concat(this.pendingPcm, this.pendingBytes);
    this.pendingPcm = [];
    this.pendingBytes = 0;

    try {
      ws.send(
        JSON.stringify({
          realtimeInput: {
            audio: {
              mimeType: `audio/pcm;rate=${AUDIO.sampleRate}`,
              data: pcm.toString('base64'),
            },
          },
        }),
      );
      this.framesSent += 1;
    } catch (err) {
      this.emit({ type: 'error', error: err as Error, fatal: false });
    }
  }

  private handleMessage(raw: string): void {
    const errMsg = extractServerError(raw);
    if (errMsg) {
      this.failure = new Error(`Gemini API: ${errMsg}`);
      this.emit({ type: 'error', error: this.failure, fatal: true });
      this.finishCommit('');
      return;
    }

    let parsed: GeminiServerMessage;
    try {
      parsed = JSON.parse(raw) as GeminiServerMessage;
    } catch {
      return;
    }

    const va = parsed.voiceActivity;
    if (va?.type === 'ACTIVITY_START') {
      this.sawSpeech = true;
    } else if (va?.type === 'ACTIVITY_END') {
      this.activityEnd = true;
    }

    const sc = parsed.serverContent;
    if (sc) {
      if (sc.interimInputTranscription?.text) {
        this.sawSpeech = true;
        this.interim = sc.interimInputTranscription.text;
        const combined = this.buildTranscript();
        if (combined) this.emit({ type: 'interim', text: combined });
      }
      if (sc.inputTranscription?.text) {
        this.sawSpeech = true;
        this.finals.push(sc.inputTranscription.text);
        this.interim = '';
        this.inputFinalSeen = true;
        const combined = this.buildTranscript();
        if (combined) this.emit({ type: 'interim', text: combined });
      }
      if (sc.generationComplete) {
        this.generationComplete = true;
      }
    }

    if (this.committing && this.inputFinalSeen && (this.generationComplete || this.activityEnd)) {
      this.finishCommit(this.buildTranscript());
    }
  }

  private handleClose(code: number, reason: string): void {
    if (this.settle) {
      const text = this.buildTranscript();
      if (!text && code !== 1000 && code !== 1005) {
        const detail = `closed before final transcript (code ${code}${reason ? `: ${reason}` : ''})`;
        this.failure = new Error(`Gemini API: ${detail}`);
        this.emit({ type: 'error', error: this.failure, fatal: true });
      }
      this.finishCommit(text);
    }
  }

  private finishCommit(text: string): void {
    const resolve = this.settle;
    this.settle = null;
    resolve?.(text);
  }

  private buildTranscript(): string {
    const parts = [...this.finals];
    if (this.interim) parts.push(this.interim);
    return parts.join(' ').replace(/\s+/g, ' ').trim();
  }

  private close(): void {
    const ws = this.ws;
    this.ws = null;
    this.settle = null;
    this.pendingPcm = [];
    this.pendingBytes = 0;
    if (!ws) return;
    ws.removeAllListeners();
    try {
      ws.close();
    } catch {
      /* already closed */
    }
  }

  private reset(): void {
    this.pendingPcm = [];
    this.pendingBytes = 0;
    this.finals = [];
    this.interim = '';
    this.sawSpeech = false;
    this.inputFinalSeen = false;
    this.generationComplete = false;
    this.activityEnd = false;
    this.committing = false;
    this.failure = null;
    this.settle = null;
    this.framesSent = 0;
  }

  private emit(e: SttEvent): void {
    for (const l of this.listeners) l(e);
  }
}
