/**
 * Claude voice: a WebSocket speech client that speaks whatever protocol.json says.
 *
 * This is the adapter for "route my audio to the endpoint my own subscription
 * already pays for". It is deliberately dumb: no protocol knowledge is embedded
 * here, only transport, buffering and lifecycle. If Anthropic changes a field you
 * re-capture and edit JSON; you never touch this file.
 *
 * The endpoint is event-typed: `TranscriptText` carries a running transcript,
 * `TranscriptEndpoint` ends the utterance and promotes that running text, and
 * `CloseStream` is what asks for the final. The socket also needs a `KeepAlive`
 * on open and every 8 s, or it idles out mid-sentence.
 */
import WebSocket from 'ws';
import {
  NotConfiguredError,
  ProtocolUnverifiedError,
  type SttEvent,
  type SttSession,
  type StartOptions,
} from '../types';
import {
  buildHeaders,
  buildWsUrl,
  decode,
  encodeAudio,
  encodeControl,
  isVerified,
  loadProtocol,
} from './protocol';

const CONNECT_TIMEOUT_MS = 8000;
const FINAL_TIMEOUT_MS = 5000;
const KEEPALIVE_INTERVAL_MS = 8000;

export class ClaudeVoiceAdapter implements SttSession {
  readonly id = 'claude-voice';
  readonly label = 'Claude (your subscription)';

  private listeners = new Set<(e: SttEvent) => void>();
  private protocol = loadProtocol();
  private ws: WebSocket | null = null;
  private finalText: string | null = null;
  private partial = '';
  private settle: ((text: string) => void) | null = null;
  private failure: Error | null = null;
  private framesSent = 0;
  private keepaliveTimer: NodeJS.Timeout | null = null;

  constructor(private readonly resolveToken: () => Promise<string>) {}

  async start(opts: StartOptions): Promise<void> {
    const p = this.protocol;
    if (!isVerified(p)) {
      throw new ProtocolUnverifiedError(
        'Claude voice',
        'Capture one real session with tools/ws-sniff.md, fill src/main/providers/claude/protocol.json, set "verified": true. Until then this provider stays off so it can never silently emit garbage.',
      );
    }

    this.reset();

    const token = await this.resolveToken();
    const ws = new WebSocket(buildWsUrl(p, token, opts), {
      headers: buildHeaders(p, token),
    });
    this.ws = ws;

    ws.on('message', (data) => this.handleMessage(data.toString()));
    ws.on('close', (code, reason) => this.handleClose(code, reason.toString()));
    ws.on('error', (err) => {
      this.failure = err;
      this.emit({ type: 'error', error: err, fatal: true });
      this.stopKeepalive();
      // Keep whatever the server already transcribed; only fail over when empty.
      const text = this.partial;
      this.settle?.(text);
      this.settle = null;
    });

    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Claude voice: no handshake within ${CONNECT_TIMEOUT_MS} ms`));
      }, CONNECT_TIMEOUT_MS);

      ws.once('open', () => {
        clearTimeout(timer);
        this.startKeepalive(ws);
        const startMsg = encodeControl(p, 'start');
        if (startMsg) ws.send(startMsg);
        resolve();
      });
      ws.once('unexpected-response', (_req, res) => {
        clearTimeout(timer);
        const hint =
          res.statusCode === 401 || res.statusCode === 403
            ? ' — the token was rejected. Run your own Claude Code client once to refresh it, then retry.'
            : '';
        reject(new Error(`Claude voice: handshake rejected with HTTP ${res.statusCode}${hint}`));
      });
      ws.once('error', (err) => {
        clearTimeout(timer);
        reject(err);
      });
    });
  }

  push(frame: Buffer): void {
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    try {
      ws.send(encodeAudio(this.protocol, frame));
      this.framesSent += 1;
    } catch (err) {
      this.emit({ type: 'error', error: err as Error, fatal: false });
    }
  }

  async commit(): Promise<string> {
    if (this.finalText !== null) {
      this.close();
      return this.finalText;
    }
    const ws = this.ws;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      this.close();
      throw this.failure ?? new NotConfiguredError('Claude voice: socket closed before commit');
    }

    const msg = encodeControl(this.protocol, 'commit');
    if (msg) ws.send(msg);

    const text = await new Promise<string>((resolve) => {
      this.settle = resolve;
      setTimeout(() => {
        // Whatever the server already sent is the best transcript we have.
        if (this.settle === resolve) {
          this.settle = null;
          resolve(this.finalText ?? this.partial);
        }
      }, FINAL_TIMEOUT_MS);
    });

    this.close();
    return text;
  }

  abort(): void {
    this.settle?.('');
    this.settle = null;
    this.close();
  }

  /** Diagnostics for the settings panel, never includes token material. */
  stats(): { framesSent: number; connected: boolean } {
    return { framesSent: this.framesSent, connected: this.ws?.readyState === WebSocket.OPEN };
  }

  on(listener: (e: SttEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private handleMessage(raw: string): void {
    const msg = decode(this.protocol, raw);
    if (msg.kind === 'interim' && msg.text) {
      this.partial = msg.text;
      this.emit({ type: 'interim', text: msg.text });
      return;
    }
    if (msg.kind === 'final') {
      // The final event carries no text: it promotes the running transcript.
      const text = msg.text ?? this.partial;
      this.finalText = text;
      if (text) this.emit({ type: 'final', text });
      this.settle?.(text);
      this.settle = null;
      return;
    }
    if (msg.kind === 'error') {
      this.failure = new Error(`Claude voice: ${msg.detail ?? 'unknown server error'}`);
      this.emit({ type: 'error', error: this.failure, fatal: true });
      this.settle?.('');
      this.settle = null;
    }
  }

  private handleClose(code: number, reason: string): void {
    this.stopKeepalive();
    if (this.settle && this.finalText === null) {
      const text = this.partial;
      if (!text && code !== 1000 && code !== 1005) {
        const detail = `closed before a final transcript (code ${code}${reason ? `: ${reason}` : ''})`;
        this.failure = new Error(`Claude voice: ${detail}`);
        this.emit({ type: 'error', error: this.failure, fatal: true });
      }
      this.settle(text);
      this.settle = null;
    }
  }

  private startKeepalive(ws: WebSocket): void {
    const msg = encodeControl(this.protocol, 'keepalive');
    if (!msg) return;
    ws.send(msg);
    this.keepaliveTimer = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send(msg);
    }, KEEPALIVE_INTERVAL_MS);
  }

  private stopKeepalive(): void {
    if (this.keepaliveTimer) clearInterval(this.keepaliveTimer);
    this.keepaliveTimer = null;
  }

  private close(): void {
    this.stopKeepalive();
    const ws = this.ws;
    this.ws = null;
    this.settle = null;
    if (!ws) return;
    ws.removeAllListeners();
    try {
      ws.close();
    } catch {
      /* already gone */
    }
  }

  private reset(): void {
    this.stopKeepalive();
    this.framesSent = 0;
    this.finalText = null;
    this.partial = '';
    this.failure = null;
    this.settle = null;
  }

  private emit(e: SttEvent): void {
    for (const l of this.listeners) l(e);
  }
}
