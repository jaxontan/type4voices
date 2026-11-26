/**
 * Utterance orchestration: buffer, stream to the primary provider, fail over.
 *
 * The buffer is the whole trick. Every frame is kept until the transcript is
 * final, so a dead socket at second four costs latency, not the sentence: we
 * replay the identical audio into the next provider.
 */
import type { ProviderHandle, SttSession } from './providers/types';

export interface BeginOptions {
  language: string;
  glossary: string[];
  maxBytes: number;
}

export interface FinishResult {
  text: string;
  provider: string | null;
  tried: string[];
  error: string | null;
}

interface LiveAttempt {
  index: number;
  handle: ProviderHandle;
  session: SttSession;
  fatal: Error | null;
  off: () => void;
}

export interface SessionDeps {
  onInterim: (text: string) => void;
  onNotice: (message: string) => void;
}

export class DictationSession {
  private chunks: Buffer[] = [];
  private bytes = 0;
  private live: LiveAttempt | null = null;
  private exhausted = false;
  private maxBytes = Infinity;
  private lastOpts: { language?: string; glossary?: string[] } = {};

  constructor(
    private readonly chain: ProviderHandle[],
    private readonly deps: SessionDeps,
  ) {}

  get bufferedBytes(): number {
    return this.bytes;
  }

  /** True once the utterance hits the ceiling; the caller should commit now. */
  get full(): boolean {
    return this.bytes >= this.maxBytes;
  }

  get providerId(): string | null {
    return this.live?.handle.info.id ?? null;
  }

  async begin(opts: BeginOptions): Promise<boolean> {
    this.reset();
    this.maxBytes = opts.maxBytes;
    this.lastOpts = { language: opts.language, glossary: opts.glossary };

    for (let i = 0; i < this.chain.length; i += 1) {
      const handle = this.chain[i]!;
      try {
        const session = handle.create();
        const attempt: LiveAttempt = {
          index: i,
          handle,
          session,
          fatal: null,
          off: () => {},
        };

        attempt.off = session.on((event) => {
          if (event.type === 'interim') {
            this.deps.onInterim(event.text);
          } else if (event.type === 'error') {
            attempt.fatal = event.error;
            if (event.fatal) {
              this.deps.onNotice(`${handle.info.label}: ${event.error.message}`);
            }
          }
        });

        await session.start({ language: opts.language, glossary: opts.glossary });
        this.live = attempt;
        this.bytes = 0;
        return true;
      } catch (err) {
        this.deps.onNotice(`${handle.info.label}: ${(err as Error).message}`);
      }
    }

    this.deps.onNotice('No speech provider is available. Add credentials or enable the mock provider.');
    return false;
  }

  push(frame: Buffer): void {
    if (this.exhausted) return;
    this.chunks.push(frame);
    this.bytes += frame.length;

    if (this.live) {
      try {
        this.live.session.push(frame);
      } catch (err) {
        this.live.fatal = err as Error;
      }
    }
  }

  async finish(): Promise<FinishResult> {
    const tried: string[] = [];
    let lastError: string | null = null;
    let lastIndex = -1;

    if (this.live) {
      const attempt = this.live;
      lastIndex = attempt.index;
      tried.push(attempt.handle.info.id);
      try {
        const text = await attempt.session.commit();
        attempt.off();
        if (text.trim() || !attempt.fatal) {
          this.live = null;
          return { text: text.trim(), provider: attempt.handle.info.id, tried, error: null };
        }
        lastError = attempt.fatal?.message ?? 'empty transcript';
      } catch (err) {
        lastError = (err as Error).message;
      }
      attempt.off();
      this.live = null;
    }

    // Failover: fresh session, replay every frame we kept, commit again.
    for (let i = lastIndex + 1; i < this.chain.length; i += 1) {
      const handle = this.chain[i]!;
      tried.push(handle.info.id);
      try {
        const session = handle.create();
        const off = session.on((event) => {
          if (event.type === 'interim') this.deps.onInterim(event.text);
        });
        await session.start(this.lastOpts);
        for (const chunk of this.chunks) session.push(chunk);
        const text = await session.commit();
        off();
        if (text.trim()) {
          return { text: text.trim(), provider: handle.info.id, tried, error: null };
        }
        lastError = 'empty transcript';
      } catch (err) {
        lastError = (err as Error).message;
        this.deps.onNotice(`${handle.info.label} fallback failed: ${lastError}`);
      }
    }

    return { text: '', provider: null, tried, error: lastError };
  }

  cancel(): void {
    this.exhausted = true;
    this.live?.off();
    try {
      this.live?.session.abort();
    } catch {
      /* ignore */
    }
    this.live = null;
    this.chunks = [];
    this.bytes = 0;
  }

  /** Discard audio; never written to disk unless the user opts in. */
  private reset(): void {
    this.chunks = [];
    this.bytes = 0;
    this.exhausted = false;
  }
}
