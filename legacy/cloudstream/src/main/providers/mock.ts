/**
 * A provider that never touches the network.
 *
 * Its job is to make the entire UX testable today: dim interim text, finalize on
 * commit, injection, history. It deliberately "mishears" an identifier so the
 * glossary corrector has something to fix.
 */
import type { SttEvent, SttSession, StartOptions } from './types';

const SCRIPT = [
  'hey can you',
  'hey can you refactor the',
  'hey can you refactor the auth handler',
  'hey can you refactor the auth handler to call use state',
  'hey can you refactor the auth handler to call use state and the post gress client',
];

export class MockSession implements SttSession {
  readonly id = 'mock';
  readonly label = 'Mock (offline, for UI work)';

  private listeners = new Set<(e: SttEvent) => void>();
  private timer: NodeJS.Timeout | null = null;
  private step = 0;
  private committed = false;

  async start(_opts: StartOptions): Promise<void> {
    this.timer = setInterval(() => {
      if (this.committed) return;
      const text = SCRIPT[this.step];
      if (text === undefined) return;
      this.step += 1;
      this.emit({ type: 'interim', text });
    }, 350);
  }

  push(_frame: Buffer): void {
    /* nothing to do: the mock ignores audio */
  }

  async commit(): Promise<string> {
    this.committed = true;
    this.stopTimer();
    const final = SCRIPT[SCRIPT.length - 1] ?? '';
    this.emit({ type: 'final', text: final });
    return final;
  }

  abort(): void {
    this.committed = true;
    this.stopTimer();
  }

  on(listener: (e: SttEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private emit(e: SttEvent): void {
    for (const l of this.listeners) l(e);
  }

  private stopTimer(): void {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }
}
