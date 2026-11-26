/**
 * "Unlimited" is a hypothesis, so we measure it instead of trusting it.
 *
 * Sends one second of silence through a provider and reports whether the round
 * trip completed. Silent audio legitimately transcribes to "", so success is
 * "no fatal error", not "non-empty text".
 */
import type { SttSession } from './types';

export interface QuotaProbe {
  provider: string;
  ok: boolean;
  ms: number;
  frames: number;
  detail: string;
}

export async function probeQuota(makeSession: () => SttSession): Promise<QuotaProbe> {
  const started = Date.now();
  let session: SttSession | null = null;
  let failure: string | null = null;
  let off = () => {};

  const silence = Buffer.alloc(640, 0);
  const frames = 50; // 1 s at 20 ms

  try {
    session = makeSession();
    off = session.on((e) => {
      if (e.type === 'error' && e.fatal) failure ??= e.error.message;
    });
    await session.start({});
    for (let i = 0; i < frames; i += 1) session.push(silence);
    const text = await session.commit();
    return {
      provider: session.id,
      ok: failure === null,
      ms: Date.now() - started,
      frames,
      detail: failure ?? (text ? 'round trip ok (unexpected transcript from silence)' : 'round trip ok'),
    };
  } catch (err) {
    return {
      provider: session?.id ?? 'unknown',
      ok: false,
      ms: Date.now() - started,
      frames,
      detail: (err as Error).message,
    };
  } finally {
    off();
    session?.abort();
  }
}
