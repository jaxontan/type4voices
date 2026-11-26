/**
 * Config-driven wire protocol.
 *
 * The point of this file is that correcting the schema never requires a code
 * change: capture the real frames, fill protocol.json, restart. If a field is
 * missing we fail loudly instead of sending something plausible and silently
 * getting garbage back.
 *
 * The Claude Code voice endpoint is event-typed rather than field-path-typed:
 * text arrives as `TranscriptText` with a `data` string, and the final is an
 * empty `TranscriptEndpoint` that promotes the running transcript. So decoding
 * keys off the message `type`, not off a "finalPath".
 */
import rawConfig from './protocol.json';

export interface ClaudeProtocol {
  verified: boolean;
  capturedAt: number | null;
  capturedFrom: string | null;
  transport: {
    wsUrl: string;
    query: Record<string, string>;
    /** Query param repeated once per glossary term; empty string disables biasing. */
    keytermsParam: string;
    headers: Record<string, string>;
  };
  audio: {
    binary: boolean;
    message: Record<string, unknown> | null;
  };
  control: {
    keepalive: Record<string, unknown> | null;
    start: Record<string, unknown> | null;
    commit: Record<string, unknown> | null;
  };
  receive: {
    /** Dot path to the message kind, e.g. "type". */
    typePath: string;
    /** Dot path to the text payload, e.g. "data". */
    textPath: string;
    interimTypes: string[];
    /** Event that ends the utterance. Its text is the last interim, not a field. */
    finalTypes: string[];
    errorTypes: string[];
    /** Tried in order; the first string found becomes the error detail. */
    errorPaths: string[];
  };
}

export function loadProtocol(): ClaudeProtocol {
  return rawConfig as unknown as ClaudeProtocol;
}

export function isVerified(p: ClaudeProtocol): boolean {
  return (
    p.verified === true &&
    typeof p.transport.wsUrl === 'string' &&
    p.transport.wsUrl.length > 0 &&
    Array.isArray(p.receive.finalTypes) &&
    p.receive.finalTypes.length > 0
  );
}

export interface UrlOptions {
  language?: string;
  keyterms?: string[];
}

function substitute(value: unknown, token: string, language: string): unknown {
  if (typeof value === 'string') {
    return value.replaceAll('{accessToken}', token).replaceAll('{language}', language);
  }
  if (Array.isArray(value)) return value.map((v) => substitute(v, token, language));
  if (value && typeof value === 'object') {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = substitute(v, token, language);
    return out;
  }
  return value;
}

const MAX_KEYTERMS = 50;
const MAX_KEYTERM_LENGTH = 64;

export function buildWsUrl(p: ClaudeProtocol, accessToken: string, opts: UrlOptions = {}): string {
  const url = new URL(p.transport.wsUrl);
  const language = opts.language || 'en';
  for (const [key, value] of Object.entries(p.transport.query ?? {})) {
    url.searchParams.set(key, String(substitute(value, accessToken, language)));
  }

  const param = p.transport.keytermsParam;
  if (param) {
    const seen = new Set<string>();
    for (const term of (opts.keyterms ?? []).slice(0, MAX_KEYTERMS)) {
      const clean = term.trim().slice(0, MAX_KEYTERM_LENGTH);
      if (!clean || seen.has(clean)) continue;
      seen.add(clean);
      url.searchParams.append(param, clean);
    }
  }
  return url.toString();
}

export function buildHeaders(p: ClaudeProtocol, accessToken: string): Record<string, string> {
  return substitute(p.transport.headers ?? {}, accessToken, 'en') as Record<string, string>;
}

export function encodeAudio(p: ClaudeProtocol, frame: Buffer): Buffer | string {
  if (p.audio.binary) return frame;
  if (!p.audio.message) {
    throw new Error('protocol.json: audio.message is null (binary framing also disabled)');
  }
  return JSON.stringify(substitute(p.audio.message, frame.toString('base64'), 'en'));
}

export function encodeControl(
  p: ClaudeProtocol,
  which: 'keepalive' | 'start' | 'commit',
): string | null {
  const msg = p.control[which];
  return msg ? JSON.stringify(msg) : null;
}

export function getPath(obj: unknown, path: string): unknown {
  if (!path) return undefined;
  let cur: unknown = obj;
  for (const part of path.split('.')) {
    if (cur === null || typeof cur !== 'object') return undefined;
    cur = (cur as Record<string, unknown>)[part];
  }
  return cur;
}

export interface DecodedMessage {
  kind: 'interim' | 'final' | 'error' | 'ignore';
  text?: string;
  detail?: string;
}

export function decode(p: ClaudeProtocol, raw: string): DecodedMessage {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { kind: 'ignore' };
  }

  const type = getPath(parsed, p.receive.typePath);
  if (typeof type !== 'string') return { kind: 'ignore' };

  if (p.receive.errorTypes.includes(type)) {
    for (const path of p.receive.errorPaths) {
      const value = getPath(parsed, path);
      if (typeof value === 'string' && value.length > 0) return { kind: 'error', detail: value };
    }
    return { kind: 'error', detail: JSON.stringify(parsed) };
  }

  if (p.receive.finalTypes.includes(type)) return { kind: 'final' };

  if (p.receive.interimTypes.includes(type)) {
    const text = getPath(parsed, p.receive.textPath);
    if (typeof text === 'string' && text.length > 0) return { kind: 'interim', text };
    return { kind: 'ignore' };
  }

  return { kind: 'ignore' };
}
