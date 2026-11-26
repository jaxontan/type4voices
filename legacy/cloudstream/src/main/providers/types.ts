/**
 * The one interface every speech backend implements.
 *
 * Audio capture is provider-agnostic: the renderer always hands up 16 kHz mono
 * pcm_s16le frames of 20 ms. Only the transport differs, so each adapter is a
 * thin translation layer over this contract.
 */

export interface AudioFormat {
  encoding: 'pcm_s16le';
  sampleRate: number;
  channels: number;
}

export interface StartOptions {
  language?: string;
  /** Personal vocabulary: identifiers, brand names, coworkers. Used for biasing if
   *  the provider supports it, otherwise fed to the post-pass corrector. */
  glossary?: string[];
}

export type SttEvent =
  | { type: 'interim'; text: string }
  | { type: 'final'; text: string }
  | { type: 'error'; error: Error; fatal: boolean };

export interface SttSession {
  readonly id: string;
  readonly label: string;
  start(opts: StartOptions): Promise<void>;
  /** One 20 ms frame. Never throws; adapters must be tolerant of a closed socket. */
  push(frame: Buffer): void;
  /** Key released (or silence timeout). Resolves with the final transcript. */
  commit(): Promise<string>;
  abort(): void;
  on(listener: (event: SttEvent) => void): () => void;
}

export type ProviderKind = 'official' | 'unofficial' | 'local';

export interface ProviderInfo {
  id: string;
  label: string;
  kind: ProviderKind;
  /** Shown in settings, in plain language. */
  note: string;
}

export interface ProviderHandle {
  info: ProviderInfo;
  create(): SttSession;
}

export const AUDIO: AudioFormat = {
  encoding: 'pcm_s16le',
  sampleRate: 16000,
  channels: 1,
};

/** 20 ms at 16 kHz mono = 320 samples = 640 bytes. */
export const FRAME_BYTES = (AUDIO.sampleRate / 1000) * 20 * 2;

export class ProtocolUnverifiedError extends Error {
  readonly code = 'PROTOCOL_UNVERIFIED';
  constructor(provider: string, hint: string) {
    super(`${provider} wire protocol is not verified yet. ${hint}`);
    this.name = 'ProtocolUnverifiedError';
  }
}

export class NotConfiguredError extends Error {
  readonly code = 'NOT_CONFIGURED';
  constructor(message: string) {
    super(message);
    this.name = 'NotConfiguredError';
  }
}
