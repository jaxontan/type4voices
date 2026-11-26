import { app } from 'electron';
import { mkdir, readFile, rename, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';

export type InjectMode = 'paste' | 'clipboard-only';

export interface Settings {
  /** Tried left to right; the first one that yields a transcript wins. */
  providerOrder: string[];
  hotkey: string;
  cancelHotkey: string;
  /** Auto-commit after this much silence. Deliberately generous. */
  silenceMs: number;
  /** Hard ceiling per utterance; the audio buffer is bounded by this. */
  maxUtteranceMs: number;
  language: string;
  glossary: string[];
  /** Opt-in: read ~/.claude/.credentials.json instead of pasting a key. */
  useClaudeCodeCredentials: boolean;
  /**
   * Exchange the Claude Code refresh token when the access token has expired,
   * and write the rotated pair back so the user's own client keeps working.
   */
  autoRefreshClaudeToken: boolean;
  /** Google AI / Gemini API key for Gemini 3.5 Transcribe Live (local only). */
  geminiApiKey: string;
  injectMode: InjectMode;
  /** Two-pass cleanup through the same provider's model. */
  cleanupPass: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  providerOrder: ['mock', 'gemini-api', 'claude-voice'],
  hotkey: 'CommandOrControl+Shift+Space',
  cancelHotkey: 'CommandOrControl+Shift+E',
  silenceMs: 15000,
  maxUtteranceMs: 300000,
  language: 'en',
  glossary: [],
  useClaudeCodeCredentials: true,
  autoRefreshClaudeToken: true,
  geminiApiKey: '',
  injectMode: 'paste',
  cleanupPass: false,
};

export interface HistoryEntry {
  at: number;
  provider: string;
  ms: number;
  chars: number;
  text: string;
}

const HISTORY_CAP = 500;

function userData(...parts: string[]): string {
  return join(app.getPath('userData'), ...parts);
}

async function readJson<T>(path: string, fallback: T): Promise<T> {
  try {
    return { ...fallback, ...(JSON.parse(await readFile(path, 'utf8')) as object) } as T;
  } catch {
    return fallback;
  }
}

async function writeJson(path: string, value: unknown): Promise<void> {
  await mkdir(dirname(path), { recursive: true });
  const tmp = `${path}.tmp`;
  await writeFile(tmp, JSON.stringify(value, null, 2), 'utf8');
  await rename(tmp, path);
}

let cached: Settings | null = null;

export async function getSettings(): Promise<Settings> {
  if (cached) return cached;
  cached = await readJson<Settings>(userData('settings.json'), DEFAULT_SETTINGS);
  return cached;
}

export async function updateSettings(patch: Partial<Settings>): Promise<Settings> {
  const next = { ...(await getSettings()), ...patch };
  cached = next;
  await writeJson(userData('settings.json'), next);
  return next;
}

export async function listHistory(limit = 50): Promise<HistoryEntry[]> {
  try {
    const parsed = JSON.parse(await readFile(userData('history.json'), 'utf8'));
    if (!Array.isArray(parsed)) return [];
    return (parsed as HistoryEntry[]).slice(-limit).reverse();
  } catch {
    return [];
  }
}

export async function hasSettingsFile(): Promise<boolean> {
  try {
    await readFile(userData('settings.json'), 'utf8');
    return true;
  } catch {
    return false;
  }
}

export async function appendHistory(entry: HistoryEntry): Promise<void> {
  const path = userData('history.json');
  let all: HistoryEntry[] = [];
  try {
    const parsed = JSON.parse(await readFile(path, 'utf8'));
    if (Array.isArray(parsed)) all = parsed;
  } catch {
    /* first run */
  }
  all.push(entry);
  if (all.length > HISTORY_CAP) all = all.slice(-HISTORY_CAP);
  await writeJson(path, all);
}
