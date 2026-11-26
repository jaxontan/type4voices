/**
 * Credential acquisition. Rules, non-negotiable:
 *   1. Tokens never leave this machine and are never written to our own files.
 *   2. Tokens are never logged. Use mask() if you must print anything.
 *   3. We do not mint tokens from an invented contract. Refresh is gated by a
 *      user setting, posts to the token endpoint observed in the installed
 *      Claude Code client (platform.claude.com/v1/oauth/token, client id
 *      9d1c250a…), and writes the rotated token back where that client
 *      expects it.
 *
 * Rule 3 exists because refresh tokens rotate: if we refreshed and kept the
 * result to ourselves, the old refresh token in ~/.claude/.credentials.json
 * would eventually be rejected and the user's own client would be signed out.
 * So the write-back is a compare-and-swap on the refresh token we used — if the
 * client refreshed first, we leave its file alone and fail over to re-reading.
 */
import { readFile, rename, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { NotConfiguredError } from './providers/types';

/** Observed in the installed Claude Code client, not guessed. */
const CLAUDE_CLIENT_ID = '9d1c250a-e61b-44d9-88ed-5944d1962f5e';
const CLAUDE_TOKEN_URL = 'https://platform.claude.com/v1/oauth/token';
const REFRESH_TIMEOUT_MS = 15_000;
const DEFAULT_SCOPES = [
  'user:file_upload',
  'user:inference',
  'user:mcp_servers',
  'user:plugins',
  'user:profile',
  'user:sessions:claude_code',
];

export class TokenExpiredError extends Error {
  readonly code = 'TOKEN_EXPIRED';
  constructor(message: string) {
    super(message);
    this.name = 'TokenExpiredError';
  }
}

export interface ClaudeToken {
  token: string;
  expiresAt: number | null;
  subscriptionType: string | null;
  scopes: string[];
}

interface ClaudeOauth {
  accessToken?: string;
  refreshToken?: string;
  expiresAt?: number;
  refreshTokenExpiresAt?: number;
  scopes?: string[];
  subscriptionType?: string;
  clientId?: string;
}

interface CredentialsFile {
  claudeAiOauth?: ClaudeOauth;
  [key: string]: unknown;
}

interface TokenResponse {
  access_token?: string;
  refresh_token?: string;
  expires_in?: number;
  refresh_token_expires_in?: number;
  scope?: string;
  error?: string;
  error_description?: string;
}

export function mask(secret: string): string {
  if (secret.length <= 12) return '***';
  return `${secret.slice(0, 6)}…${secret.slice(-4)} (${secret.length} ch)`;
}

export function credentialsPath(): string {
  return join(homedir(), '.claude', '.credentials.json');
}

let memo: { value: ClaudeToken; readAt: number } | null = null;
const MEMO_TTL_MS = 30_000;

function toToken(oauth: ClaudeOauth): ClaudeToken {
  return {
    token: oauth.accessToken!,
    expiresAt: typeof oauth.expiresAt === 'number' ? oauth.expiresAt : null,
    subscriptionType: oauth.subscriptionType ?? null,
    scopes: Array.isArray(oauth.scopes) ? oauth.scopes : [],
  };
}

async function readCredentialsFile(): Promise<CredentialsFile> {
  let raw: string;
  try {
    raw = await readFile(credentialsPath(), 'utf8');
  } catch {
    throw new NotConfiguredError(
      `No Claude Code credentials at ${credentialsPath()}. Sign in with your own Claude Code client first, or paste a token in settings.`,
    );
  }
  try {
    return JSON.parse(raw) as CredentialsFile;
  } catch {
    throw new NotConfiguredError('Claude Code credentials file is not valid JSON.');
  }
}

/**
 * Reads the token file that the user's own Claude Code client already wrote.
 * This is the "use the account you are already signed into" path: it is opt-in,
 * local, and read-only.
 */
export async function readClaudeCodeToken(): Promise<ClaudeToken> {
  if (memo && Date.now() - memo.readAt < MEMO_TTL_MS) return memo.value;

  const parsed = await readCredentialsFile();
  const oauth = parsed.claudeAiOauth;
  if (!oauth?.accessToken) {
    throw new NotConfiguredError(
      'Claude Code credentials file has no claudeAiOauth.accessToken. Sign in with your own client.',
    );
  }

  const value = toToken(oauth);
  memo = { value, readAt: Date.now() };
  return value;
}

/** Throws a message the user can act on rather than a 401 later. */
export function assertFresh(token: ClaudeToken, skewMs = 60_000): void {
  if (token.expiresAt === null) return;
  if (Date.now() + skewMs >= token.expiresAt) {
    const mins = Math.round((token.expiresAt - Date.now()) / 60000);
    throw new TokenExpiredError(
      `Claude token expired (${mins} min). Run your Claude Code client once to refresh it, then retry.`,
    );
  }
}

async function writeBack(path: string, usedRefreshToken: string, oauth: ClaudeOauth): Promise<boolean> {
  let current: CredentialsFile;
  try {
    current = JSON.parse(await readFile(path, 'utf8')) as CredentialsFile;
  } catch {
    return false;
  }

  // If the user's own client refreshed while we were talking to the server, its
  // result wins. Never clobber a login we did not rotate.
  if (current.claudeAiOauth?.refreshToken !== usedRefreshToken) return false;

  const next: CredentialsFile = {
    ...current,
    claudeAiOauth: { ...current.claudeAiOauth, ...oauth },
  };
  const tmp = `${path}.cloudstream.tmp`;
  await writeFile(tmp, JSON.stringify(next, null, 2), { encoding: 'utf8', mode: 0o600 });
  await rename(tmp, path);
  return true;
}

let inFlight: Promise<ClaudeToken> | null = null;

/**
 * Exchanges the refresh token in the user's credentials file for a fresh access
 * token, using the exact request the installed client makes. Rotated tokens are
 * written back so the user's own client keeps working. Callers must gate this
 * behind an explicit setting: it is the only place we ever mint credentials.
 */
export function refreshClaudeCodeToken(): Promise<ClaudeToken> {
  // Concurrent callers (session + probe) share one exchange: refresh tokens are
  // single-use-ish, and two parallel exchanges would race the write-back.
  inFlight ??= doRefresh().finally(() => {
    inFlight = null;
  });
  return inFlight;
}

async function doRefresh(): Promise<ClaudeToken> {
  const path = credentialsPath();
  const parsed = await readCredentialsFile();
  const oauth = parsed.claudeAiOauth;
  if (!oauth?.refreshToken) {
    throw new NotConfiguredError(
      'Claude Code credentials file has no refresh token. Run your Claude Code client to sign in again.',
    );
  }
  if (typeof oauth.refreshTokenExpiresAt === 'number' && Date.now() >= oauth.refreshTokenExpiresAt) {
    throw new TokenExpiredError(
      'Claude refresh token has expired. Run your Claude Code client to sign in again.',
    );
  }

  const usedRefreshToken = oauth.refreshToken;
  const scopes = oauth.scopes?.length ? oauth.scopes : DEFAULT_SCOPES;
  const body = {
    grant_type: 'refresh_token',
    refresh_token: usedRefreshToken,
    client_id: oauth.clientId ?? CLAUDE_CLIENT_ID,
    scope: scopes.join(' '),
  };

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REFRESH_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(CLAUDE_TOKEN_URL, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    const detail = (err as Error).name === 'AbortError' ? 'timed out' : (err as Error).message;
    throw new Error(`Claude token refresh failed: ${detail}`);
  } finally {
    clearTimeout(timer);
  }

  const text = await res.text();
  let data: TokenResponse = {};
  try {
    data = JSON.parse(text) as TokenResponse;
  } catch {
    /* non-JSON error page; status below carries the message */
  }

  if (!res.ok) {
    const detail = data.error_description || data.error || `HTTP ${res.status}`;
    if (res.status === 400 || res.status === 401) {
      throw new TokenExpiredError(
        `Claude refresh token was rejected (${detail}). Run your Claude Code client to sign in again.`,
      );
    }
    throw new Error(`Claude token refresh failed: ${detail}`);
  }

  if (!data.access_token || typeof data.expires_in !== 'number') {
    throw new Error('Claude token refresh returned no usable access token.');
  }

  const now = Date.now();
  const refreshToken = data.refresh_token || usedRefreshToken;
  const refreshTokenExpiresAt =
    typeof data.refresh_token_expires_in === 'number'
      ? now + data.refresh_token_expires_in * 1000
      : oauth.refreshTokenExpiresAt;

  const updated: ClaudeOauth = {
    ...oauth,
    accessToken: data.access_token,
    refreshToken,
    expiresAt: now + data.expires_in * 1000,
    ...(refreshTokenExpiresAt !== undefined && { refreshTokenExpiresAt }),
    ...(data.scope && { scopes: data.scope.split(' ').filter(Boolean) }),
  };

  await writeBack(path, usedRefreshToken, updated).catch(() => false);

  const value: ClaudeToken = toToken(updated);
  memo = { value, readAt: Date.now() };
  return value;
}

export function forgetToken(): void {
  memo = null;
}
