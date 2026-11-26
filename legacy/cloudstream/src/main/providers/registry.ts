/**
 * Provider catalog. This is the only place that knows which backends exist.
 *
 * Ordering and enablement live in user settings; the session layer walks this
 * order and fails over. Unbuilt providers are listed but throw a message that
 * says what is missing, so "why is Gemini greyed out" answers itself.
 */
import type { Settings } from '../store';
import { assertFresh, readClaudeCodeToken, refreshClaudeCodeToken, TokenExpiredError } from '../tokens';
import { ClaudeVoiceAdapter } from './claude';
import { GeminiLiveAdapter, resolveGeminiApiKey } from './gemini';
import { MockSession } from './mock';
import { NotConfiguredError, type ProviderHandle, type ProviderInfo, type SttSession } from './types';

type Factory = (settings: Settings) => SttSession;

const FACTORIES: Record<string, Factory> = {
  mock: () => new MockSession(),

  'claude-voice': (settings) => {
    if (!settings.useClaudeCodeCredentials) {
      throw new NotConfiguredError(
        'Claude voice needs credentials. Enable "use my Claude Code credentials" in settings.',
      );
    }
    return new ClaudeVoiceAdapter(async () => {
      const token = await readClaudeCodeToken();
      try {
        assertFresh(token);
        return token.token;
      } catch (err) {
        // Only an expired access token is worth an exchange; a missing file or
        // rejected login must surface as itself, not as a refresh failure.
        if (!(err instanceof TokenExpiredError) || !settings.autoRefreshClaudeToken) throw err;
        const refreshed = await refreshClaudeCodeToken();
        assertFresh(refreshed);
        return refreshed.token;
      }
    });
  },

  'gemini-api': (settings) => {
    const key = resolveGeminiApiKey(settings.geminiApiKey);
    if (!key) {
      throw new NotConfiguredError(
        'Gemini API needs an API key. Add it in settings or set GEMINI_API_KEY.',
      );
    }
    return new GeminiLiveAdapter(() => resolveGeminiApiKey(settings.geminiApiKey));
  },

  'gemini-web': () => {
    throw new NotConfiguredError(
      'Gemini web provider is not built and is last on the roadmap: it requires driving a browser session, breaks without notice, and is against Google\u2019s terms. Not shipped.',
    );
  },

  local: () => {
    throw new NotConfiguredError(
      'Local provider (whisper.cpp / Parakeet) is not built yet. It is the offline fallback: zero cost, zero terms risk, works on a plane.',
    );
  },
};

export const PROVIDERS: ProviderInfo[] = [
  {
    id: 'mock',
    label: 'Mock (offline)',
    kind: 'local',
    note: 'No network. Proves the capture → interim → inject loop while the real backends are unverified.',
  },
  {
    id: 'claude-voice',
    label: 'Claude (your subscription)',
    kind: 'unofficial',
    note: 'Streams to the voice endpoint your own subscription already pays for, using credentials your own client wrote. Unofficial and unsupported: fields can change without notice, and heavy automation may violate Anthropic\u2019s terms.',
  },
  {
    id: 'gemini-api',
    label: 'Gemini API (Transcribe Live)',
    kind: 'official',
    note: 'Sanctioned and stable. Streams to Gemini 3.5 Transcribe Live over BidiGenerateContent using your Gemini API key (settings or GEMINI_API_KEY), with custom vocabulary biasing.',
  },
  {
    id: 'gemini-web',
    label: 'Gemini web session',
    kind: 'unofficial',
    note: 'Would reuse a signed-in browser session. Against Google\u2019s terms and very brittle. Not built, and not recommended.',
  },
  {
    id: 'local',
    label: 'Local (whisper.cpp)',
    kind: 'local',
    note: 'Runs on your machine. Offline, free, no terms risk. The honest answer to "unlimited". Not built yet.',
  },
];

export function chainFor(settings: Settings): ProviderHandle[] {
  const order = settings.providerOrder?.length ? settings.providerOrder : ['mock'];
  const seen = new Set<string>();

  return order
    .filter((id) => {
      if (!FACTORIES[id] || seen.has(id)) return false;
      seen.add(id);
      return true;
    })
    .map((id) => {
      const info = PROVIDERS.find((p) => p.id === id)!;
      return {
        info,
        create: () => FACTORIES[id]!(settings),
      };
    });
}
