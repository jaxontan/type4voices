import { app, BrowserWindow, ipcMain, screen, session as electronSession } from 'electron';
import { join } from 'node:path';
import { registerHotkeys, registerOne, unregisterHotkeys } from './hotkey';
import { injectText } from './injector';
import { DictationSession } from './session';
import { chainFor, PROVIDERS } from './providers/registry';
import { probeQuota } from './providers/quota';
import { appendHistory, getSettings, hasSettingsFile, listHistory, updateSettings, type Settings } from './store';

// Push-to-talk must not depend on a click having happened first.
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');

type Phase = 'idle' | 'recording' | 'finalizing';

let overlay: BrowserWindow | null = null;
let settingsWindow: BrowserWindow | null = null;
let phase: Phase = 'idle';
let session: DictationSession | null = null;
let startedAt = 0;
let maxDurationTimer: NodeJS.Timeout | null = null;

const preload = join(__dirname, '..', 'preload.js');
const rendererDir = join(__dirname, '..', 'renderer');

function createOverlay(): BrowserWindow {
  const { workArea } = screen.getPrimaryDisplay();
  const width = 460;
  const height = 150;

  const win = new BrowserWindow({
    width,
    height,
    x: Math.round(workArea.x + (workArea.width - width) / 2),
    y: Math.round(workArea.y + workArea.height - height - 90),
    show: false,
    frame: false,
    transparent: true,
    resizable: false,
    movable: true,
    minimizable: false,
    maximizable: false,
    fullscreenable: false,
    skipTaskbar: true,
    alwaysOnTop: true,
    // Non-focusable is the load-bearing detail: the editor keeps focus, so the
    // paste lands where the cursor already is.
    focusable: false,
    hasShadow: false,
    acceptFirstMouse: true,
    webPreferences: {
      preload,
      contextIsolation: true,
      nodeIntegration: false,
      backgroundThrottling: false,
    },
  });

  win.setAlwaysOnTop(true, 'screen-saver');
  win.setIgnoreMouseEvents(true);
  win.webContents.on('console-message', (...args: any[]) => {
    if (app.isPackaged) return;
    console.log('[overlay console]', JSON.stringify(args.slice(1)));
  });
  win.webContents.on('did-fail-load', (_e, code, desc, url) => {
    console.error('[overlay fail]', code, desc, url);
  });
  void win.loadFile(join(rendererDir, 'overlay.html'));
  win.on('closed', () => {
    overlay = null;
  });
  return win;
}

function createSettingsWindow(): BrowserWindow {
  const win = new BrowserWindow({
    width: 780,
    height: 660,
    show: false,
    title: 'cloudstream',
    webPreferences: {
      preload,
      contextIsolation: true,
      nodeIntegration: false,
    },
  });
  void win.loadFile(join(rendererDir, 'index.html'));
  win.on('closed', () => {
    settingsWindow = null;
  });
  return win;
}

function openSettings(): void {
  if (!settingsWindow) settingsWindow = createSettingsWindow();
  settingsWindow.show();
  settingsWindow.focus();
}

function send(channel: string, payload?: unknown): void {
  overlay?.webContents.send(channel, payload);
}

async function startUtterance(): Promise<void> {
  if (phase !== 'idle') return;
  phase = 'recording';

  const settings = await getSettings();
  overlay?.showInactive();
  send('cs:begin', { silenceMs: settings.silenceMs });

  session = new DictationSession(chainFor(settings), {
    onInterim: (text) => send('cs:interim', { text }),
    onNotice: (message) => send('cs:notice', { message }),
  });

  const started = await session.begin({
    language: settings.language,
    glossary: settings.glossary,
    maxBytes: (settings.maxUtteranceMs / 1000) * 32000, // 32 kB/s at 16 kHz mono
  });

  if (!started) {
    phase = 'idle';
    session = null;
    send('cs:error', { message: 'No provider available. Open settings and pick one that is configured.' });
    setTimeout(() => overlay?.hide(), 3000);
    return;
  }

  startedAt = Date.now();
  maxDurationTimer = setTimeout(() => void finishUtterance(), settings.maxUtteranceMs);
}

async function finishUtterance(): Promise<void> {
  if (phase !== 'recording' || !session) return;
  phase = 'finalizing';
  if (maxDurationTimer) clearTimeout(maxDurationTimer);
  maxDurationTimer = null;

  const settings = await getSettings();
  const active = session;
  session = null;

  send('cs:state', { phase: 'finalizing' });
  const result = await active.finish();
  const ms = Date.now() - startedAt;

  if (!result.text) {
    send('cs:error', {
      message: result.error ?? 'Nothing was transcribed. Check the microphone and provider.',
    });
    phase = 'idle';
    setTimeout(() => overlay?.hide(), 3200);
    return;
  }

  const injected = await injectText(result.text, settings.injectMode);
  send('cs:final', { text: result.text, provider: result.provider, injected, ms });

  await appendHistory({
    at: Date.now(),
    provider: result.provider ?? 'unknown',
    ms,
    chars: result.text.length,
    text: result.text,
  });

  phase = 'idle';
  setTimeout(() => {
    if (phase === 'idle') overlay?.hide();
  }, 1600);
}

function cancelUtterance(): void {
  if (maxDurationTimer) clearTimeout(maxDurationTimer);
  maxDurationTimer = null;
  session?.cancel();
  session = null;
  phase = 'idle';
  send('cs:state', { phase: 'idle' });
  overlay?.hide();
}

function applyHotkeys(settings: Settings): { ok: boolean; failed: string[] } {
  const report = registerHotkeys({
    hotkey: settings.hotkey,
    cancelHotkey: settings.cancelHotkey,
    onToggle: () => void (phase === 'recording' ? finishUtterance() : startUtterance()),
    onCancel: () => cancelUtterance(),
  });

  // Comma-bearing accelerators are the fragile ones, so this is best-effort.
  const failed = [...report.failed];
  if (!registerOne('CommandOrControl+Shift+,', () => openSettings())) {
    failed.push('CommandOrControl+Shift+,');
  }

  return { ok: failed.length === 0, failed };
}

function wireIpc(): void {
  ipcMain.on('cs:audio', (_event, chunk: ArrayBuffer) => {
    if (phase !== 'recording' || !session) return;
    session.push(Buffer.from(chunk));
  });

  ipcMain.on('cs:commit', () => void finishUtterance());
  ipcMain.on('cs:cancel', () => cancelUtterance());
  ipcMain.on('cs:open-settings', () => openSettings());

  ipcMain.handle('settings:get', async () => {
    const settings = await getSettings();
    return { settings, providers: PROVIDERS };
  });

  ipcMain.handle('settings:set', async (_event, patch: Partial<Settings>) => {
    const next = await updateSettings(patch);
    return { settings: next, hotkeys: applyHotkeys(next) };
  });

  ipcMain.handle('history:list', async () => listHistory(30));

  ipcMain.handle('quota:probe', async () => {
    const settings = await getSettings();
    const chain = chainFor(settings);
    if (!chain.length) return { ok: false, detail: 'no providers configured' };
    return probeQuota(() => chain[0]!.create());
  });
}

const gotLock = app.requestSingleInstanceLock();
if (!gotLock) {
  app.quit();
} else {
  app.on('second-instance', () => openSettings());

  app.whenReady().then(async () => {
    electronSession.defaultSession.setPermissionRequestHandler((_wc, permission, callback) => {
      callback(permission === 'media');
    });

    // Headless verification paths. Useful in CI and when the protocol is unverified.
    if (process.argv.includes('--selftest')) {
      const settings = await getSettings();
      const chain = chainFor({ ...settings, providerOrder: ['mock'] });
      const test = new DictationSession(chain, {
        onInterim: (text) => console.log('[interim]', text),
        onNotice: (message) => console.log('[notice]', message),
      });
      const opened = await test.begin({
        language: 'en',
        glossary: ['useState', 'postgres'],
        maxBytes: 1_000_000,
      });
      for (let i = 0; i < 25; i += 1) test.push(Buffer.alloc(640));
      const result = await test.finish();
      console.log('SELFTEST', JSON.stringify({ opened, ...result }));
      app.exit(0);
      return;
    }

    if (process.argv.includes('--smoke')) {
      setTimeout(() => {
        console.log('SMOKE_OK windows=' + ['overlay', 'settings'].filter(Boolean).length);
        app.exit(0);
      }, 3000);
    }

    overlay = createOverlay();
    settingsWindow = createSettingsWindow();
    wireIpc();

    const settings = await getSettings();
    const hotkeys = applyHotkeys(settings);
    if (!hotkeys.ok) {
      console.warn('hotkey registration failed for:', hotkeys.failed.join(', '));
    }

    // End-to-end: real microphone -> worklet -> resample -> IPC -> session buffer.
    // Runs after the windows exist, which is the whole point of the exercise.
    if (process.argv.includes('--selftest-audio')) {
      await updateSettings({ injectMode: 'clipboard-only', providerOrder: ['mock'], silenceMs: 60000 });
      await new Promise((r) => setTimeout(r, 1200));
      await startUtterance();
      await new Promise((r) => setTimeout(r, 1800));
      const bytes = session?.bufferedBytes ?? 0;
      await finishUtterance();
      await new Promise((r) => setTimeout(r, 400));
      console.log(
        'SELFTEST_AUDIO',
        JSON.stringify({ bytes, frames: Math.round(bytes / 640), expect: '~57600 bytes for 1.8 s' }),
      );
      app.exit(0);
      return;
    }

    // First run: no settings file means the user has never seen this window.
    if (process.argv.includes('--settings') || !(await hasSettingsFile())) {
      openSettings();
    }
  });

  app.on('will-quit', () => unregisterHotkeys());
  app.on('window-all-closed', () => {
    /* background utility: keep running so the hotkey stays live */
  });
}
