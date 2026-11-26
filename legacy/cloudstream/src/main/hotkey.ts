import { globalShortcut } from 'electron';

export interface HotkeyReport {
  hotkey: string;
  cancelHotkey: string;
  ok: boolean;
  failed: string[];
}

/**
 * Tap-to-toggle rather than true hold-to-talk.
 *
 * Electron's globalShortcut has no key-up event, so "hold" needs a low-level
 * keyboard hook (uiohook-napi) and a native dependency. Toggle gives the same
 * interaction with the 15s silence cutoff doing the work of "I am done talking",
 * and it is what we ship first.
 */
export function registerHotkeys(opts: {
  hotkey: string;
  cancelHotkey: string;
  onToggle: () => void;
  onCancel: () => void;
}): HotkeyReport {
  globalShortcut.unregisterAll();
  const failed: string[] = [];

  for (const [accel, handler] of [
    [opts.hotkey, opts.onToggle],
    [opts.cancelHotkey, opts.onCancel],
  ] as const) {
    if (!accel) continue;
    if (!registerOne(accel, handler)) failed.push(accel);
  }

  return { hotkey: opts.hotkey, cancelHotkey: opts.cancelHotkey, ok: failed.length === 0, failed };
}

/**
 * Never throws. A rejected accelerator (bad key name, already owned by another
 * app) must degrade to "this one shortcut is dead", not to "the tray app failed
 * to start".
 */
export function registerOne(accel: string, handler: () => void): boolean {
  try {
    return globalShortcut.register(accel, handler);
  } catch {
    return false;
  }
}

export function unregisterHotkeys(): void {
  globalShortcut.unregisterAll();
}
