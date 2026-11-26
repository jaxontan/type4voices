/**
 * Text delivery into whatever app currently has focus.
 *
 * The overlay window is created focusable:false so the user's editor or terminal
 * keeps focus the whole time. That means delivery is a paste at the OS level, not
 * a synthetic DOM event we control.
 *
 * Strategy: put the transcript on the clipboard, synthesize Ctrl/Cmd+V, then put
 * the previous clipboard contents back — but only if nothing else has taken the
 * clipboard in the meantime, so we never clobber something the user copied.
 */
import { clipboard } from 'electron';
import { execFile } from 'node:child_process';
import type { InjectMode } from './store';

const WIN_PASTE = `
Add-Type -AssemblyName System.Windows.Forms
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class CsKeys {
  [DllImport("user32.dll")]
  public static extern void keybd_event(byte bVk, byte bScan, uint dwFlags, UIntPtr dwExtraInfo);
}
"@
[CsKeys]::keybd_event(0x11, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 12
[CsKeys]::keybd_event(0x56, 0, 0, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 12
[CsKeys]::keybd_event(0x56, 0, 2, [UIntPtr]::Zero)
Start-Sleep -Milliseconds 12
[CsKeys]::keybd_event(0x11, 0, 2, [UIntPtr]::Zero)
`;

const MAC_PASTE = `
tell application "System Events" to keystroke "v" using command down
`;

function run(cmd: string, args: string[]): Promise<void> {
  return new Promise((resolve, reject) => {
    execFile(cmd, args, { windowsHide: true, timeout: 5000 }, (err) =>
      err ? reject(err) : resolve(),
    );
  });
}

async function synthesizePaste(): Promise<void> {
  if (process.platform === 'win32') {
    await run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', WIN_PASTE]);
    return;
  }
  if (process.platform === 'darwin') {
    await run('osascript', ['-e', MAC_PASTE]);
    return;
  }
  await run('xdotool', ['key', '--clearmodifiers', 'ctrl+v']);
}

/** Dictation should insert text, not submit it. Drop a single trailing newline. */
function normalize(text: string): string {
  return text.replace(/\r\n/g, '\n').replace(/\n+$/, '').trim();
}

export async function injectText(text: string, mode: InjectMode): Promise<boolean> {
  const payload = normalize(text);
  if (!payload) return false;

  const previous = await clipboard.readText();
  await clipboard.writeText(payload);

  if (mode === 'clipboard-only') return true;

  try {
    await synthesizePaste();
  } catch {
    // Clipboard now holds the transcript, so the user can paste manually.
    return false;
  }

  setTimeout(() => {
    void (async () => {
      if ((await clipboard.readText()) === payload) await clipboard.writeText(previous);
    })();
  }, 500);

  return true;
}
