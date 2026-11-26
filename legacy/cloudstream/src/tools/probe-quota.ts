/**
 * CLI quota probe: `npm run probe`
 *
 * Sends one second of silence through the first configured provider and reports
 * whether the round trip completed. Run this before believing any claim about
 * "unlimited".
 */
import { app } from 'electron';
import { join } from 'node:path';
import { chainFor } from '../main/providers/registry';
import { probeQuota } from '../main/providers/quota';
import { getSettings } from '../main/store';

// Running `electron dist/tools/probe-quota.js` bypasses package.json so
// userData would otherwise default to %APPDATA%/Electron instead of cloudstream.
app.setName('cloudstream');
app.setPath('userData', join(app.getPath('appData'), 'cloudstream'));

app.whenReady().then(async () => {
  // A probe is a deliberate diagnostic: allow the one-time token exchange even
  // if the user has turned auto-refresh off in the app.
  const settings = { ...(await getSettings()), autoRefreshClaudeToken: true };
  const chain = chainFor(settings);

  if (!chain.length) {
    console.log('no providers configured; run the app once to create settings');
    app.exit(1);
    return;
  }

  // `npm run probe -- gemini-api` probes a specific provider (even if not yet
  // enabled in providerOrder); the default is whatever is first in the chain.
  const wanted = process.argv.slice(2).find((arg) => !arg.startsWith('-'));
  const handle = wanted
    ? chain.find((h) => h.info.id === wanted) ?? chainFor({ ...settings, providerOrder: [wanted] })[0]
    : chain[0]!;
  if (!handle) {
    console.log(`provider "${wanted}" is unknown. Available in chain: ${chain.map((h) => h.info.id).join(', ')}`);
    app.exit(1);
    return;
  }

  console.log(`probing ${handle.info.id} (${handle.info.label})…`);
  const result = await probeQuota(() => handle.create());
  console.log(JSON.stringify(result, null, 2));
  app.exit(result.ok ? 0 : 1);
}).catch((err) => {
  console.error(err);
  app.exit(1);
});
