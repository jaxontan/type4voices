/* Settings window. Plain DOM, no framework: this screen changes rarely and the
 * whole point is that a user can read what happens to their audio. */

const el = (id) => document.getElementById(id);
const statusEl = el('status');

let providers = [];
let current = null;

function setStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = kind || '';
}

function renderProviders() {
  const host = el('providers');
  host.textContent = '';

  const order = current.providerOrder || [];
  const enabled = new Set(order);

  for (const provider of providers) {
    const index = order.indexOf(provider.id);
    const row = document.createElement('div');
    row.className = 'provider';

    const check = document.createElement('input');
    check.type = 'checkbox';
    check.checked = index >= 0;
    check.dataset.id = provider.id;

    const priority = document.createElement('input');
    priority.type = 'number';
    priority.min = '1';
    priority.value = String(index >= 0 ? index + 1 : order.length + 1);
    priority.dataset.priority = provider.id;
    priority.disabled = index < 0;

    const body = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'name';
    name.textContent = provider.label;
    const badge = document.createElement('span');
    badge.className = 'badge ' + provider.kind;
    badge.textContent = provider.kind;
    name.appendChild(badge);
    const note = document.createElement('div');
    note.className = 'note';
    note.textContent = provider.note;
    body.append(name, note);

    check.addEventListener('change', () => {
      priority.disabled = !check.checked;
      if (check.checked && !priority.value) {
        priority.value = String(order.length + 1);
      }
    });

    row.append(check, priority, body);
    host.appendChild(row);
  }
}

function collectOrder() {
  const checks = [...document.querySelectorAll('#providers input[type="checkbox"]')];
  const rows = checks
    .filter((c) => c.checked)
    .map((c) => {
      const id = c.dataset.id;
      const priority = document.querySelector(`#providers input[data-priority="${id}"]`);
      const n = Number(priority && priority.value);
      return { id, n: Number.isFinite(n) && n > 0 ? n : 999 };
    });

  rows.sort((a, b) => a.n - b.n);
  return rows.map((r) => r.id);
}

function fillForm(settings) {
  el('hotkey').value = settings.hotkey;
  el('cancelHotkey').value = settings.cancelHotkey;
  el('silenceMs').value = String(settings.silenceMs);
  el('maxUtteranceMs').value = String(settings.maxUtteranceMs);
  el('language').value = settings.language;
  el('injectMode').value = settings.injectMode;
  el('cleanupPass').checked = Boolean(settings.cleanupPass);
  el('autoRefreshClaudeToken').checked = Boolean(settings.autoRefreshClaudeToken);
  el('geminiApiKey').value = settings.geminiApiKey || '';
  el('glossary').value = (settings.glossary || []).join('\n');
}

function parseGlossary(raw) {
  return raw
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 1000);
}

async function loadAll() {
  const data = await window.cs.invoke('settings:get');
  current = data.settings;
  providers = data.providers;
  renderProviders();
  fillForm(current);
  await loadHistory();
}

async function loadHistory() {
  const entries = await window.cs.invoke('history:list');
  const host = el('history');
  host.textContent = '';

  if (!entries || !entries.length) {
    const empty = document.createElement('div');
    empty.className = 'meta';
    empty.textContent = 'Nothing yet. Audio is never stored; only the text you see here.';
    host.appendChild(empty);
    return;
  }

  for (const entry of entries) {
    const item = document.createElement('div');
    item.className = 'item';

    const meta = document.createElement('div');
    meta.className = 'meta';
    const when = new Date(entry.at).toLocaleString();
    meta.textContent = `${when} · ${entry.provider} · ${(entry.ms / 1000).toFixed(1)}s · ${entry.chars} chars`;

    const text = document.createElement('div');
    text.textContent = entry.text;

    item.append(meta, text);
    host.appendChild(item);
  }
}

el('save').addEventListener('click', async () => {
  const order = collectOrder();
  if (!order.length) {
    setStatus('Enable at least one provider.', 'bad');
    return;
  }

  const patch = {
    providerOrder: order,
    hotkey: el('hotkey').value.trim(),
    cancelHotkey: el('cancelHotkey').value.trim(),
    silenceMs: Number(el('silenceMs').value) || 15000,
    maxUtteranceMs: Number(el('maxUtteranceMs').value) || 300000,
    language: el('language').value.trim() || 'en',
    injectMode: el('injectMode').value,
    cleanupPass: el('cleanupPass').checked,
    autoRefreshClaudeToken: el('autoRefreshClaudeToken').checked,
    geminiApiKey: el('geminiApiKey').value.trim(),
    glossary: parseGlossary(el('glossary').value),
  };

  const result = await window.cs.invoke('settings:set', patch);
  current = result.settings;
  renderProviders();

  if (result.hotkeys && !result.hotkeys.ok) {
    setStatus(`Saved, but the OS refused: ${result.hotkeys.failed.join(', ')}`, 'bad');
  } else {
    setStatus('Saved.', 'good');
  }
});

el('probe').addEventListener('click', async () => {
  setStatus('Probing…');
  el('probe').disabled = true;
  try {
    const result = await window.cs.invoke('quota:probe');
    const label = result.ok ? 'ok' : 'failed';
    setStatus(`${result.provider}: ${label} in ${result.ms} ms — ${result.detail}`, result.ok ? 'good' : 'bad');
  } catch (err) {
    setStatus(String(err && err.message ? err.message : err), 'bad');
  } finally {
    el('probe').disabled = false;
  }
});

void loadAll();
