import { contextBridge, ipcRenderer } from 'electron';

const SEND = new Set(['cs:audio', 'cs:commit', 'cs:cancel', 'cs:open-settings']);
const INVOKE = new Set(['settings:get', 'settings:set', 'history:list', 'quota:probe']);
const RECV = new Set([
  'cs:begin',
  'cs:state',
  'cs:interim',
  'cs:final',
  'cs:error',
  'cs:notice',
]);

const api = {
  send(channel: string, payload?: unknown): void {
    if (SEND.has(channel)) ipcRenderer.send(channel, payload);
  },
  invoke(channel: string, payload?: unknown): Promise<unknown> {
    if (!INVOKE.has(channel)) return Promise.reject(new Error(`blocked channel ${channel}`));
    return ipcRenderer.invoke(channel, payload);
  },
  on(channel: string, callback: (payload: any) => void): () => void {
    if (!RECV.has(channel)) return () => {};
    const listener = (_event: unknown, payload: unknown) => callback(payload);
    ipcRenderer.on(channel, listener);
    return () => {
      ipcRenderer.removeListener(channel, listener);
    };
  },
};

contextBridge.exposeInMainWorld('cs', api);
