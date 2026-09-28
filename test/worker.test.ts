import { readFileSync } from 'node:fs';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import type { WorkerRequest, WorkerResponse } from '../src/worker.js';

// Runs the worker entry in-process with a stand-in for the worker global scope.
const posted: WorkerResponse[] = [];
const scope: { onmessage: ((e: { data: WorkerRequest }) => void) | null; postMessage(m: WorkerResponse): void } = {
  onmessage: null, postMessage: m => { posted.push(m); },
};
const send = (data: WorkerRequest) => scope.onmessage!({ data });
const next = async (n: number) => { await vi.waitFor(() => expect(posted.length).toBeGreaterThanOrEqual(n), { timeout: 20_000 }); return posted[n - 1]; };
const bytes = readFileSync(new URL('../weights/maia3-5m.int8.bin', import.meta.url));
const fetchMock = vi.fn(async (url: string | URL) => String(url).endsWith('missing.bin') ? new Response(null, { status: 404 }) : new Response(bytes));
const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

describe('worker protocol', () => {
  beforeAll(async () => {
    vi.stubGlobal('self', scope); vi.stubGlobal('fetch', fetchMock);
    await import('../src/worker.js');
  });
  afterAll(() => { vi.unstubAllGlobals(); });

  it('reports an error until weights are set', async () => {
    send({ type: 'analyse', id: 1, fens: [start], legal: ['e2e4'], selfElo: 1500, oppoElo: 1500 });
    expect(await next(1)).toMatchObject({ type: 'error', id: 1, error: expect.stringContaining('load') });
  });

  it('reports failed loads and retries them', async () => {
    send({ type: 'load', weights: 'https://example.test/missing.bin' });
    send({ type: 'analyse', id: 2, fens: [start], legal: ['e2e4'], selfElo: 1500, oppoElo: 1500 });
    expect(await next(2)).toMatchObject({ type: 'error', id: 2, error: expect.stringContaining('404') });
  });

  it('answers requests in order with move probabilities and win/draw/loss', async () => {
    send({ type: 'load', weights: 'https://example.test/maia3-5m.int8.bin' });
    send({ type: 'analyse', id: 3, fens: [start], legal: ['e2e4', 'd2d4', 'a2a3'], selfElo: 1500, oppoElo: 1500 });
    send({ type: 'analyse', id: 4, fens: [start], legal: [], selfElo: 1500, oppoElo: 1500 });
    const a = await next(3), b = await next(4);
    expect(a.type === 'result' && a.id).toBe(3);
    if (a.type !== 'result') throw new Error(a.error);
    expect(a.moves.map(m => m.move).sort()).toEqual(['a2a3', 'd2d4', 'e2e4']);
    expect(a.moves[2].move).toBe('a2a3');
    expect(a.value.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 5);
    expect(b).toMatchObject({ type: 'result', id: 4, moves: [] });
    expect(fetchMock.mock.calls.filter(([u]) => String(u).endsWith('int8.bin'))).toHaveLength(1);
  }, 30_000);
});
