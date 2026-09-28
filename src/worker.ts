// Optional Web Worker entry: runs Maia-3 off the main thread.
//
// Protocol (all messages are plain objects):
//
//   -> { type: 'load', weights: string }
//        Sets the weight file URL and starts loading it. Send this first.
//        Loading is retried on the next request if it fails.
//   -> { type: 'analyse', id, fens, legal, selfElo, oppoElo }
//        fens: every position so far as FEN, oldest first, current last.
//        legal: legal moves of the current position in UCI (e.g. 'e2e4').
//   <- { type: 'result', id, moves: { move, p }[], value: [loss, draw, win] }
//        moves: the legal moves, most likely first. value: for the side to move.
//   <- { type: 'error', id, error: string }
//
// Requests are answered in the order they arrive.
import { Maia3, analyse, loadMaia3, type Analysis } from './maia3.js';

export interface LoadRequest { type: 'load'; weights: string }
export interface AnalyseRequest { type: 'analyse'; id: number; fens: string[]; legal: string[]; selfElo: number; oppoElo: number }
export type WorkerRequest = LoadRequest | AnalyseRequest;
export type WorkerResponse =
  | ({ type: 'result'; id: number } & Analysis)
  | { type: 'error'; id: number; error: string };

interface Scope { onmessage: ((e: MessageEvent<WorkerRequest>) => void) | null; postMessage(message: WorkerResponse): void }
const scope = self as unknown as Scope;

let weights: string | null = null, model: Promise<Maia3> | null = null;
const load = () => {
  if (!weights) return Promise.reject(new Error('No weights: send { type: "load", weights } first.'));
  if (!model) {
    const pending: Promise<Maia3> = loadMaia3(weights).catch(error => { if (model === pending) model = null; throw error; });
    model = pending;
  }
  return model;
};

let queue: Promise<void> = Promise.resolve();
scope.onmessage = e => {
  const req = e.data;
  if (req.type === 'load') {
    if (req.weights !== weights) { weights = req.weights; model = null; }
    load().catch(() => undefined);
    return;
  }
  if (req.type !== 'analyse') return;
  queue = queue.then(async () => {
    try {
      const result = analyse(await load(), req.fens, req.legal, req.selfElo, req.oppoElo);
      scope.postMessage({ type: 'result', id: req.id, ...result });
    } catch (error) {
      scope.postMessage({ type: 'error', id: req.id, error: error instanceof Error ? error.message : String(error) });
    }
  });
};
