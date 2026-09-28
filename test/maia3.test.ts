import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { Chess } from 'chess.js';
import { Maia3, analyse, encodeHistory, legalMoveProbabilities, moveIndex, parseWeights, positionFromFen, tokenize } from '../src/maia3.js';
import reference from './reference.json' with { type: 'json' };

// Outputs of the upstream fp32 PyTorch model, written by tools/export_weights.py.
interface Case { start: string; moves: string[]; fen: string; selfElo: number; oppoElo: number; top: [string, number][]; value: [number, number, number] }
const cases = reference as Case[];
const file = readFileSync(new URL('../weights/maia3-5m.int8.bin', import.meta.url));
const net = new Maia3(parseWeights(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength)));

function replay(c: Case) {
  const chess = new Chess(c.start), fens = [chess.fen()];
  for (const m of c.moves) { chess.move({ from: m.slice(0, 2), to: m.slice(2, 4), promotion: m[4] }); fens.push(chess.fen()); }
  return { chess, fens, legal: chess.moves({ verbose: true }).map(m => m.from + m.to + (m.promotion ?? '')) };
}

describe('Maia-3 port', () => {
  it('has reference positions with history, promotions, and both sides to move', () => {
    expect(cases.length).toBe(12);
    expect(cases.some(c => c.moves.length >= 8)).toBe(true);
    expect(cases.some(c => c.top.some(([m]) => m.length === 5))).toBe(true);
    expect(new Set(cases.map(c => c.fen.split(' ')[1]))).toEqual(new Set(['w', 'b']));
  });

  it('matches the upstream model on reference positions', () => {
    let worst = 0;
    for (const c of cases) {
      const { chess, fens, legal } = replay(c);
      expect(chess.fen()).toBe(c.fen);
      const out = net.forward(encodeHistory(fens.map(positionFromFen)), c.selfElo, c.oppoElo);
      const probs = legalMoveProbabilities(out.policy, legal, chess.turn()), byMove = new Map(probs.map(p => [p.move, p.p]));
      expect(probs[0].move, c.fen).toBe(c.top[0][0]);
      expect(probs.reduce((a, p) => a + p.p, 0)).toBeCloseTo(1, 5);
      for (const [move, p] of c.top) worst = Math.max(worst, Math.abs((byMove.get(move) ?? 0) - p));
      for (let i = 0; i < 3; i++) worst = Math.max(worst, Math.abs(out.value[i] - c.value[i]));
    }
    // int8 weights: small, bounded drift from fp32.
    expect(worst).toBeLessThan(.03);
  }, 60_000);

  it('analyse combines encoding, the network, and the legal-move softmax', () => {
    const c = cases.find(k => k.moves.length > 0)!, { fens, legal } = replay(c);
    const a = analyse(net, fens, legal, c.selfElo, c.oppoElo);
    expect(a.moves.map(m => m.move).sort()).toEqual([...legal].sort());
    expect(a.moves[0].move).toBe(c.top[0][0]);
    expect(a.value.reduce((x, y) => x + y, 0)).toBeCloseTo(1, 5);
    // A finished game has no legal moves, but still has a value.
    expect(analyse(net, fens, [], c.selfElo, c.oppoElo).moves).toEqual([]);
  }, 30_000);

  it('encodes positions from the side to move', () => {
    // After 1.e4, Black to move: the board is mirrored, so Black's e7 pawn is an own pawn on e2.
    const t = tokenize(positionFromFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'));
    expect(t[12 * 12 + 0]).toBe(1); // e2: own pawn
    expect(t[(4 * 8 + 4) * 12 + 6]).toBe(1); // White's e4 pawn: an opponent pawn on the mirrored e5
    expect(moveIndex('e7e5', 'b')).toBe(moveIndex('e2e4', 'w'));
    expect(moveIndex('g2g1n', 'b')).toBe(4096 + (6 * 8 + 6) * 4 + 3);
    // Short histories repeat the earliest position.
    const start = positionFromFen(new Chess().fen()), one = encodeHistory([start]);
    expect(one.subarray(0, 12)).toEqual(one.subarray(12 * 7, 12 * 8));
  });
});
