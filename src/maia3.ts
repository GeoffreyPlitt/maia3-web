// Maia-3 (5M) inference in plain TypeScript, with no runtime dependencies.
//
// A port of the network in https://github.com/CSSLab/maia3 (AGPL-3.0), which
// predicts the move a human of a given rating would play. The weight file is
// produced by tools/export_weights.py; see NOTICE for attribution.
//
// Squares use python-chess numbering (a1 = 0, h1 = 7, a8 = 56). The network
// always sees the position from the side to move, so Black's positions are
// mirrored (ranks flipped, colors swapped) before encoding.

export interface Maia3Config {
  history: number; dim_emb: number; dim_vit: number; num_blocks: number; num_heads: number;
  mlp_ratio: number; head_hid_dim: number; gab_gen_size: number; gab_intermediate_dim: number;
}
interface Header { format: string; config: Maia3Config; tensors: { name: string; shape: number[]; offset: number; dtype: 'fp16' | 'int8' }[] }
export interface Maia3Weights { config: Maia3Config; tensors: Map<string, Float32Array> }
export interface Maia3Output { policy: Float32Array; value: [loss: number, draw: number, win: number] }

const MOVES = 64 * 64 + 8 * 8 * 4;
const PROMOS = ['q', 'r', 'b', 'n'];
const PIECES = 'pnbrqk';

function halfToFloat(h: number) {
  const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
  if (e === 0) return s * f * 2 ** -24;
  if (e === 31) return f ? NaN : s * Infinity;
  return s * (1 + f / 1024) * 2 ** (e - 15);
}

/** Parses the single-file weight format: uint32 header length, JSON header, then tensor data. */
export function parseWeights(buffer: ArrayBuffer): Maia3Weights {
  const view = new DataView(buffer), length = view.getUint32(0, true);
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, length))) as Header;
  const base = 4 + length, tensors = new Map<string, Float32Array>();
  for (const t of header.tensors) {
    const size = t.shape.reduce((a, b) => a * b, 1), out = new Float32Array(size), at = base + t.offset;
    if (t.dtype === 'int8') {
      const rows = t.shape[0], cols = size / rows;
      const scales = new Float32Array(buffer.slice(at, at + rows * 4)), q = new Int8Array(buffer, at + rows * 4, size);
      for (let r = 0; r < rows; r++) for (let c = 0; c < cols; c++) out[r * cols + c] = q[r * cols + c] * scales[r];
    } else {
      for (let i = 0; i < size; i++) out[i] = halfToFloat(view.getUint16(at + i * 2, true));
    }
    tensors.set(t.name, out);
  }
  return { config: header.config, tensors };
}

// --- numeric helpers ----------------------------------------------------------

/** y[n, out] = x[n, in] · W[out, in]ᵀ + b. Four rows at a time share each weight load. */
function linear(x: Float32Array, n: number, inDim: number, w: Float32Array, outDim: number, b?: Float32Array) {
  const y = new Float32Array(n * outDim);
  let i = 0;
  for (; i + 3 < n; i += 4) {
    const x0 = i * inDim, x1 = x0 + inDim, x2 = x1 + inDim, x3 = x2 + inDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim; let s0 = 0, s1 = 0, s2 = 0, s3 = 0;
      for (let k = 0; k < inDim; k++) {
        const wk = w[wo + k];
        s0 += x[x0 + k] * wk; s1 += x[x1 + k] * wk; s2 += x[x2 + k] * wk; s3 += x[x3 + k] * wk;
      }
      const bo = b ? b[o] : 0;
      y[i * outDim + o] = s0 + bo; y[(i + 1) * outDim + o] = s1 + bo;
      y[(i + 2) * outDim + o] = s2 + bo; y[(i + 3) * outDim + o] = s3 + bo;
    }
  }
  for (; i < n; i++) {
    const xo = i * inDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim; let s = 0;
      for (let k = 0; k < inDim; k++) s += x[xo + k] * w[wo + k];
      y[i * outDim + o] = s + (b ? b[o] : 0);
    }
  }
  return y;
}
// erf with |error| < 1.2e-7 (Numerical Recipes erfc approximation).
function erf(x: number) {
  const z = Math.abs(x), t = 1 / (1 + .5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (.37409196 + t * (.09678418 + t * (-.18628806 +
    t * (.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-.82215223 + t * .17087277)))))))));
  return x >= 0 ? 1 - r : r - 1;
}
function gelu(x: Float32Array) { for (let i = 0; i < x.length; i++) x[i] = .5 * x[i] * (1 + erf(x[i] / Math.SQRT2)); return x; }
function layerNorm(x: Float32Array, n: number, d: number, w: Float32Array, b: Float32Array, eps = 1e-5) {
  for (let i = 0; i < n; i++) {
    const o = i * d; let mean = 0, v = 0;
    for (let k = 0; k < d; k++) mean += x[o + k]; mean /= d;
    for (let k = 0; k < d; k++) v += (x[o + k] - mean) ** 2;
    const inv = 1 / Math.sqrt(v / d + eps);
    for (let k = 0; k < d; k++) x[o + k] = (x[o + k] - mean) * inv * w[k] + b[k];
  }
  return x;
}
// torch.nn.RMSNorm with eps=None uses the float32 machine epsilon.
function rmsNorm(x: Float32Array, n: number, d: number, w: Float32Array, eps = 1.1920928955078125e-7) {
  for (let i = 0; i < n; i++) {
    const o = i * d; let ss = 0;
    for (let k = 0; k < d; k++) ss += x[o + k] * x[o + k];
    const inv = 1 / Math.sqrt(ss / d + eps);
    for (let k = 0; k < d; k++) x[o + k] = x[o + k] * inv * w[k];
  }
  return x;
}
function softmaxInPlace(x: Float32Array, start: number, len: number) {
  let m = -Infinity; for (let i = 0; i < len; i++) m = Math.max(m, x[start + i]);
  let s = 0; for (let i = 0; i < len; i++) { const e = Math.exp(x[start + i] - m); x[start + i] = e; s += e; }
  for (let i = 0; i < len; i++) x[start + i] /= s;
}

// --- network -------------------------------------------------------------------

export class Maia3 {
  readonly cfg: Maia3Config;
  private t: Map<string, Float32Array>;
  constructor(weights: Maia3Weights) { this.cfg = weights.config; this.t = weights.tensors; }
  private w(name: string) { const v = this.t.get(name); if (!v) throw new Error(`Missing Maia-3 tensor ${name}`); return v; }

  private eloEmbedding(elo: number) {
    const d = this.cfg.dim_emb, low = this.w('elo_embedding_low.weight'), high = this.w('elo_embedding_high.weight');
    const wl = Math.min(5000, Math.max(0, Math.trunc(elo))) / 5000, out = new Float32Array(d);
    for (let i = 0; i < d; i++) out[i] = wl * low[i] + (1 - wl) * high[i];
    return out;
  }

  /** tokens: 64 × (12 · history) board features, oldest position first. */
  forward(tokens: Float32Array, selfElo: number, oppoElo: number): Maia3Output {
    const c = this.cfg, D = c.dim_vit, H = c.num_heads, hd = D / H, F = Math.round(D * c.mlp_ratio);
    const tokDim = 12 * c.history, inDim = tokDim + 2 * c.dim_emb;
    const se = this.eloEmbedding(selfElo), oe = this.eloEmbedding(oppoElo), emb = new Float32Array(64 * inDim);
    for (let s = 0; s < 64; s++) {
      emb.set(tokens.subarray(s * tokDim, (s + 1) * tokDim), s * inDim);
      emb.set(se, s * inDim + tokDim); emb.set(oe, s * inDim + tokDim + c.dim_emb);
    }
    const x = linear(emb, 64, inDim, this.w('token_projection.weight'), D, this.w('token_projection.bias'));
    const gab = this.w('gab_shared_weight'), G = c.gab_gen_size, scale = 1 / Math.sqrt(hd);
    for (let l = 0; l < c.num_blocks; l++) {
      const p = `transformer.layers.${l}.`;
      // Geometric attention bias: a per-position, per-head 64×64 bias generated from the mean token.
      const mean = new Float32Array(D);
      for (let s = 0; s < 64; s++) for (let k = 0; k < D; k++) mean[k] += x[s * D + k] / 64;
      let g = gelu(linear(mean, 1, D, this.w(p + 'self_attn.sm2.weight'), c.gab_intermediate_dim, this.w(p + 'self_attn.sm2.bias')));
      layerNorm(g, 1, c.gab_intermediate_dim, this.w(p + 'self_attn.ln1.weight'), this.w(p + 'self_attn.ln1.bias'));
      g = gelu(linear(g, 1, c.gab_intermediate_dim, this.w(p + 'self_attn.sm3.weight'), H * G, this.w(p + 'self_attn.sm3.bias')));
      layerNorm(g, 1, H * G, this.w(p + 'self_attn.ln2.weight'), this.w(p + 'self_attn.ln2.bias'));
      const bias = linear(g, H, G, gab, 4096); // (H, 64·64)

      const qkv = linear(x, 64, D, this.w(p + 'self_attn.mha.in_proj_weight'), 3 * D);
      const att = new Float32Array(64 * D), scores = new Float32Array(64 * 64);
      for (let h = 0; h < H; h++) {
        const ho = h * hd;
        for (let i = 0; i < 64; i++) {
          const qo = i * 3 * D + ho;
          for (let j = 0; j < 64; j++) {
            const ko = j * 3 * D + D + ho; let s = 0;
            for (let k = 0; k < hd; k++) s += qkv[qo + k] * qkv[ko + k];
            scores[i * 64 + j] = s * scale + bias[h * 4096 + i * 64 + j];
          }
          softmaxInPlace(scores, i * 64, 64);
          for (let j = 0; j < 64; j++) {
            const a = scores[i * 64 + j], vo = j * 3 * D + 2 * D + ho, oo = i * D + ho;
            for (let k = 0; k < hd; k++) att[oo + k] += a * qkv[vo + k];
          }
        }
      }
      const sa = linear(att, 64, D, this.w(p + 'self_attn.mha.out_proj.weight'), D);
      for (let i = 0; i < x.length; i++) x[i] += sa[i];
      rmsNorm(x, 64, D, this.w(p + 'norm1.weight'));
      const ff = linear(gelu(linear(x, 64, D, this.w(p + 'linear1.weight'), F, this.w(p + 'linear1.bias'))), 64, F,
        this.w(p + 'linear2.weight'), D, this.w(p + 'linear2.bias'));
      for (let i = 0; i < x.length; i++) x[i] += ff[i];
      rmsNorm(x, 64, D, this.w(p + 'norm2.weight'));
    }
    layerNorm(x, 64, D, this.w('transformer.norm.weight'), this.w('transformer.norm.bias'));

    // Policy: from-square × to-square scores, plus promotion biases on the 8th rank.
    const hh = c.head_hid_dim, from = linear(x, 64, D, this.w('proj_sq_from.weight'), hh), to = linear(x, 64, D, this.w('proj_sq_to.weight'), hh);
    const policy = new Float32Array(MOVES), root = Math.sqrt(hh);
    for (let i = 0; i < 64; i++) for (let j = 0; j < 64; j++) {
      let s = 0; for (let k = 0; k < hh; k++) s += from[i * hh + k] * to[j * hh + k];
      policy[i * 64 + j] = s / root;
    }
    const promo = linear(to.subarray(56 * hh), 8, hh, this.w('promo_bias_proj.weight'), 4);
    for (let ff = 0; ff < 8; ff++) for (let tf = 0; tf < 8; tf++) for (let pc = 0; pc < 4; pc++)
      policy[4096 + (ff * 8 + tf) * 4 + pc] = policy[(48 + ff) * 64 + 56 + tf] + promo[tf * 4 + pc] * root;

    // Value: loss / draw / win for the side to move.
    const pooled = new Float32Array(D);
    for (let s = 0; s < 64; s++) for (let k = 0; k < D; k++) pooled[k] += x[s * D + k] / 64;
    layerNorm(pooled, 1, D, this.w('last_ln.weight'), this.w('last_ln.bias'));
    const hid = linear(pooled, 1, D, this.w('fc_value_hid.weight'), hh, this.w('fc_value_hid.bias')).map(v => Math.max(0, v));
    const value = linear(hid, 1, hh, this.w('fc_value.weight'), 3, this.w('fc_value.bias'));
    softmaxInPlace(value, 0, 3);
    return { policy, value: [value[0], value[1], value[2]] };
  }
}

// --- encoding --------------------------------------------------------------------

/** A position as the board part of a FEN (8 ranks) plus the side to move. */
export interface Position { placement: string; turn: 'w' | 'b' }
export function positionFromFen(fen: string): Position {
  const [placement, turn] = fen.split(' ');
  return { placement, turn: turn === 'b' ? 'b' : 'w' };
}
/** 64 × 12 one-hot planes for one position, seen from the side to move. */
export function tokenize(p: Position) {
  const out = new Float32Array(64 * 12), ranks = p.placement.split('/');
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of ranks[r]) {
      if (ch >= '1' && ch <= '8') { file += +ch; continue; }
      const white = ch === ch.toUpperCase(), type = PIECES.indexOf(ch.toLowerCase()), rank = 7 - r;
      // Mirror for Black to move: flip the rank and swap colors.
      const sq = p.turn === 'w' ? rank * 8 + file : (7 - rank) * 8 + file;
      const own = p.turn === 'w' ? white : !white;
      out[sq * 12 + type + (own ? 0 : 6)] = 1; file++;
    }
  }
  return out;
}
/** Network input for the last `history` positions (current last), padding with the earliest one. */
export function encodeHistory(positions: Position[], history = 8) {
  const recent = positions.slice(-history), boards = recent.map(tokenize);
  while (boards.length < history) boards.unshift(boards[0]);
  const tokens = new Float32Array(64 * 12 * history);
  for (let s = 0; s < 64; s++) for (let b = 0; b < history; b++) tokens.set(boards[b].subarray(s * 12, s * 12 + 12), s * 12 * history + b * 12);
  return tokens;
}
const sqIndex = (sq: string) => (sq.charCodeAt(0) - 97) + (+sq[1] - 1) * 8;
const mirror = (sq: string) => sq[0] + (9 - +sq[1]);
/** Index of a UCI move in the policy vector, from the perspective of the side to move. */
export function moveIndex(uci: string, turn: 'w' | 'b') {
  const from = turn === 'w' ? uci.slice(0, 2) : mirror(uci.slice(0, 2)), to = turn === 'w' ? uci.slice(2, 4) : mirror(uci.slice(2, 4));
  if (uci.length > 4) return 4096 + ((from.charCodeAt(0) - 97) * 8 + (to.charCodeAt(0) - 97)) * 4 + PROMOS.indexOf(uci[4]);
  return sqIndex(from) * 64 + sqIndex(to);
}
export interface MoveProbability { move: string; p: number }
/** Softmax of the policy over the given legal moves, most likely first. */
export function legalMoveProbabilities(policy: Float32Array, legal: string[], turn: 'w' | 'b'): MoveProbability[] {
  const logits = legal.map(m => policy[moveIndex(m, turn)]), max = Math.max(...logits);
  const exp = logits.map(l => Math.exp(l - max)), sum = exp.reduce((a, b) => a + b, 0);
  return legal.map((move, i) => ({ move, p: exp[i] / sum })).sort((a, b) => b.p - a.p);
}

export interface Analysis {
  /** The given legal moves, most likely first. */
  moves: MoveProbability[];
  /** Loss, draw, and win chances for the side to move. */
  value: [loss: number, draw: number, win: number];
}
/**
 * Maia's model of a `selfElo` player to move against an `oppoElo` opponent.
 * `fens` lists every position so far, oldest first and the current one last
 * (only the last `history` positions are used). `legal` holds the legal moves
 * of the current position in UCI notation (e.g. `e2e4`, `e7e8q`); with none, `moves` is empty.
 */
export function analyse(net: Maia3, fens: string[], legal: string[], selfElo: number, oppoElo: number): Analysis {
  if (fens.length === 0) throw new Error('analyse needs at least one position.');
  const positions = fens.map(positionFromFen), turn = positions[positions.length - 1].turn;
  const out = net.forward(encodeHistory(positions, net.cfg.history), selfElo, oppoElo);
  return { moves: legalMoveProbabilities(out.policy, legal, turn), value: out.value };
}

/** Fetches and parses a weight file, e.g. `weights/maia3-5m.int8.bin` from this package. */
export async function loadMaia3(url: string | URL, init?: RequestInit): Promise<Maia3> {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(`Maia-3 weights did not load (HTTP ${response.status}).`);
  return new Maia3(parseWeights(await response.arrayBuffer()));
}
