# maia3-web

[Maia-3](https://github.com/CSSLab/maia3) is a chess transformer that predicts
the move a human of a given rating would play, and how the game is likely to
end. This package runs its 5M-parameter model (Maia3-5M) in the browser or in
Node.js:

- `src/maia3.ts`: the network and its input encoding in plain TypeScript, with
  no runtime dependencies.
- `src/worker.ts`: an optional Web Worker entry with a small message protocol.
- `weights/maia3-5m.int8.bin`: the Maia3-5M checkpoint converted to one 5.4 MB
  file (int8 matrices, float16 for the rest). `weights/manifest.json` records
  its hash, the upstream versions, and the quantization.

It does not know the rules of chess: you pass the positions as FEN and the
legal moves of the current position (from [chess.js](https://github.com/jhlywa/chess.js)
or similar), and it returns a probability for each legal move plus loss, draw,
and win chances for the side to move.

## Install

```sh
npm install https://github.com/GeoffreyPlitt/maia3-web/releases/download/v0.1.0/maia3-web-0.1.0.tgz
```

The release tarball contains the compiled JavaScript (`dist/`), type
declarations, the TypeScript source, and the weights. The type declarations
use `fetch` types, so a TypeScript project needs the DOM or WebWorker lib, or
`@types/node`.

## Usage

### Plain TypeScript

```ts
import { analyse, loadMaia3 } from 'maia3-web';
import weightsUrl from 'maia3-web/weights/maia3-5m.int8.bin?url'; // Vite; or any URL serving the file

const net = await loadMaia3(weightsUrl);
const fens = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
];
const { moves, value } = analyse(net, fens, ['e7e5', 'c7c5', 'e7e6' /* ...all legal moves */], 1500, 1500);
// moves: [{ move, p }, ...] over the legal moves given, most likely first
// value: [loss, draw, win] for the side to move (Black here)
```

- `fens` is every position of the game so far, oldest first; the model reads the
  last eight, which helps it play like a human in the middle of a game.
- `selfElo` is the rating of the player to move and `oppoElo` that of the
  opponent. Maia-3 learned from Lichess games, so ratings far outside the
  usual Lichess range are poorly covered.
- In Node.js, read the file and call `new Maia3(parseWeights(arrayBuffer))`.
- Lower-level pieces are exported too: `Maia3#forward` (raw policy logits and
  value), `encodeHistory`, `tokenize`, `positionFromFen`, `moveIndex`, and
  `legalMoveProbabilities`.

One position takes a few hundred milliseconds of CPU, so in a page prefer the
worker.

### Web Worker

`maia3-web/worker` is a module worker entry. Send it the weights URL once, then
analysis requests; it answers each request in order.

```ts
import weightsUrl from 'maia3-web/weights/maia3-5m.int8.bin?url';
import type { WorkerRequest, WorkerResponse } from 'maia3-web/worker';

// Bundlers need a local file to point the worker at; with Vite, a one-line
// `maia.worker.ts` containing `import 'maia3-web/worker';` works.
const worker = new Worker(new URL('./maia.worker.ts', import.meta.url), { type: 'module' });
worker.postMessage({ type: 'load', weights: new URL(weightsUrl, location.href).href } satisfies WorkerRequest);
worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
  if (e.data.type === 'error') console.error(e.data.id, e.data.error);
  else console.log(e.data.id, e.data.moves, e.data.value);
};
worker.postMessage({ type: 'analyse', id: 1, fens, legal, selfElo: 1500, oppoElo: 1500 } satisfies WorkerRequest);
```

| Message | Fields |
| --- | --- |
| in: `load` | `weights`: URL of `maia3-5m.int8.bin`. Starts loading; a failed load is retried on the next request. |
| in: `analyse` | `id`, `fens` (oldest first, current last), `legal` (UCI moves of the current position), `selfElo`, `oppoElo` |
| out: `result` | `id`, `moves` (`{ move, p }[]`, most likely first), `value` (`[loss, draw, win]` for the side to move) |
| out: `error` | `id`, `error` (message) |

## Accuracy

`test/reference.json` holds outputs of the upstream fp32 PyTorch model for 12
positions (including promotions and positions with move history). With the
int8 weights, the TypeScript port picks the same top move on all of them and
stays within 0.03 of every reference probability. `tools/quantcheck.py`
compares the quantized weights with fp32 in PyTorch: over 400 random positions
int8 picks the same top move 99.5% of the time (mean KL divergence about 9e-5).

## Reproducing the weights

The weights are converted from the Maia3-5M checkpoint with the upstream code at
a pinned commit. `tools/export_weights.py` refuses other versions.

```sh
tools/fetch_upstream.sh upstream   # CSSLab/maia3 @ 1e13597, maia3-5m.pt (SHA-256 checked)
python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple -r tools/requirements.txt
.venv/bin/python tools/export_weights.py upstream/maia3 upstream/maia3-5m.pt out int8
node tools/verify_weights.mjs out/maia3-5m.int8.bin   # same SHA-256 as weights/manifest.json
.venv/bin/python tools/quantcheck.py upstream/maia3 upstream/maia3-5m.pt
```

The export is deterministic, and CI repeats it on every push. It also writes
`out/reference.json`, the source of `test/reference.json`.

## Development

```sh
npm install
npm run typecheck
npm test
npm run verify:weights
npm pack   # builds dist/ and the release tarball
```

## Credits

Maia-3 is by the University of Toronto Computational Social Science Lab
([CSSLab/maia3](https://github.com/CSSLab/maia3), weights at
[UofTCSSLab/Maia3-5M](https://huggingface.co/UofTCSSLab/Maia3-5M)). If you use
it, please cite their paper:

> Daniel Monroe, George Eilender, Philip Chalmers, Zhenwei Tang, and Ashton
> Anderson. [Chessformer: A Unified Architecture for Chess Modeling](https://openreview.net/forum?id=2ltBRzEHyd).
> ICLR 2026.

This package is an independent port and is not affiliated with the Maia-3
authors. See NOTICE for attribution and the changes made.

## License

GNU Affero General Public License v3.0 (see LICENSE), the license of the
upstream code and weights. If you run a modified version for users over a
network, AGPL-3.0 section 13 requires offering them its source.
