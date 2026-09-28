"""Export the Maia3-5M checkpoint to this package's weight format, plus reference outputs.

Usage: python tools/export_weights.py <maia3 repo dir> <maia3-5m.pt> <out dir> [int8|fp16] [--weights-only]

Writes <out dir>/maia3-5m.<fmt>.bin (uint32 header length, JSON header, tensor
data) and <out dir>/reference.json (policy and value outputs of the upstream
fp32 PyTorch model for 12 positions, which test/maia3.test.ts compares
against). The upstream code and checkpoint must be the pinned versions below;
see README.md ("Reproducing the weights") and weights/manifest.json.

Quantization (int8): every 2-D tensor with at least 4096 entries is stored as
symmetric per-row int8 with a float32 scale per row (scale = max|row| / 127);
every other tensor is stored as float16. The export is deterministic.
"""
import hashlib, json, subprocess, sys, struct, random
from pathlib import Path
from types import SimpleNamespace

UPSTREAM_REPO = "https://github.com/CSSLab/maia3"
UPSTREAM_COMMIT = "1e13597c42d4858b7cfd7cfdae01e297263364b2"
CHECKPOINT_SHA256 = "ba14208b2992d85502f5fb501934abf6aaaeb355e9f3fdf90e326911f562524f"

import numpy as np
import torch
import chess

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 3:
    sys.exit(__doc__)
repo, ckpt, out = Path(args[0]), Path(args[1]), Path(args[2])
fmt = args[3] if len(args) > 3 else "int8"
if fmt not in ("int8", "fp16"):
    sys.exit(f"unknown format {fmt!r}: use int8 or fp16")
commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
if commit != UPSTREAM_COMMIT:
    sys.exit(f"{repo} is at {commit}; check out {UPSTREAM_COMMIT} from {UPSTREAM_REPO}")
digest = hashlib.sha256(ckpt.read_bytes()).hexdigest()
if digest != CHECKPOINT_SHA256:
    sys.exit(f"{ckpt} has SHA-256 {digest}; expected {CHECKPOINT_SHA256}")
sys.path.insert(0, str(repo))
from maia3.model_registry import resolve_model_spec  # noqa: E402
from maia3.models import MAIA3Model  # noqa: E402
from maia3.dataset import tokenize_board, get_legal_moves_mask  # noqa: E402
from maia3.utils import get_all_possible_moves, mirror_move  # noqa: E402

spec = resolve_model_spec("maia3-5m")
cfg = SimpleNamespace(**spec.config)
model = MAIA3Model(cfg)
state = torch.load(ckpt, map_location="cpu", weights_only=True)
state = state.get("model_state_dict", state)
state = {k.replace("smolgen", "gab"): v for k, v in state.items()}
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, missing
print("unused checkpoint entries:", unexpected)
model.eval()

out.mkdir(parents=True, exist_ok=True)
sd = model.state_dict()
# Skip buffers we can recompute and duplicated shared GAB weights.
tensors = {}
for k, v in sd.items():
    if ".self_attn.gab_weight" in k:
        continue
    tensors[k] = v.detach().float().numpy()

manifest = {"format": fmt, "config": spec.config, "tensors": []}
blob = bytearray()
for name, arr in tensors.items():
    entry = {"name": name, "shape": list(arr.shape)}
    flat = arr.reshape(-1)
    while len(blob) % 4:
        blob.append(0)
    entry["offset"] = len(blob)
    if fmt == "int8" and arr.ndim == 2 and arr.size >= 4096:
        # symmetric per-row quantisation
        scale = np.abs(arr).max(axis=1) / 127.0
        scale[scale == 0] = 1.0
        q = np.clip(np.round(arr / scale[:, None]), -127, 127).astype(np.int8)
        entry["dtype"] = "int8"
        blob += scale.astype(np.float32).tobytes()
        blob += q.tobytes()
    else:
        entry["dtype"] = "fp16"
        blob += flat.astype(np.float16).tobytes()
    manifest["tensors"].append(entry)

print("params", sum(a.size for a in tensors.values()), "bytes", len(blob))
header = json.dumps(manifest, separators=(",", ":")).encode()
header += b" " * ((-len(header) - 4) % 16)
data = struct.pack("<I", len(header)) + header + bytes(blob)
name = f"maia3-5m.{fmt}.bin"
(out / name).write_bytes(data)
print(name, len(data), "bytes, SHA-256", hashlib.sha256(data).hexdigest())
if "--weights-only" in sys.argv:
    sys.exit(0)

# Reference outputs: random positions from random games, with real history.
all_moves = get_all_possible_moves()
moves_dict = {m: i for i, m in enumerate(all_moves)}
rng = random.Random(7)
cases = []
fixed = [
    # Black to move with a promotion available (mirroring + promotion head).
    "8/8/8/8/8/1K6/3k2p1/8 b - - 0 60",
    # White to move with an underpromotion choice.
    "8/5P2/8/8/8/1k6/8/1K6 w - - 0 60",
]
while len(cases) < 12:
    board = chess.Board()
    history = []
    plies = rng.randint(0, 60)
    if fixed:
        board = chess.Board(fixed.pop())
    else:
        for _ in range(plies):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
        if board.is_game_over():
            continue
    # rebuild the last 8 boards from the move stack
    moves = [m.uci() for m in board.move_stack]
    replay = chess.Board(board.root().fen()) if board.move_stack else chess.Board(board.fen())
    boards = [tokenize_board(replay)]
    for m in moves:
        replay.push_uci(m)
        boards.append(tokenize_board(replay))
    hist = boards[-8:]
    tokens = torch.cat(hist, dim=1)
    if len(hist) < 8:
        tokens = torch.cat([hist[0].repeat(1, 8 - len(hist)), tokens], dim=1)
    tokens = torch.cat([tokens, torch.zeros(64, 1)], dim=1).unsqueeze(0)
    self_elo = rng.choice([600, 1100, 1500, 1900, 2400])
    oppo_elo = rng.choice([800, 1200, 1500, 2000])
    with torch.no_grad():
        lm, lv, _ = model(tokens, torch.tensor([self_elo]), torch.tensor([oppo_elo]))
    mask = get_legal_moves_mask(board, moves_dict)
    logits = lm[0].masked_fill(~mask, float("-inf"))
    probs = torch.softmax(logits, -1)
    top = torch.topk(probs, k=min(5, int(mask.sum())))
    top_moves = []
    for p, i in zip(top.values.tolist(), top.indices.tolist()):
        u = all_moves[i]
        if board.turn == chess.BLACK:
            u = mirror_move(u)
        top_moves.append([u, p])
    wdl = torch.softmax(lv[0], -1).tolist()  # loss, draw, win for side to move
    cases.append({"start": replay.root().fen(), "moves": moves, "fen": board.fen(), "selfElo": self_elo, "oppoElo": oppo_elo,
                  "top": top_moves, "value": wdl})
(out / "reference.json").write_text(json.dumps(cases, indent=1))
print("wrote", len(cases), "cases")
