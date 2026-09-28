"""Measure how far the int8 and fp16 exports drift from the fp32 checkpoint.

Usage: python tools/quantcheck.py <maia3 repo dir> <maia3-5m.pt> [positions]

Applies the same quantization as tools/export_weights.py to the upstream
PyTorch model, then compares each variant with fp32 over random positions from
random games (default 400): top-move agreement and mean KL divergence of the
legal-move distribution. The published int8 weights agree on 99.5% of 400
positions with mean KL divergence about 9e-5.
"""
import copy, random, sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import chess

args = [a for a in sys.argv[1:] if not a.startswith("--")]
if len(args) < 2:
    sys.exit(__doc__)
repo, ckpt = Path(args[0]), Path(args[1])
N = int(args[2]) if len(args) > 2 else 400
sys.path.insert(0, str(repo))
from maia3.model_registry import resolve_model_spec  # noqa: E402
from maia3.models import MAIA3Model  # noqa: E402
from maia3.dataset import tokenize_board, get_legal_moves_mask  # noqa: E402
from maia3.utils import get_all_possible_moves  # noqa: E402

spec = resolve_model_spec("maia3-5m")
base = MAIA3Model(SimpleNamespace(**spec.config))
state = torch.load(ckpt, map_location="cpu", weights_only=True)
state = state.get("model_state_dict", state)
base.load_state_dict({k.replace("smolgen", "gab"): v for k, v in state.items()}, strict=False)
base.eval()


def variant(fmt):
    m = copy.deepcopy(base)
    with torch.no_grad():
        for _, p in m.named_parameters():
            a = p.detach().numpy()
            if fmt == "int8" and a.ndim == 2 and a.size >= 4096:
                s = np.abs(a).max(1) / 127
                s[s == 0] = 1
                q = np.clip(np.round(a / s[:, None]), -127, 127)
                p.copy_(torch.tensor(q * s[:, None], dtype=torch.float32))
            else:
                p.copy_(torch.tensor(a.astype(np.float16).astype(np.float32)))
    return m


variants = {"fp16": variant("fp16"), "int8": variant("int8")}
all_moves = get_all_possible_moves()
moves_dict = {m: i for i, m in enumerate(all_moves)}
rng = random.Random(1)
agree = {k: 0 for k in variants}
kl = {k: 0.0 for k in variants}
for _ in range(N):
    board = chess.Board()
    for _ in range(rng.randint(0, 50)):
        if board.is_game_over():
            break
        board.push(rng.choice(list(board.legal_moves)))
    if board.is_game_over():
        board = chess.Board()
    tokens = torch.cat([tokenize_board(board)] * 8, 1)
    tokens = torch.cat([tokens, torch.zeros(64, 1)], 1)[None]
    elo = torch.tensor([rng.choice([800, 1200, 1600, 2000])])
    mask = get_legal_moves_mask(board, moves_dict)
    with torch.no_grad():
        ref = torch.log_softmax(base(tokens, elo, elo)[0][0].masked_fill(~mask, -1e9), -1)
        for k, m in variants.items():
            out = torch.log_softmax(m(tokens, elo, elo)[0][0].masked_fill(~mask, -1e9), -1)
            agree[k] += int(out.argmax() == ref.argmax())
            kl[k] += float((ref.exp() * (ref - out)).sum())
for k in variants:
    print(f"{k}: top-move agreement {agree[k] / N:.4f}, mean KL {kl[k] / N:.2e} over {N} positions")
