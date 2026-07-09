"""Step 1: train the baseline substrates whose block-norm distributions we measure.

Three substrates (the shape you measure is biased by what you train with, so we
train all three and compare — see theory file §6):
  - vanilla_topk   : VanillaBSF, hard block-TopK   (l0=8)  -> expect NO gap
  - grass_topk     : GrassmannianBSF, hard block-TopK (l0=8) -> expect NO gap
  - grouplasso_soft: GroupLassoBSF (paper_version=True, true soft-threshold)
                     -> expect a gap by construction (shrinks inactive to ~0)

Saves state_dicts + achieved metrics to results/models/. CPU/MPS, a few min each.
"""
import pathlib, sys, json, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
(RES / "models").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import bsf
from bsf.train import train, recon_r2, l0_dead

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
x = np.load(RES / "acts.npy")
d = x.shape[1]
print(f"device={device}  x={x.shape}  d={d}")

CONFIGS = {
    "vanilla_topk":    dict(cls="VanillaBSF",     kw=dict(d=d, n_groups=256, group_size=3, l0=8),
                            train=dict(epochs=300, lr=3e-3)),
    "grass_topk":      dict(cls="GrassmannianBSF", kw=dict(d=d, n_groups=256, group_size=3, l0=8),
                            train=dict(epochs=300, lr=3e-3)),
    "grouplasso_soft": dict(cls="GroupLassoBSF",   kw=dict(d=d, n_groups=256, group_size=3,
                                                           target_l0=8, coef=1e-2, paper_version=True),
                            train=dict(epochs=300)),
}

summary = {}
for name, c in CONFIGS.items():
    t0 = time.time()
    model = getattr(bsf, c["cls"])(**c["kw"]).to(device)
    print(f"\n=== training {name} ({c['cls']}) ===")
    train(model, x, device=device, **c["train"])
    model.eval()
    sub = torch.as_tensor(x[np.random.permutation(len(x))[:20000]], dtype=torch.float32, device=device)
    r2 = recon_r2(model, sub); l0, dead = l0_dead(model, sub)
    torch.save(model.state_dict(), RES / "models" / f"{name}.pt")
    summary[name] = dict(cls=c["cls"], kw={k:v for k,v in c["kw"].items() if k!="d"},
                         train=c["train"], R2=round(r2,4), L0=round(l0,2),
                         dead=int(dead), secs=round(time.time()-t0,1))
    print(f"  -> R2={r2:.4f} L0={l0:.2f} dead={dead}/256  ({summary[name]['secs']}s)")

json.dump(summary, open(RES / "models" / "summary.json", "w"), indent=2)
print("\nsaved:", RES / "models" / "summary.json")
print(json.dumps(summary, indent=2))
