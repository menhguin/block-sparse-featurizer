"""Step 4 (E2 / robustness): does chi-floor give operating-point robustness that
fixed top-k lacks?  (Minh's reframe, 2026-07-09)

FAITHFUL TRANSFER of top-nsigma's real selling point: not higher R2, but
STABILITY across an operating condition that breaks the fixed-parameter baseline.
top-nsigma: one setting stable across TEMPERATURE. Here: one chi-floor setting
stable across SPARSITY BUDGET (mean-L0), where top-k needs retraining per point.

Design:
  * Train Vanilla top-k at k in {4, 8, 16}  -> the NATIVE ENVELOPE (best R2 when
    the dictionary is trained for that exact budget).
  * Train ChiFloorBSF (chi-floor selection IN the training loop) at a couple of
    fixed thresholds q.
  * Eval EVERY trained model by sweeping ITS OWN selector across L0 in [2,30]
    (top-k: vary k at eval; chi: vary q at eval), tracing a single-model curve.
  * Robustness = how closely a single model's eval-sweep tracks the native
    envelope. Fixed-k specialises to its training budget (should sag off-point);
    chi-floor's stable statistical threshold should track flatter.

Outputs: train_eval_matrix.json (R2 at L0={4,8,16} for every trained model),
robustness_curves.png (native envelope + each single-model eval-sweep).

NOTE: ChiFloorBSF is defined inline (subclass of VanillaBSF) to keep the repo
package clean during exploration. Selection mask is hard/detached exactly like
group_topk, so gradients flow through magnitudes the same way. Per-patch robust
noise scale sigma_i = median of that patch's block norms; keep block g iff
||z_g|| >= q * sigma_i.
"""
import pathlib, sys, json, time
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
(RES / "models_e2").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))
import bsf
from bsf.base import unit_blocks
from bsf.train import train

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
x_np = np.load(RES / "acts.npy"); d = x_np.shape[1]
xt = torch.as_tensor(x_np, dtype=torch.float32, device=device)
sub_idx = np.random.RandomState(1).permutation(len(x_np))[:20000]
xsub = torch.as_tensor(x_np[sub_idx], dtype=torch.float32, device=device)
SS_tot = float(((xsub - xsub.mean(0, keepdim=True)) ** 2).sum())
G = 256


class ChiFloorBSF(bsf.VanillaBSF):
    """Vanilla encoder/decoder, but selection = per-patch noise-referenced
    chi-floor instead of block-TopK. q is a fixed threshold multiplier."""
    def __init__(self, d, n_groups, group_size=3, q=3.0):
        super().__init__(d, n_groups, group_size, l0=8)  # l0 unused
        self.q = float(q)

    def encode(self, x):
        a = (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)
        gn = a.norm(dim=-1)                                   # (N,G)
        sigma = gn.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        mask = (gn >= self.q * sigma).to(a.dtype)             # hard, detached
        return a * mask.unsqueeze(-1)


@torch.no_grad()
def pregate(model, t):
    a = (t @ model.W_enc + model.b_enc)
    return a.reshape(t.shape[0], model.n_groups, model.group_size)

@torch.no_grad()
def r2_mask(model, a, mask):
    z = a * mask.unsqueeze(-1)
    ss_res = float(((xsub - model.decode(z)) ** 2).sum())
    return 1.0 - ss_res / SS_tot, float(mask.float().sum(1).mean())

@torch.no_grad()
def eval_sweep(model):
    """Sweep the model's OWN selector -> list of (meanL0, R2)."""
    a = pregate(model, xsub); gn = a.norm(dim=-1)
    pts = []
    if isinstance(model, ChiFloorBSF):
        med = gn.median(1, keepdim=True).values
        for q in np.linspace(1.0, 6.0, 22):
            pts.append(r2_mask(model, a, (gn >= q * med)))
    else:
        for k in [1,2,3,4,5,6,7,8,10,12,14,16,20,24,30]:
            idxk = gn.topk(k, 1).indices
            pts.append(r2_mask(model, a, torch.zeros_like(gn).scatter_(1, idxk, 1.0)))
    return sorted(pts)

def r2_at(pts, target):
    L0 = np.array([p[1] for p in pts]); R2 = np.array([p[0] for p in pts])
    o = np.argsort(L0); return float(np.interp(target, L0[o], R2[o]))

# ---- train ----
trained = {}
specs = [("topk_k4","VanillaBSF",dict(l0=4)),
         ("topk_k8","VanillaBSF",dict(l0=8)),
         ("topk_k16","VanillaBSF",dict(l0=16)),
         ("chi_q2.5","ChiFloorBSF",dict(q=2.5)),
         ("chi_q3.5","ChiFloorBSF",dict(q=3.5))]
for name, cls, kw in specs:
    t0=time.time()
    ctor = ChiFloorBSF if cls=="ChiFloorBSF" else bsf.VanillaBSF
    m = ctor(d=d, n_groups=256, group_size=3, **kw).to(device)
    print(f"\n=== {name} ({cls} {kw}) ===")
    train(m, x_np, epochs=200, lr=3e-3, device=device, log_every=50)
    trained[name]=m
    torch.save(m.state_dict(), RES/"models_e2"/f"{name}.pt")
    print(f"  trained in {time.time()-t0:.0f}s")

# ---- native envelope + single-model curves ----
curves = {name: eval_sweep(m) for name,m in trained.items()}
TARGETS=[4,8,16]
native = {4:"topk_k4",8:"topk_k8",16:"topk_k16"}
matrix={}
for name,pts in curves.items():
    matrix[name]={f"L0_{t}": round(r2_at(pts,t),4) for t in TARGETS}
matrix["_native_envelope"]={f"L0_{t}": matrix[native[t]][f"L0_{t}"] for t in TARGETS}
json.dump(matrix, open(RES/"train_eval_matrix.json","w"), indent=2)

plt.figure(figsize=(9,6))
for name,pts in curves.items():
    L0=[p[1] for p in pts]; R2=[p[0] for p in pts]
    ls = "--" if name.startswith("chi") else "-"
    plt.plot(L0,R2,marker="o",ms=4,ls=ls,label=name)
env_L0=TARGETS; env_R2=[matrix["_native_envelope"][f"L0_{t}"] for t in TARGETS]
plt.plot(env_L0,env_R2,"k*",ms=18,label="native top-k envelope",zorder=10)
plt.xlim(0,30); plt.xlabel("mean L0"); plt.ylabel("recon R2")
plt.title("Operating-point robustness: single-model eval-sweep vs native envelope")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(RES/"robustness_curves.png", dpi=115)
print("\nsaved robustness_curves.png + train_eval_matrix.json")
print(json.dumps(matrix, indent=2))
