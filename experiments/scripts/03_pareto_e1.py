"""Step 3 (E1): apply selectors to FROZEN magnitudes, trace R2-vs-mean-L0 Pareto.

Selectors (all read the same pre-gate block codes `a` (N,G,K) of a top-k-trained
model, so the ONLY thing that differs is the selection rule):

  topk(k)          fixed count per patch (baseline)           -- sweep k
  min_z(p)         per-patch, max-referenced: ||z|| >= p*rowmax -- sweep p  (min-p analogue)
  global_floor(m)  single global norm threshold                -- sweep target mean-L0
  perpatch_chi(q)  per-patch noise-ref: ||z|| >= q*sigma_i,
                   sigma_i = median(row norms) (robust noise scale) -- sweep q

THEORY NOTE (why global_floor is special): for an orthonormal decoder,
reconstruction error = sum of squared norms of DROPPED blocks. Minimising total
error at a fixed TOTAL block budget B = sum_i m_i is a water-filling: rank ALL
(patch, block) pairs by norm^2 and keep the top B. That is exactly a single
GLOBAL threshold. So global_floor is the R2-OPTIMAL variable-width selector at
matched mean-L0 (exact for Grassmannian, approx for Vanilla). top-k is provably
suboptimal (wastes budget on low-energy blocks that clear a simple patch's local
top-k but lose to the 9th block of a rich patch). min_z / perpatch_chi are
adaptive but NOT recon-optimal -- interesting to see where they land vs the
global envelope.
"""
import pathlib, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[2]   # repo root
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT))
import bsf

device = "cpu"
x_all = np.load(RES / "acts.npy")
summary = json.load(open(RES / "models" / "summary.json"))
rng = np.random.RandomState(0)
idx = rng.permutation(len(x_all))[:20000]
x = torch.as_tensor(x_all[idx], dtype=torch.float32, device=device)
SS_tot = float(((x - x.mean(0, keepdim=True)) ** 2).sum())

def load(name):
    c = summary[name]
    m = getattr(bsf, c["cls"])(d=x.shape[1], **c["kw"]).to(device)
    m.load_state_dict(torch.load(RES / "models" / f"{name}.pt", map_location=device))
    return m.eval()

@torch.no_grad()
def pregate_code(model, t):
    """(N,G,K) signed pre-gate block codes for each featurizer."""
    if isinstance(model, bsf.GrassmannianBSF):
        atoms = model.decoder_atoms(); a = torch.exp(model.log_gamma) * (t @ atoms.t())
    elif isinstance(model, bsf.GroupLassoBSF):
        a = model.preact(t).reshape(t.shape[0], -1)
    else:
        a = (t @ model.W_enc + model.b_enc)
    return a.reshape(t.shape[0], model.n_groups, model.group_size)

@torch.no_grad()
def r2_of_mask(model, a, mask):
    z = a * mask.unsqueeze(-1)
    xhat = model.decode(z)
    ss_res = float(((x - xhat) ** 2).sum())
    return 1.0 - ss_res / SS_tot, float(mask.float().sum(1).mean())

def sweep(model, a):
    gn = a.norm(dim=-1)                     # (N,G)
    N, G = gn.shape
    rowmax = gn.max(1, keepdim=True).values
    med = gn.median(1, keepdim=True).values  # robust per-patch noise scale
    allflat = gn.flatten()
    curves = {"topk": [], "min_z": [], "global_floor": [], "perpatch_chi": []}

    for k in [1,2,3,4,5,6,7,8,10,12,15,20,25,30]:
        idx_k = gn.topk(k, dim=1).indices
        m = torch.zeros_like(gn).scatter_(1, idx_k, 1.0).bool()
        curves["topk"].append(r2_of_mask(model, a, m))

    for p in np.linspace(0.02, 0.75, 16):
        m = (gn >= p * rowmax)
        curves["min_z"].append(r2_of_mask(model, a, m))

    for target in [2,3,4,5,6,7,8,10,12,15,20,25,30]:
        tau = torch.quantile(allflat, 1.0 - target / G)
        m = (gn >= tau)
        curves["global_floor"].append(r2_of_mask(model, a, m))

    for q in np.linspace(1.0, 6.0, 16):
        m = (gn >= q * med)
        curves["perpatch_chi"].append(r2_of_mask(model, a, m))

    return {k: sorted(v) for k, v in curves.items()}

MODELS = ["vanilla_topk", "grass_topk"]
out = {}
fig, axes = plt.subplots(1, len(MODELS), figsize=(15, 6))
for ax, name in zip(axes, MODELS):
    model = load(name)
    a = pregate_code(model, x)
    curves = sweep(model, a)
    out[name] = curves
    for sel, pts in curves.items():
        L0 = [p[1] for p in pts]; R2 = [p[0] for p in pts]
        style = dict(topk=("k","o","-"), min_z=("tab:orange","s","--"),
                     global_floor=("tab:green","^","-"), perpatch_chi=("tab:blue","d","--"))[sel]
        ax.plot(L0, R2, color=style[0], marker=style[1], ls=style[2], ms=5, label=sel)
    ax.set_xlim(0, 30); ax.set_xlabel("mean L0 (blocks/patch)"); ax.set_ylabel("recon R2")
    ax.set_title(f"{name}\nR2 vs mean-L0 Pareto (frozen magnitudes)")
    ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(RES / "pareto_E1.png", dpi=115)

# numeric: at matched mean-L0 ~= 8, compare selectors (interpolate)
def r2_at(pts, target=8.0):
    L0 = np.array([p[1] for p in pts]); R2 = np.array([p[0] for p in pts])
    o = np.argsort(L0)
    return float(np.interp(target, L0[o], R2[o]))

deltas = {}
for name, curves in out.items():
    base = r2_at(curves["topk"])
    deltas[name] = {sel: dict(R2_at_L0_8=round(r2_at(curves[sel]),4),
                               delta_vs_topk=round(r2_at(curves[sel]) - base, 4))
                    for sel in curves}
json.dump(deltas, open(RES / "pareto_E1_deltas.json", "w"), indent=2)
print("saved results/pareto_E1.png")
print(json.dumps(deltas, indent=2))
