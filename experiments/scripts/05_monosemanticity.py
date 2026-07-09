"""Step 5 (fork a): monosemanticity / concept-purity — chi-floor vs top-k at matched L0.

Tests MECHANISM 2 (the thesis-aligned claim R2 can't see): top-k forces a fixed
count on every patch incl near-blank ones -> blocks trained to fire where their
concept is absent -> polluted features. chi-floor lets absent concepts stay off
-> each feature fires on a more coherent set of inputs.

Matched operating point: reuse topk_k8 (native L0=8) from models_e2; train a
chi-floor model tuned to land near L0=8 natively (q≈2.0) so the comparison is at
equal sparsity budget. Report R2 for both (chi likely slightly lower — that's the
tradeoff to weigh).

THREE metrics (designed to resist trivially flattering chi):

  M1 passenger-firing rate: for sampled patches, single-block ablation R2-drop for
     each ACTIVE block. Fraction of active firings whose ablation drop < 1% of the
     patch's total explained variance = "passengers" (forced-on but ~useless).
     mechanism-2 predicts top-k HIGHER (forced firings on blank patches).

  M2 feature selectivity: per alive block, activation frequency (frac of patches
     active). Monosemantic features are selective (fire rarely, strongly). Report
     median firing frequency over alive blocks. Lower = more selective.

  M3 top-firing input coherence (THE money metric): per alive block, take the
     top-50 patches by activation, compute mean pairwise cosine sim of their 768-d
     INPUT vectors x. High = the feature corresponds to a coherent region of input
     space = monosemantic. mechanism-2 predicts chi HIGHER.

FAIRNESS GUARDS: exclude dead/rarely-firing blocks (min 50 firings) from M2/M3;
report n_alive for both (chi concentrates into fewer features — if chi "wins" M3
only by having fewer alive blocks, that's a caveat, not a win). Also report a
RANDOM-selection control for M3 (coherence of 50 random patches) as the floor.
"""
import pathlib, sys, json, time
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT))
import bsf
from bsf.train import train

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
x_np = np.load(RES / "acts.npy"); d = x_np.shape[1]
idx = np.random.RandomState(2).permutation(len(x_np))[:20000]
xs_np = x_np[idx]
xs = torch.as_tensor(xs_np, dtype=torch.float32, device=device)
SS_tot = float(((xs - xs.mean(0, keepdim=True)) ** 2).sum())


class ChiFloorBSF(bsf.VanillaBSF):
    def __init__(self, d, n_groups, group_size=3, q=2.0):
        super().__init__(d, n_groups, group_size, l0=8)
        self.q = float(q)
    def encode(self, x):
        a = (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)
        gn = a.norm(dim=-1)
        sigma = gn.median(dim=1, keepdim=True).values.clamp_min(1e-6)
        return a * (gn >= self.q * sigma).to(a.dtype).unsqueeze(-1)


def load_vanilla(name):
    m = bsf.VanillaBSF(d=d, n_groups=256, group_size=3, l0=8).to(device)
    m.load_state_dict(torch.load(RES / "models_e2" / f"{name}.pt", map_location=device))
    return m.eval()

# --- get a chi model at L0~8 (train q=2.0 if not cached) ---
chi_path = RES / "models_e2" / "chi_q2.0.pt"
chi = ChiFloorBSF(d=d, n_groups=256, group_size=3, q=2.0).to(device)
if chi_path.exists():
    chi.load_state_dict(torch.load(chi_path, map_location=device))
else:
    print("=== training chi_q2.0 (target L0~8) ===")
    train(chi, x_np, epochs=200, lr=3e-3, device=device, log_every=50)
    torch.save(chi.state_dict(), chi_path)
chi.eval()
topk = load_vanilla("topk_k8")

MODELS = {"topk_k8": topk, "chi_q2.0": chi}

@torch.no_grad()
def codes_and_norms(model):
    z = model.encode(xs)                      # (N,G,K) gated
    gn = z.norm(dim=-1)                        # (N,G)
    xhat = model.decode(z)
    r2 = 1.0 - float(((xs - xhat) ** 2).sum()) / SS_tot
    l0 = float((gn > 1e-6).float().sum(1).mean())
    return z, gn, r2, l0

@torch.no_grad()
def passenger_rate(model, gn, z, n_patches=400, thresh=0.01):
    """Fraction of active blocks whose single-ablation recon-drop < thresh*patch_var."""
    rows = np.random.RandomState(3).permutation(gn.shape[0])[:n_patches]
    passengers = tot = 0
    for r in rows:
        active = torch.nonzero(gn[r] > 1e-6).flatten()
        if len(active) == 0: continue
        xr = xs[r:r+1]
        z_full = z[r:r+1].clone()
        base = float(((xr - model.decode(z_full)) ** 2).sum())
        var = float(((xr - xr.mean()) ** 2).sum()) + 1e-9
        for b in active:
            zz = z_full.clone(); zz[0, b] = 0
            drop = (float(((xr - model.decode(zz)) ** 2).sum()) - base) / var
            passengers += int(drop < thresh); tot += 1
    return passengers / max(tot, 1)

def coherence(gn_np, min_fire=50, topn=50):
    """M3 + M2 over alive blocks."""
    xn = xs_np / (np.linalg.norm(xs_np, axis=1, keepdims=True) + 1e-9)
    freqs, cohs = [], []
    for b in range(gn_np.shape[1]):
        col = gn_np[:, b]; fire = (col > 1e-6).sum()
        if fire < min_fire: continue
        freqs.append(fire / len(col))
        top = np.argsort(-col)[:topn]
        V = xn[top]
        S = V @ V.T
        cohs.append((S.sum() - np.trace(S)) / (len(top) * (len(top) - 1)))
    return len(freqs), float(np.median(freqs)), float(np.mean(cohs))

# random control for M3 floor
def random_coherence(topn=50, reps=200):
    xn = xs_np / (np.linalg.norm(xs_np, axis=1, keepdims=True) + 1e-9)
    vals = []
    rs = np.random.RandomState(9)
    for _ in range(reps):
        V = xn[rs.permutation(len(xn))[:topn]]
        S = V @ V.T
        vals.append((S.sum() - np.trace(S)) / (topn * (topn - 1)))
    return float(np.mean(vals))

report = {"_random_coherence_floor": round(random_coherence(), 4)}
for name, model in MODELS.items():
    z, gn, r2, l0 = codes_and_norms(model)
    gn_np = gn.cpu().numpy()
    n_alive, med_freq, mean_coh = coherence(gn_np)
    pr = passenger_rate(model, gn, z)
    report[name] = dict(R2=round(r2,4), meanL0=round(l0,2), n_alive=n_alive,
                        M1_passenger_rate=round(pr,4),
                        M2_median_fire_freq=round(med_freq,4),
                        M3_topfire_coherence=round(mean_coh,4))
    print(f"{name}: R2={r2:.3f} L0={l0:.1f} alive={n_alive} "
          f"passenger={pr:.3f} fire_freq={med_freq:.3f} coherence={mean_coh:.4f}")

json.dump(report, open(RES / "monosemanticity.json", "w"), indent=2)
print("\n" + json.dumps(report, indent=2))
