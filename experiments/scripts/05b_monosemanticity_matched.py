"""Step 5b: matched-L0 monosemanticity — train topk_k12 to match chi_q2.0 (L0~12).

The q->L0 map for chi-floor is unstable (q2.0->11.9, q2.3->4.4, q2.9->18.9;
dead blocks swing 2..145), so instead of tuning chi to L0=8 we match top-k UP to
chi_q2.0's operating point (L0~12). Both models then compared at equal budget.
Reuses metrics from step 5. Instability of chi's budget targeting is logged as a
finding in its own right.
"""
import pathlib, sys, json, time
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]   # repo root
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
        super().__init__(d, n_groups, group_size, l0=8); self.q = float(q)
    def encode(self, x):
        a = (x @ self.W_enc + self.b_enc).reshape(-1, self.n_groups, self.group_size)
        gn = a.norm(dim=-1); s = gn.median(1, keepdim=True).values.clamp_min(1e-6)
        return a * (gn >= self.q * s).to(a.dtype).unsqueeze(-1)


# train topk_k12 (match chi_q2.0 L0~12)
tk = bsf.VanillaBSF(d=d, n_groups=256, group_size=3, l0=12).to(device)
p = RES / "models_e2" / "topk_k12.pt"
if p.exists(): tk.load_state_dict(torch.load(p, map_location=device))
else:
    print("=== training topk_k12 ==="); train(tk, x_np, epochs=200, lr=3e-3, device=device, log_every=50)
    torch.save(tk.state_dict(), p)
tk.eval()

chi = ChiFloorBSF(d=d, n_groups=256, group_size=3, q=2.0).to(device)
chi.load_state_dict(torch.load(RES / "models_e2" / "chi_q2.0.pt", map_location=device)); chi.eval()

MODELS = {"topk_k12": tk, "chi_q2.0": chi}

@torch.no_grad()
def codes(model):
    z = model.encode(xs); gn = z.norm(dim=-1)
    r2 = 1.0 - float(((xs - model.decode(z)) ** 2).sum()) / SS_tot
    l0 = float((gn > 1e-6).float().sum(1).mean())
    return z, gn, r2, l0

@torch.no_grad()
def passenger_rate(model, gn, z, n=400, thresh=0.01):
    rows = np.random.RandomState(3).permutation(gn.shape[0])[:n]; ps = tot = 0
    for r in rows:
        active = torch.nonzero(gn[r] > 1e-6).flatten()
        if not len(active): continue
        xr = xs[r:r+1]; zf = z[r:r+1].clone()
        base = float(((xr - model.decode(zf)) ** 2).sum())
        var = float(((xr - xr.mean()) ** 2).sum()) + 1e-9
        for b in active:
            zz = zf.clone(); zz[0, b] = 0
            ps += int((float(((xr - model.decode(zz)) ** 2).sum()) - base) / var < thresh); tot += 1
    return ps / max(tot, 1)

def coherence(gn_np, min_fire=50, topn=50):
    xn = xs_np / (np.linalg.norm(xs_np, axis=1, keepdims=True) + 1e-9)
    freqs, cohs = [], []
    for b in range(gn_np.shape[1]):
        col = gn_np[:, b]; fire = (col > 1e-6).sum()
        if fire < min_fire: continue
        freqs.append(fire / len(col))
        V = xn[np.argsort(-col)[:topn]]; S = V @ V.T
        cohs.append((S.sum() - np.trace(S)) / (topn * (topn - 1)))
    return len(freqs), float(np.median(freqs)), float(np.mean(cohs))

report = {}
for name, m in MODELS.items():
    z, gn, r2, l0 = codes(m); gn_np = gn.cpu().numpy()
    na, mf, mc = coherence(gn_np); pr = passenger_rate(m, gn, z)
    report[name] = dict(R2=round(r2,4), meanL0=round(l0,2), n_alive=na,
                        M1_passenger_rate=round(pr,4), M2_median_fire_freq=round(mf,4),
                        M3_topfire_coherence=round(mc,4))
    print(f"{name}: R2={r2:.3f} L0={l0:.1f} alive={na} passenger={pr:.3f} freq={mf:.3f} coh={mc:.4f}")

json.dump(report, open(RES / "monosemanticity_matched.json", "w"), indent=2)
print("\n" + json.dumps(report, indent=2))
