"""Test A-extended: add p-less, DiffSampling-cut, Top-H to the k*-correlation table.

Same frozen topk_k8 model, same peak-based k* ground truth as script 06. Question:
which distribution-shape estimator best predicts each patch's TRUE support k*?
Existing baselines (from 06): fixed_k corr 0, perpatch_chi 0.576, min_z 0.517, global 0.331.

NEW selectors (all parameter-free except Top-H's single alpha):
  p_less_L1   : shares p_i=norm_i/sum(norm); threshold L=sum(p^2); keep p_i>=L
  p_less_E    : energy shares p_i=norm_i^2/sum(norm^2) (BSF recon is quadratic); keep p_i>=L
  diffsample  : sort desc, cut after largest gap (norm_i - norm_{i+1})
  top_h(alpha): sort desc, keep adding until entropy of retained renorm shares >= alpha*H(full)
                alpha tuned so mean selected ~ mean(k*)
"""
import pathlib, sys, json
import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]   # repo root
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT))
import bsf

device = "cpu"
x_np = np.load(RES / "acts.npy")
idx = np.random.RandomState(0).permutation(len(x_np))[:12000]
xs = torch.as_tensor(x_np[idx], dtype=torch.float32)
m = bsf.VanillaBSF(d=768, n_groups=256, group_size=3, l0=8)
m.load_state_dict(torch.load(RES / "models_e2" / "topk_k8.pt", map_location=device)); m.eval()

with torch.no_grad():
    a = (xs @ m.W_enc + m.b_enc).reshape(-1, 256, 3)
    gn = a.norm(dim=-1).numpy()               # (N,256) pre-gate norms
    Wb = m.W_dec.reshape(256, 3, -1)
    contrib = torch.einsum('ngk,gkd->ngd', a, Wb)
    order = torch.argsort(-a.norm(dim=-1), dim=1)
    cs = torch.gather(contrib, 1, order.unsqueeze(-1).expand(-1, -1, 768)).cumsum(1).numpy()
N, G = gn.shape

# ---- k* : 95% of per-patch PEAK R2 (same as script 06) ----
ks_grid = np.array(list(range(1, 41)) + [45,50,60,80,100,140,200,256])
xs_np = xs.numpy()
sst = ((xs_np - xs_np.mean(1, keepdims=True))**2).sum(1)
r2c = np.zeros((N, len(ks_grid)))
for j, k in enumerate(ks_grid):
    ssr = ((xs_np - cs[:, k-1])**2).sum(1)
    r2c[:, j] = 1 - ssr/np.maximum(sst, 1e-9)
peak = r2c.max(1); target = 0.95*peak
kstar = np.array([ks_grid[np.argmax(r2c[i] >= target[i])] for i in range(N)], float)
print(f"k*: mean={kstar.mean():.2f} CV={kstar.std()/kstar.mean():.3f}")

sorted_gn = np.sort(gn, axis=1)[:, ::-1]                      # desc
# ---- selectors -> per-patch selected COUNT ----
def c_fixed(k): return np.full(N, k)
def c_pless(power):
    # shares over norm^power, threshold = sum(share^2), keep share>=thr
    w = gn**power
    p = w / w.sum(1, keepdims=True)
    L = (p**2).sum(1, keepdims=True)
    return (p >= L).sum(1)
def c_diff():
    # cut after largest gap in sorted desc sequence
    gaps = sorted_gn[:, :-1] - sorted_gn[:, 1:]              # (N,255)
    return gaps.argmax(1) + 1
def c_toph(alpha):
    # keep adding (desc) until entropy of retained renorm shares >= alpha*H(full)
    p_full = gn / gn.sum(1, keepdims=True)
    H_full = -(p_full*np.log(p_full+1e-12)).sum(1)           # (N,)
    counts = np.zeros(N, int)
    sp = np.sort(gn, axis=1)[:, ::-1]
    csum = np.cumsum(sp, axis=1)
    # entropy of top-k renormalized set, vectorized over k via cumulative
    for i in range(N):
        tgt = alpha*H_full[i]
        for k in range(1, G+1):
            s = sp[i, :k]; ps = s/csum[i, k-1]
            Hk = -(ps*np.log(ps+1e-12)).sum()
            if Hk >= tgt: counts[i]=k; break
        else: counts[i]=G
    return counts

target_mean = kstar.mean()
def tune_toph():
    lo, hi = 0.05, 0.99
    for _ in range(18):
        mid=(lo+hi)/2; c=c_toph(mid).mean()
        if c < target_mean: lo=mid          # higher alpha -> more kept
        else: hi=mid
    return (lo+hi)/2
alpha = tune_toph()

sel = {
    "fixed_k=5": c_fixed(int(round(target_mean))),
    "p_less_L1": c_pless(1.0),
    "p_less_E":  c_pless(2.0),
    "diffsample": c_diff(),
    f"top_h(a={alpha:.2f})": c_toph(alpha),
}

rep = {"kstar_mean": round(float(target_mean),2), "kstar_CV": round(float(kstar.std()/kstar.mean()),3),
       "prior_baselines": {"perpatch_chi": 0.576, "min_z": 0.517, "global_floor": 0.331, "fixed_k": 0.0},
       "selectors": {}}
for name, nsel in sel.items():
    nsel = nsel.astype(float)
    over = np.maximum(0, nsel-kstar); trunc = np.maximum(0, kstar-nsel)
    corr = float(np.corrcoef(nsel, kstar)[0,1]) if nsel.std()>0 else 0.0
    rep["selectors"][name] = dict(mean_selected=round(float(nsel.mean()),2),
        std_selected=round(float(nsel.std()),2), corr_with_kstar=round(corr,3),
        mean_over=round(float(over.mean()),2), mean_trunc=round(float(trunc.mean()),2),
        total_error=round(float((over+trunc).mean()),2))
    print(f"{name:16s} sel={nsel.mean():5.2f}±{nsel.std():4.2f} corr={corr:+.3f} err={float((over+trunc).mean()):.2f}")

json.dump(rep, open(RES/"testA_extended.json","w"), indent=2)
print("\n"+json.dumps(rep, indent=2))
