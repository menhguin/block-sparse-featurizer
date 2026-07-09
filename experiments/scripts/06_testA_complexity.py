"""Test A: complexity-shift / selection-correctness vs per-patch TRUE support.

The reframe (Minh, 2026-07-09): E1/E2 tested in-distribution R²-average, which
AVERAGES OVER the complexity distribution and hides tail errors. top-nsigma's
actual win was ROBUSTNESS to a shift the fixed rule isn't tuned for. Effective-
support CV here = 0.675 (measured) => there IS real per-patch complexity variance;
the question is whether fixed-k mis-selects on the tails while chi-floor tracks.

GROUND TRUTH (rule-agnostic ruler): per patch, greedily add blocks by descending
norm, record per-patch R² curve, define k*(i) = min #blocks to reach 95% of that
patch's OWN full-256 R². This is a property of the patch's activation geometry vs
the frozen dictionary — not of any selection rule. Both fixed-k and chi rank by
norm, so k* is NEUTRAL for the fixed-vs-adaptive comparison we care about
(caveat: mildly sympathetic to norm-based selection generally — stated, not fatal).

Then per patch, per selector, compare selected count n_rule to k*:
  over_select = max(0, n_rule - k*)   # noise admitted (waste / pollution)
  truncate    = max(0, k* - n_rule)   # real signal dropped
  r2_gap      = R2(top-k* blocks) - R2(top-n_rule blocks)   # actual recon cost
Reported per complexity TERCILE (simple / medium / complex by k*).

Selectors compared at MATCHED mean-L0 ~= 8 (tune each rule's param to hit mean
selected count ≈ mean k* ≈ 8 so the comparison is about ALLOCATION not budget):
  fixed_k=8          | chi_floor(q*) | min_z(p*) | global_floor(tau*)
Headline plot: x=k* (true support), y=selected count, one line/rule. fixed-k is a
flat line at 8; complexity-aware rules track the diagonal.
"""
import pathlib, sys, json
import numpy as np
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT))
import bsf

device = "cpu"
x_np = np.load(RES / "acts.npy")
idx = np.random.RandomState(0).permutation(len(x_np))[:12000]   # 12k patches, keep greedy loop cheap
xs = torch.as_tensor(x_np[idx], dtype=torch.float32, device=device)

m = bsf.VanillaBSF(d=768, n_groups=256, group_size=3, l0=8).to(device)
m.load_state_dict(torch.load(RES / "models_e2" / "topk_k8.pt", map_location=device))
m.eval()

with torch.no_grad():
    a = (xs @ m.W_enc + m.b_enc).reshape(-1, 256, 3)      # (N,256,3) pre-gate signed codes
    gn = a.norm(dim=-1)                                    # (N,256)
N, G = gn.shape

# ---- GROUND TRUTH k* via per-patch greedy R2 saturation ----
# For each patch: order blocks desc by norm, cumulatively decode, find count to hit
# 95% of full-256 per-patch R2. Vectorised over patches by cumulative masks is heavy;
# do it in chunks.
order = torch.argsort(-gn, dim=1)                          # (N,256)
W = m.W_dec                                                # (768? ) decoder (G*K, d)
@torch.no_grad()
def per_patch_r2_curve(chunk_a, chunk_order, xrow):
    # chunk_a (n,256,3), chunk_order (n,256), xrow (n,768)
    n = chunk_a.shape[0]
    ss_tot = ((xrow - xrow.mean(1, keepdim=True))**2).sum(1).clamp_min(1e-9)  # (n,)
    # precompute per-block contribution vectors: z_g @ W_block -> (n,256,768)
    # decode of a single block g: (a[:,g,:] @ W_dec_block_g)
    Wb = W.reshape(256, 3, -1)                             # (256,3,768)
    contrib = torch.einsum('ngk,gkd->ngd', chunk_a, Wb)    # (n,256,768) per-block recon
    # cumulative in norm-descending order
    ks = list(range(1,41)) + [45,50,60,80,100,140,200,256]
    r2s = np.zeros((n, len(ks)))
    idx_sorted = chunk_order                                # (n,256)
    # gather contribs in sorted order, cumsum
    contrib_sorted = torch.gather(contrib, 1, idx_sorted.unsqueeze(-1).expand(-1,-1,contrib.shape[-1]))
    cum = torch.cumsum(contrib_sorted, dim=1)               # (n,256,768)
    for j,k in enumerate(ks):
        recon = cum[:, k-1, :]
        ss_res = ((xrow - recon)**2).sum(1)
        r2s[:, j] = (1 - ss_res/ss_tot).cpu().numpy()
    return np.array(ks), r2s

ks_grid, r2_curves = [], []
CH = 2000
for s in range(0, N, CH):
    ks_grid, r2c = per_patch_r2_curve(a[s:s+CH], order[s:s+CH], xs[s:s+CH])
    r2_curves.append(r2c)
r2_curves = np.concatenate(r2_curves, 0)                    # (N, len(ks))
# NOTE: overcomplete non-orthogonal decoder => per-patch R2 PEAKS then DECLINES as
# low-norm blocks inject garbage. So "true support" is defined vs each patch's OWN
# ACHIEVABLE PEAK R2 (argmax over the grid), not full-256 (which is bad recon).
peak_r2 = r2_curves.max(1)                                   # best achievable per patch
peak_k = ks_grid[r2_curves.argmax(1)]                        # where peak occurs
# k* = smallest grid-k reaching 95% of that patch's PEAK
target = 0.95 * peak_r2
kstar = np.zeros(N)
for i in range(N):
    hit = np.where(r2_curves[i] >= target[i])[0]
    kstar[i] = ks_grid[hit[0]] if len(hit) else ks_grid[-1]
print(f"peak_r2: mean={peak_r2.mean():.3f}  peak_k: mean={peak_k.mean():.1f} median={np.median(peak_k):.0f}")
print(f"k* : mean={kstar.mean():.2f} median={np.median(kstar):.0f} "
      f"p5={np.percentile(kstar,5):.0f} p95={np.percentile(kstar,95):.0f} CV={kstar.std()/kstar.mean():.3f}")

# ---- selectors, each tuned to mean selected-count ~= mean(k*) for fair allocation test ----
gn_np = gn.cpu().numpy()
rowmax = gn_np.max(1, keepdims=True)
med = np.median(gn_np, axis=1, keepdims=True)
target_mean = kstar.mean()

def count_fixed(k): return np.full(N, k)
def count_minz(p): return (gn_np >= p*rowmax).sum(1)
def count_chi(q):  return (gn_np >= q*med).sum(1)
def count_global(tau): return (gn_np >= tau).sum(1)

# tune params to hit target_mean via bisection
def tune(fn, lo, hi):
    for _ in range(40):
        mid=(lo+hi)/2; c=fn(mid).mean()
        # higher param -> fewer selected, so invert
        if c>target_mean: lo=mid
        else: hi=mid
    return (lo+hi)/2
p_star   = tune(count_minz, 0.0, 1.0)
q_star   = tune(count_chi, 0.5, 8.0)
tau_star = tune(count_global, gn_np.min(), gn_np.max())
k_fixed  = int(round(target_mean))

sel = {
    f"fixed_k={k_fixed}": count_fixed(k_fixed),
    f"chi_floor(q={q_star:.2f})": count_chi(q_star),
    f"min_z(p={p_star:.2f})": count_minz(p_star),
    f"global_floor": count_global(tau_star),
}

# ---- tercile analysis ----
terc = np.digitize(kstar, np.quantile(kstar, [1/3, 2/3]))   # 0=simple,1=med,2=complex
names = ["simple","medium","complex"]
report = {"target_mean_L0": round(float(target_mean),2),
          "kstar": dict(mean=round(float(kstar.mean()),2), CV=round(float(kstar.std()/kstar.mean()),3),
                        p5=int(np.percentile(kstar,5)), p95=int(np.percentile(kstar,95))),
          "selectors": {}}
for sname, nsel in sel.items():
    over = np.maximum(0, nsel - kstar)
    trunc = np.maximum(0, kstar - nsel)
    per_terc = {}
    for t,tn in enumerate(names):
        mask = terc==t
        per_terc[tn] = dict(mean_kstar=round(float(kstar[mask].mean()),1),
                            mean_selected=round(float(nsel[mask].mean()),1),
                            over_select=round(float(over[mask].mean()),2),
                            truncate=round(float(trunc[mask].mean()),2))
    report["selectors"][sname]=dict(corr_with_kstar=round(float(np.corrcoef(nsel,kstar)[0,1]),3),
                                    mean_over=round(float(over.mean()),2),
                                    mean_trunc=round(float(trunc.mean()),2),
                                    total_error=round(float((over+trunc).mean()),2),
                                    terciles=per_terc)

json.dump(report, open(RES/"testA_selection_correctness.json","w"), indent=2)

# ---- headline plot: selected count vs k* ----
plt.figure(figsize=(9,7))
kk = np.linspace(kstar.min(), min(kstar.max(),60), 50)
for sname, nsel in sel.items():
    # bin selected count by k*
    binned=[nsel[(kstar>=kk[j])&(kstar<kk[j+1])].mean() if ((kstar>=kk[j])&(kstar<kk[j+1])).sum()>5 else np.nan
            for j in range(len(kk)-1)]
    plt.plot(kk[:-1], binned, marker="o", ms=3, label=sname)
plt.plot(kk, kk, "k:", lw=1.5, label="ideal (selected = k*)")
plt.xlim(0,60); plt.ylim(0,60)
plt.xlabel("k* (true per-patch support, 95% R²)"); plt.ylabel("mean selected count")
plt.title("Test A: does selection track true per-patch support?\n(fixed-k is horizontal by construction; complexity-aware rules track diagonal)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig(RES/"testA_selection_vs_support.png", dpi=115)
print("saved testA_selection_vs_support.png + testA_selection_correctness.json")
print(json.dumps(report, indent=2))
