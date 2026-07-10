"""FDR v1: per-patch Benjamini-Hochberg selection on chi_3 p-values.

Answers Minh's "define a threshold per sample" via multiple-testing, using our
MEASURED chi_3 null (not manufactured knockoffs — see reasoning: knockoffs are
for the no-analytic-null, supervised, dataset-level case; we have a measured
null and need per-input unsupervised selection, so BH-per-patch is the match).

Per patch:
  1. block norms gn (256,)
  2. robust per-patch noise scale sigma_hat from the low-norm bulk:
     under chi_3, median(norm)/sqrt(chi2_3.ppf(0.5)) estimates sigma; use a LOW
     quantile (e.g. 40th pct) so a few active blocks don't inflate it.
  3. p_i = 1 - CDF_chi3(norm_i / sigma_hat)   (one-sided upper-tail: is block above noise?)
  4. Benjamini-Hochberg at level alpha -> variable-size selected set
     (BH: sort p asc, largest i with p_(i) <= (i/m)*alpha, keep those)
  Also BH-Yekutieli (dependence-robust) variant.

Compare to fixed-k, chi-floor, p-less on the SAME k* ground truth (script 06/07):
corr with k*, over/under-select, total selection error, at matched mean-L0.
Sweep alpha to trace the FDR curve; report the alpha that matches mean-L0 ~ mean(k*).
"""
import pathlib, sys, json
import numpy as np
import torch
from scipy import stats

RES = pathlib.Path("/Users/minh/local/ai-research/2026-07-09-bsf-distribution-aware-thresholds/results")
ROOT = pathlib.Path("/Users/minh/local/ai-research/2026-07-09-bsf-distribution-aware-thresholds/block-sparse-featurizer")
sys.path.insert(0, str(ROOT)); import bsf

device="cpu"
x_np = np.load(RES/"acts.npy")
idx = np.random.RandomState(0).permutation(len(x_np))[:12000]
xs = torch.as_tensor(x_np[idx], dtype=torch.float32)
m = bsf.VanillaBSF(d=768, n_groups=256, group_size=3, l0=8)
m.load_state_dict(torch.load(RES/"models_e2"/"topk_k8.pt", map_location=device)); m.eval()

with torch.no_grad():
    a = (xs @ m.W_enc + m.b_enc).reshape(-1,256,3)
    gn = a.norm(dim=-1).numpy()
    Wb = m.W_dec.reshape(256,3,-1)
    contrib = torch.einsum('ngk,gkd->ngd', a, Wb)
    order = torch.argsort(-a.norm(dim=-1), dim=1)
    cs = torch.gather(contrib,1,order.unsqueeze(-1).expand(-1,-1,768)).cumsum(1).numpy()
N,G = gn.shape

# ---- k* ground truth (peak-based, same as 06/07) ----
ks_grid = np.array(list(range(1,41))+[45,50,60,80,100,140,200,256])
xsn = xs.numpy(); sst = ((xsn-xsn.mean(1,keepdims=True))**2).sum(1)
r2c = np.stack([1-((xsn-cs[:,k-1])**2).sum(1)/np.maximum(sst,1e-9) for k in ks_grid],1)
peak=r2c.max(1); tgt=0.95*peak
kstar=np.array([ks_grid[np.argmax(r2c[i]>=tgt[i])] for i in range(N)],float)

DF=3
# robust per-patch sigma from low-norm bulk (40th percentile / chi_3 median-ish anchor)
def sigma_hat(row, q=0.4):
    anchor = np.quantile(row, q)
    return anchor / stats.chi.ppf(q, DF)     # invert chi_3 cdf at that quantile
def pvals(row):
    s = max(sigma_hat(row), 1e-6)
    return stats.chi.sf(row/s, DF)           # upper-tail p under chi_3
def bh(p, alpha, c=1.0):
    # c=1 -> BH; c=H_m (harmonic) -> Benjamini-Yekutieli (dependence-robust)
    mm=len(p); o=np.argsort(p); ps=p[o]
    thresh=(np.arange(1,mm+1)/(mm*c))*alpha
    below=np.where(ps<=thresh)[0]
    if len(below)==0: return np.zeros(mm,bool)
    kmax=below.max(); sel=np.zeros(mm,bool); sel[o[:kmax+1]]=True; return sel
H_m = np.log(G)+0.5772  # harmonic approx for BY

def counts(alpha, by=False):
    c = H_m if by else 1.0
    return np.array([bh(pvals(gn[i]), alpha, c).sum() for i in range(N)])

# tune alpha so mean selected ~ mean(k*)
target=kstar.mean()
def tune(by):
    lo,hi=1e-4,0.999
    for _ in range(22):
        mid=(lo+hi)/2; cc=counts(mid,by).mean()
        if cc<target: lo=mid
        else: hi=mid
    return (lo+hi)/2
a_bh=tune(False); a_by=tune(True)

def score(nsel,name):
    nsel=nsel.astype(float)
    over=np.maximum(0,nsel-kstar); tr=np.maximum(0,kstar-nsel)
    corr=float(np.corrcoef(nsel,kstar)[0,1]) if nsel.std()>0 else 0.0
    print(f"{name:22s} sel={nsel.mean():5.2f}±{nsel.std():4.2f} corr={corr:+.3f} err={(over+tr).mean():.2f}")
    return dict(mean_selected=round(float(nsel.mean()),2),std=round(float(nsel.std()),2),
                corr_with_kstar=round(corr,3),total_error=round(float((over+tr).mean()),2))

rep={"kstar_mean":round(float(target),2),"kstar_CV":round(float(kstar.std()/kstar.mean()),3),
     "alpha_bh_matched":round(float(a_bh),4),"alpha_by_matched":round(float(a_by),4),
     "prior_baselines":{"perpatch_chi":0.576,"p_less_L1":0.695,"fixed_k":0.0},"results":{}}
print(f"k*: mean={target:.2f} CV={kstar.std()/kstar.mean():.3f}")
print(f"matched alpha: BH={a_bh:.4f}  BY={a_by:.4f}\n")
rep["results"]["BH_chi_matched"]=score(counts(a_bh,False),f"BH(a={a_bh:.3f})")
rep["results"]["BY_chi_matched"]=score(counts(a_by,True), f"BY(a={a_by:.3f})")
# also fixed alpha=0.1 (the "principled default" — no tuning to k*)
rep["results"]["BH_chi_alpha0.1"]=score(counts(0.1,False),"BH(a=0.10 default)")
rep["alpha_interpretation"]="alpha = per-patch target FDR (expected false-active fraction among selected blocks). Dimensionless, transferable."

json.dump(rep, open(RES/"fdr_v1.json","w"),indent=2)
print("\n"+json.dumps(rep,indent=2))
