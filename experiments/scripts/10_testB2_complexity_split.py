"""Test B'': within-rabbit complexity-split transfer. ZERO dictionary shift.

The clean isolation of RULE transfer (Test B was confounded by dictionary domain-
shift; here the same rabbit dictionary reconstructs both halves equally well, so any
transfer failure is the SELECTION RULE's, not the dictionary's).

Design:
  1. Compute per-patch k* (true support) on rabbit patches (peak-R2 rule).
  2. Split patches into LOW-complexity (k* below median) and HIGH-complexity (above).
     These are genuinely different sub-distributions of concept count, same dictionary.
  3. CALIBRATE each rule's knob on the LOW half to hit its own mean-L0 there.
  4. FREEZE the knob, apply to the HIGH half. Measure:
     - does mean-selected track the HIGH half's higher k*? (adaptivity survives transfer)
     - realized FDR for BH (does alpha hold its meaning across the complexity gap?)
     - selection error, corr — vs fixed-k (which CANNOT adapt: frozen count = wrong on high)
  5. Also do the reverse (calibrate HIGH -> test LOW) for symmetry.

This is fixed-k's kill zone IF the thesis holds: a count tuned on simple patches is
too small for complex ones; an alpha-referenced rule should auto-widen.
"""
import pathlib, sys, json
import numpy as np, torch
from scipy import stats

RES=pathlib.Path(__file__).resolve().parents[1] / "results"
ROOT=pathlib.Path(__file__).resolve().parents[2]   # repo root
sys.path.insert(0,str(ROOT)); import bsf
DF=3

x_np=np.load(RES/"acts.npy"); idx=np.random.RandomState(0).permutation(len(x_np))[:12000]
xs=torch.as_tensor(x_np[idx],dtype=torch.float32)
m=bsf.VanillaBSF(d=768,n_groups=256,group_size=3,l0=8)
m.load_state_dict(torch.load(RES/"models_e2"/"topk_k8.pt",map_location="cpu")); m.eval()
with torch.no_grad():
    a=(xs@m.W_enc+m.b_enc).reshape(-1,256,3); gn=a.norm(dim=-1).numpy()
    Wb=m.W_dec.reshape(256,3,-1); contrib=torch.einsum('ngk,gkd->ngd',a,Wb)
    order=torch.argsort(-a.norm(dim=-1),dim=1)
    cs=torch.gather(contrib,1,order.unsqueeze(-1).expand(-1,-1,768)).cumsum(1).numpy()
N,G=gn.shape
ks_grid=np.array(list(range(1,41))+[45,50,60,80,100,140,200,256])
xn=xs.numpy(); sst=((xn-xn.mean(1,keepdims=True))**2).sum(1)
r2c=np.stack([1-((xn-cs[:,k-1])**2).sum(1)/np.maximum(sst,1e-9) for k in ks_grid],1)
peak=r2c.max(1); tgt=0.95*peak
kstar=np.array([ks_grid[np.argmax(r2c[i]>=tgt[i])] for i in range(N)],float)

med_k=np.median(kstar)
low=kstar<=med_k; high=kstar>med_k
print(f"split: LOW k* (n={low.sum()}, mean k*={kstar[low].mean():.2f}) | HIGH k* (n={high.sum()}, mean k*={kstar[high].mean():.2f})")

def sig(row,q=0.4): return max(np.quantile(row,q)/stats.chi.ppf(q,DF),1e-6)
def pvals(row): return stats.chi.sf(row/sig(row),DF)
def bh(p,alpha):
    mm=len(p);o=np.argsort(p);ps=p[o];th=(np.arange(1,mm+1)/mm)*alpha
    b=np.where(ps<=th)[0]
    if len(b)==0: return np.zeros(mm,bool)
    s=np.zeros(mm,bool); s[o[:b.max()+1]]=True; return s

def bh_count(mask,alpha): return np.array([bh(pvals(gn[i]),alpha).sum() for i in np.where(mask)[0]])
def chi_count(mask,q):
    sub=gn[mask]; med=np.median(sub,1,keepdims=True); return (sub>=q*med).sum(1)
def fixed_count(mask,k): return np.full(mask.sum(),k)

def calibrate(mask, kind):
    target=kstar[mask].mean()
    if kind=="fixed": return int(round(target))
    lo,hi=(1e-4,0.999) if kind=="bh" else (0.5,8.0)
    for _ in range(24):
        mid=(lo+hi)/2
        c=(bh_count(mask,mid) if kind=="bh" else chi_count(mask,mid)).mean()
        if kind=="bh": # higher alpha -> more
            if c<target: lo=mid
            else: hi=mid
        else: # higher q -> fewer
            if c>target: lo=mid
            else: hi=mid
    return (lo+hi)/2

def realized_fdr(mask,alpha):
    fps=tots=0
    for i in np.where(mask)[0]:
        p=pvals(gn[i]); sel=bh(p,alpha)
        if sel.sum()==0: continue
        fps+=(p[sel]>0.5).sum(); tots+=sel.sum()
    return fps/max(tots,1)

def evaluate(test_mask, counts, name):
    ks=kstar[test_mask]; nsel=counts.astype(float)
    over=np.maximum(0,nsel-ks); tr=np.maximum(0,ks-nsel)
    corr=float(np.corrcoef(nsel,ks)[0,1]) if nsel.std()>0 else 0.0
    print(f"  {name:16s} sel={nsel.mean():5.2f} (test k*={ks.mean():.2f}) corr={corr:+.3f} err={(over+tr).mean():.2f}")
    return dict(mean_selected=round(float(nsel.mean()),2),test_kstar=round(float(ks.mean()),2),
                corr=round(corr,3),total_error=round(float((over+tr).mean()),2))

out={"low_kstar":round(float(kstar[low].mean()),2),"high_kstar":round(float(kstar[high].mean()),2),"transfers":{}}

for calib,test,tag in [(low,high,"CALIB_LOW->TEST_HIGH"),(high,low,"CALIB_HIGH->TEST_LOW")]:
    print(f"\n=== {tag} ===")
    kf=calibrate(calib,"fixed"); ab=calibrate(calib,"bh"); qc=calibrate(calib,"chi")
    print(f"  calibrated on {'LOW' if calib is low else 'HIGH'}: fixed_k={kf}, BH_alpha={ab:.4f}, chi_q={qc:.3f}")
    r={}
    r["fixed_k"]=evaluate(test, fixed_count(test,kf), f"fixed_k={kf}")
    r["BH_chi"]=evaluate(test, bh_count(test,ab), f"BH a={ab:.3f}")
    r["chi_floor"]=evaluate(test, chi_count(test,qc), f"chi q={qc:.2f}")
    # realized FDR: calibrated on calib, measured on test — does alpha hold?
    r["realized_fdr_calib"]=round(float(realized_fdr(calib,ab)),3)
    r["realized_fdr_test"]=round(float(realized_fdr(test,ab)),3)
    print(f"  realized-FDR proxy: calib={r['realized_fdr_calib']} test={r['realized_fdr_test']} (alpha={ab:.3f}) — does it hold across the gap?")
    out["transfers"][tag]=dict(calib_fixed_k=kf,calib_bh_alpha=round(float(ab),4),calib_chi_q=round(float(qc),3),**r)

json.dump(out,open(RES/"testB2.json","w"),indent=2)
print("\n"+json.dumps(out,indent=2))
