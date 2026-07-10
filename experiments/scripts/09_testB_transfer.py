"""Test B: cross-distribution transfer. Freeze rabbit-calibrated knobs, apply to
Imagenette (10 diverse ImageNet classes), measure whether the FDR alpha holds its
meaning while fixed-k's effective error swings.

Pipeline: rabbit-trained VanillaBSF dictionary (frozen) + rabbit-recomputed pos_mean
convention, applied to Imagenette DINOv3 activations. The KEY question: with knobs
FROZEN at rabbit values, does realized selection behavior transfer?

Metrics on Imagenette (with its OWN k* ground truth, computed the same peak-R2 way):
  - realized selection stats per rule at FROZEN rabbit knob
  - fixed_k=5 (frozen count) vs BH-chi alpha=0.10 (frozen FDR) vs chi-floor q (frozen)
  - does mean-selected track Imagenette's k* mean? does corr hold?
  - THE headline: fixed-k can't move its count (rigid 5 regardless of new complexity);
    alpha-BH auto-adjusts count to the new distribution's complexity while keeping the
    same FDR interpretation.
"""
import pathlib, sys, json, glob
import numpy as np
import torch
from scipy import stats
from PIL import Image

BASE = pathlib.Path("/Users/minh/local/ai-research/2026-07-09-bsf-distribution-aware-thresholds")
RES = BASE/"results"; ROOT = BASE/"block-sparse-featurizer"
sys.path.insert(0, str(ROOT)); import bsf
import timm

device = "mps" if torch.backends.mps.is_available() else "cpu"
DF=3; N_PREFIX=5; IMG=224

# ---- Imagenette activations through the SAME DINOv3 pipeline ----
val_imgs = sorted(glob.glob(str(BASE/"data/imagenette2-320/val/**/*.JPEG"), recursive=True))
rng = np.random.RandomState(0); rng.shuffle(val_imgs)
val_imgs = val_imgs[:300]   # match rabbit sample size

model = timm.create_model("vit_base_patch16_dinov3.lvd1689m", pretrained=True, num_classes=0, img_size=IMG).to(device).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"]).view(1,3,1,1); std = torch.tensor(cfg["std"]).view(1,3,1,1)

@torch.no_grad()
def dino_acts(paths, bs=32):
    out=[]
    for s in range(0,len(paths),bs):
        batch=[]
        for p in paths[s:s+bs]:
            im=Image.open(p).convert("RGB").resize((IMG,IMG))
            batch.append(np.asarray(im,dtype=np.float32)/255.0)
        t=torch.from_numpy(np.stack(batch)).permute(0,3,1,2)
        t=((t-mean)/std).to(device)
        f=model.forward_features(t)
        out.append(f[:,N_PREFIX:].float().cpu().numpy())
    return np.concatenate(out,0)

acts = dino_acts(val_imgs)                     # (300,196,768)
pos_mean = acts.mean(0)                          # imagenette's own positional mean
acts = acts - pos_mean[None]
import einops
x = einops.rearrange(acts,"n p d -> (n p) d")
x = x/np.sqrt((x**2).sum(1).mean())*np.sqrt(x.shape[1])
x = x.astype(np.float32)
np.save(RES/"acts_imagenette.npy", x)
print(f"imagenette acts: {x.shape}")

# subsample patches to 12000 to match rabbit harness
idx = np.random.RandomState(1).permutation(len(x))[:12000]
xt = torch.as_tensor(x[idx], dtype=torch.float32)

# ---- frozen rabbit dictionary ----
m = bsf.VanillaBSF(d=768,n_groups=256,group_size=3,l0=8)
m.load_state_dict(torch.load(RES/"models_e2"/"topk_k8.pt", map_location="cpu")); m.eval()
with torch.no_grad():
    a=(xt@m.W_enc+m.b_enc).reshape(-1,256,3); gn=a.norm(dim=-1).numpy()
    Wb=m.W_dec.reshape(256,3,-1); contrib=torch.einsum('ngk,gkd->ngd',a,Wb)
    order=torch.argsort(-a.norm(dim=-1),dim=1)
    cs=torch.gather(contrib,1,order.unsqueeze(-1).expand(-1,-1,768)).cumsum(1).numpy()
N,G=gn.shape

# ---- imagenette's OWN k* ----
ks_grid=np.array(list(range(1,41))+[45,50,60,80,100,140,200,256])
xn=xt.numpy(); sst=((xn-xn.mean(1,keepdims=True))**2).sum(1)
r2c=np.stack([1-((xn-cs[:,k-1])**2).sum(1)/np.maximum(sst,1e-9) for k in ks_grid],1)
peak=r2c.max(1); tgt=0.95*peak
kstar=np.array([ks_grid[np.argmax(r2c[i]>=tgt[i])] for i in range(N)],float)

# ---- rabbit-frozen knobs (from prior scripts) ----
RABBIT_K = 5           # fixed_k matched to rabbit k*=5.4
RABBIT_ALPHA = 0.10    # BH target FDR (principled default, NOT retuned)
RABBIT_Q = 2.71        # chi-floor q matched on rabbits (from script 07 tuning)

def sig(row,q=0.4): return max(np.quantile(row,q)/stats.chi.ppf(q,DF),1e-6)
def pvals(row): return stats.chi.sf(row/sig(row),DF)
def bh(p,alpha):
    mm=len(p);o=np.argsort(p);ps=p[o];th=(np.arange(1,mm+1)/mm)*alpha
    b=np.where(ps<=th)[0]
    if len(b)==0: return np.zeros(mm,bool)
    s=np.zeros(mm,bool); s[o[:b.max()+1]]=True; return s

sel_fixed = np.full(N,RABBIT_K)
sel_bh = np.array([bh(pvals(gn[i]),RABBIT_ALPHA).sum() for i in range(N)])
med=np.median(gn,axis=1,keepdims=True)
sel_chi = (gn >= RABBIT_Q*med).sum(1)

def rep(nsel,name):
    nsel=nsel.astype(float)
    over=np.maximum(0,nsel-kstar);tr=np.maximum(0,kstar-nsel)
    corr=float(np.corrcoef(nsel,kstar)[0,1]) if nsel.std()>0 else 0.0
    print(f"{name:20s} sel={nsel.mean():5.2f} (k*={kstar.mean():.2f}) corr={corr:+.3f} err={(over+tr).mean():.2f}")
    return dict(mean_selected=round(float(nsel.mean()),2),corr=round(corr,3),
                total_error=round(float((over+tr).mean()),2),
                mean_over=round(float(over.mean()),2),mean_trunc=round(float(tr.mean()),2))

print(f"\n=== TEST B: rabbit-frozen knobs on IMAGENETTE ===")
print(f"imagenette k*: mean={kstar.mean():.2f} CV={kstar.std()/kstar.mean():.3f}  (rabbit k* was 5.41/0.308)\n")
out={"imagenette_kstar_mean":round(float(kstar.mean()),2),"imagenette_kstar_CV":round(float(kstar.std()/kstar.mean()),3),
     "rabbit_kstar_mean":5.41,"frozen_knobs":{"fixed_k":RABBIT_K,"bh_alpha":RABBIT_ALPHA,"chi_q":RABBIT_Q},"results":{}}
out["results"]["fixed_k_frozen"]=rep(sel_fixed,"fixed_k=5 (frozen)")
out["results"]["BH_chi_frozen"]=rep(sel_bh,"BH alpha=0.10 (frozen)")
out["results"]["chi_floor_frozen"]=rep(sel_chi,"chi-floor q=2.71 (frozen)")

# realized FDR check for BH: fraction of selected blocks that are below-noise (est false positives)
# use the chi p-value: a "false discovery" ~ selected block whose p > alpha region under null
def realized_fdr(alpha):
    fps=tots=0
    for i in range(N):
        p=pvals(gn[i]); sel=bh(p,alpha)
        if sel.sum()==0: continue
        # est false discoveries = selected blocks with large p (consistent w/ null)
        fps += (p[sel]>0.5).sum(); tots += sel.sum()
    return fps/max(tots,1)
out["realized_fdr_proxy_rabbitalpha_on_imagenette"]=round(float(realized_fdr(RABBIT_ALPHA)),3)
print(f"\nrealized-FDR proxy (BH a=0.10) on imagenette: {out['realized_fdr_proxy_rabbitalpha_on_imagenette']}")

json.dump(out,open(RES/"testB_transfer.json","w"),indent=2)
print("\n"+json.dumps(out,indent=2))
