"""Step 0a (Path B / timm): one-time DINOv3 activation pass over 300 rabbit images.

WHY timm instead of the repo's HF AutoModel path:
  facebook/dinov3-vitb16-pretrain-lvd1689m is a GATED HF repo; access request is
  pending Meta author review (hours-to-days). timm/vit_base_patch16_dinov3.lvd1689m
  is the IDENTICAL Meta checkpoint (hash 73cec8be), non-gated. For measuring the
  *qualitative distribution shape* of block norms this is sound — convention
  differences (final-norm, register tokens, exact resize) cannot flip a
  distribution unimodal<->bimodal. Re-run on the facebook repo later for
  publication-grade bit-for-bit numbers if Meta approves.

CONSEQUENCE: the repo ships bsf/pos_mean.npy computed on the HF pipeline. Under
timm the activation convention differs slightly, so that array is NOT valid here.
We therefore RECOMPUTE the positional mean from THESE activations (mean over
images, per patch position) so the positional subtraction is internally
consistent with the pipeline that produced the activations. Saved as
results/pos_mean_timm.npy for provenance.

Preprocessing mirrors the notebook spirit:
  acts = dino patch tokens (drop 5 prefix)          # (300, 196, 768)
  acts = acts - POS_MEAN_timm                        # remove positional main effect
  x = rearrange 'n p d -> (n p) d'                    # (58800, 768)
  x = x / sqrt(mean ||x||^2) * sqrt(d)                # scale so ||x|| ~ sqrt(d)

Caches x -> results/acts.npy. Run once from the experiment root with venv active.
"""
import pathlib, sys
import numpy as np
import torch
import einops
import timm

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
OUT = pathlib.Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))
from bsf import data as bsf_data  # only for load_rabbit_images + patch_grid

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device={device}")

MODEL_ID = "vit_base_patch16_dinov3.lvd1689m"
N_PREFIX = 5           # CLS + 4 register tokens (matches repo N_REGISTER_TOKENS)
IMG = 224              # force 224 -> 14x14 = 196 patch tokens (repo convention)

images = bsf_data.load_rabbit_images(ROOT / "rabbit.npz")   # (300, 224, 224, 3) uint8
print(f"images: {images.shape} {images.dtype}")

model = timm.create_model(MODEL_ID, pretrained=True, num_classes=0, img_size=IMG).to(device).eval()
cfg = timm.data.resolve_data_config({}, model=model)
mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1)
std = torch.tensor(cfg["std"]).view(1, 3, 1, 1)
print(f"model={MODEL_ID}  img={IMG}  num_prefix={model.num_prefix_tokens}  d={model.embed_dim}")

@torch.no_grad()
def acts_for(imgs, bs=64):
    out = []
    for s in range(0, len(imgs), bs):
        chunk = imgs[s:s + bs].astype(np.float32) / 255.0            # (b,H,W,3)
        t = torch.from_numpy(chunk).permute(0, 3, 1, 2)              # (b,3,H,W)
        t = (t - mean) / std
        t = t.to(device)
        f = model.forward_features(t)                                # (b, T, d)
        out.append(f[:, N_PREFIX:].float().cpu().numpy())            # drop prefix -> (b,196,d)
    return np.concatenate(out, 0)

acts = acts_for(images)                     # (300, 196, 768)
print(f"raw patch acts: {acts.shape}")
assert acts.shape[1] == 196, f"expected 196 patch tokens, got {acts.shape[1]}"

# recompute positional mean from THESE activations (per-patch-position mean over images)
pos_mean = acts.mean(axis=0)                # (196, 768)
np.save(OUT / "pos_mean_timm.npy", pos_mean.astype(np.float32))

acts = acts - pos_mean[None]
x = einops.rearrange(acts, "n p d -> (n p) d")
x = x / np.sqrt((x ** 2).sum(1).mean()) * np.sqrt(x.shape[1])
grid = bsf_data.patch_grid(acts.shape[1])

x = np.asarray(x, dtype=np.float32)
np.save(OUT / "acts.npy", x)
np.save(OUT / "grid.npy", np.array(grid))
print(f"saved acts.npy: {x.shape}  patch_grid={grid}")
print(f"mean ||x|| = {np.sqrt((x**2).sum(1)).mean():.3f}  (target ~ sqrt(d)={np.sqrt(x.shape[1]):.3f})")
print("saved pos_mean_timm.npy (recomputed, internally consistent)")
