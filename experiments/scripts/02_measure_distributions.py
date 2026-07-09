"""Step 2: measure & plot the "default shape" of block-norm distributions.

For each trained substrate, encode all patches, take per-block L2 norms (N, 256),
and characterise the shape that decides whether distribution-aware selection can
help (theory file §6):

  Panel A  Global histogram of ALL block norms (log-y). Bimodal (noise lump +
           signal cluster) => a real boundary exists. Unimodal smooth decay =>
           no natural boundary; min-z ~ static top-k (min-p uniform-equiv thm).

  Panel B  ACTIVE vs INACTIVE split. "Active" = the blocks a top-k=8 selector
           would keep per patch (top-8 by norm); "inactive" = the rest. Overlaid
           histograms show separation (or lack of it) between the two populations.

  Panel C  chi-fit on INACTIVE norms. Inactive block = norm of a group_size(=3)-dim
           near-Gaussian => norm ~ chi_3 * scale. Fit chi_3 to the inactive pool,
           overlay. Good fit => chi-floor selector has a real false-positive-rate
           interpretation. (tests chi-floor viability, weaker than bimodality.)

  Panel D  Per-patch sorted block-norm profile (mean +/- IQR over patches),
           x=rank. A knee near rank-8 => structure min-z can exploit; smooth
           convex decay => no knee. (tests min-z viability directly.)

Also dumps numeric shape stats to results/shape_stats.json.
"""
import pathlib, sys, json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1] / "block-sparse-featurizer"
RES = pathlib.Path(__file__).resolve().parents[1] / "results"
sys.path.insert(0, str(ROOT))
import bsf

device = "cpu"   # measurement only, keep it deterministic/simple
x = np.load(RES / "acts.npy")
summary = json.load(open(RES / "models" / "summary.json"))
GS = 3  # group_size

def load(name):
    c = summary[name]
    model = getattr(bsf, c["cls"])(d=x.shape[1], **c["kw"]).to(device)
    model.load_state_dict(torch.load(RES / "models" / f"{name}.pt", map_location=device))
    model.eval()
    return model

@torch.no_grad()
def block_norms(model, xb):
    z = model.encode(torch.as_tensor(xb, dtype=torch.float32, device=device))
    return z.norm(dim=-1).cpu().numpy()   # (N, 256)

names = list(summary.keys())
stats_out = {}
fig, axes = plt.subplots(len(names), 4, figsize=(22, 5 * len(names)))
if len(names) == 1:
    axes = axes[None, :]

# For the shape question we want the RAW pre-gate norm geometry, not the
# post-gate (which is zeroed by construction for inactive blocks). So we read
# norms from the *pre-activation* where possible; for top-k/grassmannian the
# encode() zeroes non-selected blocks, so we monkey-read the pre-gate norms.
@torch.no_grad()
def pregate_norms(model, xb):
    t = torch.as_tensor(xb, dtype=torch.float32, device=device)
    # replicate each encoder's pre-gate block code
    if isinstance(model, bsf.GrassmannianBSF):
        atoms = model.decoder_atoms()
        a = torch.exp(model.log_gamma) * (t @ atoms.t())
    elif isinstance(model, bsf.GroupLassoBSF):
        a = model.preact(t).reshape(t.shape[0], -1)
    else:  # VanillaBSF
        a = (t @ model.W_enc + model.b_enc)
    a = a.reshape(t.shape[0], model.n_groups, model.group_size)
    return a.norm(dim=-1).cpu().numpy()

for i, name in enumerate(names):
    model = load(name)
    idx = np.random.RandomState(0).permutation(len(x))[:20000]
    gn = pregate_norms(model, x[idx])          # (N,256) pre-gate
    N, G = gn.shape
    flat = gn.ravel()

    # top-8 per patch => "active" mask
    k = 8
    order = np.argsort(-gn, axis=1)
    active_mask = np.zeros_like(gn, dtype=bool)
    rows = np.arange(N)[:, None]
    active_mask[rows, order[:, :k]] = True
    active = gn[active_mask]
    inactive = gn[~active_mask]

    # ---- Panel A: global histogram, log-y
    ax = axes[i, 0]
    ax.hist(flat, bins=200, color="steelblue", log=True)
    ax.set_title(f"{name}\nA. all block norms (log-y)")
    ax.set_xlabel("block L2 norm"); ax.set_ylabel("count")

    # ---- Panel B: active vs inactive
    ax = axes[i, 1]
    lo, hi = np.percentile(flat, [0, 99.5])
    bins = np.linspace(lo, hi, 120)
    ax.hist(inactive, bins=bins, alpha=0.6, label=f"inactive (rank>{k})", color="gray", density=True)
    ax.hist(active, bins=bins, alpha=0.6, label=f"active (top-{k})", color="crimson", density=True)
    ax.set_title("B. active vs inactive (per-patch top-8)")
    ax.set_xlabel("block L2 norm"); ax.legend()

    # ---- Panel C: chi_3 fit on inactive
    ax = axes[i, 2]
    ax.hist(inactive, bins=120, density=True, alpha=0.6, color="gray", label="inactive")
    # fit chi with df fixed at group_size
    c_df, c_loc, c_scale = stats.chi.fit(inactive, f0=GS)
    xs = np.linspace(inactive.min(), inactive.max(), 400)
    ax.plot(xs, stats.chi.pdf(xs, c_df, c_loc, c_scale), "r-", lw=2,
            label=f"chi(df={GS}) fit")
    # KS test
    ks_D, ks_p = stats.kstest(inactive, "chi", args=(c_df, c_loc, c_scale))
    ax.set_title(f"C. inactive vs chi_{GS}  (KS D={ks_D:.3f})")
    ax.set_xlabel("block L2 norm"); ax.legend()

    # ---- Panel D: sorted per-patch profile
    ax = axes[i, 3]
    srt = np.sort(gn, axis=1)[:, ::-1]           # descending
    med = np.median(srt, axis=0)
    q25 = np.percentile(srt, 25, axis=0)
    q75 = np.percentile(srt, 75, axis=0)
    ranks = np.arange(1, G + 1)
    ax.plot(ranks, med, "b-", lw=1.5, label="median")
    ax.fill_between(ranks, q25, q75, alpha=0.25, color="blue", label="IQR")
    ax.axvline(k, color="k", ls="--", lw=1, label=f"top-k={k}")
    ax.set_xlim(0, 40)
    ax.set_title("D. sorted block-norm profile (zoom rank<=40)")
    ax.set_xlabel("rank"); ax.set_ylabel("block norm"); ax.legend()

    # numeric shape stats
    stats_out[name] = dict(
        R2=summary[name]["R2"], L0=summary[name]["L0"], dead=summary[name]["dead"],
        active_mean=float(active.mean()), inactive_mean=float(inactive.mean()),
        active_inactive_ratio=float(active.mean() / max(inactive.mean(), 1e-9)),
        frac_inactive_below_1pct_active=float((inactive < 0.01 * active.mean()).mean()),
        chi_fit_df=float(c_df), chi_fit_scale=float(c_scale), chi_KS_D=float(ks_D), chi_KS_p=float(ks_p),
        # bimodality: dip between the two pops. crude separation index
        separation=float((active.mean() - inactive.mean()) / (active.std() + inactive.std() + 1e-9)),
        median_norm_rank8=float(med[7]), median_norm_rank9=float(med[8]),
        knee_ratio_r8_r9=float(med[7] / max(med[8], 1e-9)),
    )

plt.tight_layout()
plt.savefig(RES / "block_norm_distributions.png", dpi=110)
json.dump(stats_out, open(RES / "shape_stats.json", "w"), indent=2)
print("saved results/block_norm_distributions.png")
print(json.dumps(stats_out, indent=2))
