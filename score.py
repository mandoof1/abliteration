"""Stage 2: per-layer refusal direction, quality score, top-k SVD directions."""
import argparse, json
import torch

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/out/acts.pt")
p.add_argument("--out", default="/out/dirs.pt")
p.add_argument("--k", type=int, default=4, help="top-k SVD directions per layer")
a = p.parse_args()

d = torch.load(a.acts, weights_only=True)
H = d["harmful"].float()      # [N, L, 4096]
G = d["harmless"].float()
N, NL, D = H.shape
print("harmful%s harmless%s" % (tuple(H.shape), tuple(G.shape)))

mu_h = H.mean(0)              # [L, D]
mu_g = G.mean(0)
diff = mu_h - mu_g            # [L, D]
r = torch.nn.functional.normalize(diff, dim=-1)

snr = diff.norm(dim=-1) / torch.maximum(mu_h.norm(dim=-1), mu_g.norm(dim=-1))
cos = torch.nn.functional.cosine_similarity(mu_h, mu_g, dim=-1)
dissim = 1.0 - cos
quality = snr * dissim

# cross-layer directional stability: cos(r_l, r_l+1)
stab = torch.nn.functional.cosine_similarity(r[1:], r[:-1], dim=-1)

# matched-pair contrast matrix per layer -> SVD for multi-direction subspaces
svd_dirs, svd_var = [], []
for l in range(NL):
    # do NOT center: the mean of this contrast IS the refusal signal
    Dm = H[:, l, :] - G[:, l, :]           # [N, D]
    U, S, Vh = torch.linalg.svd(Dm, full_matrices=False)
    top = Vh[:a.k].clone()                 # [k, D]
    var = (S[:a.k] ** 2) / (S ** 2).sum().clamp_min(1e-12)
    svd_dirs.append(torch.nn.functional.normalize(top, dim=-1))
    svd_var.append(var)

svd_dirs = torch.stack(svd_dirs)           # [L, k, D]
svd_var = torch.stack(svd_var)             # [L, k]

# which layers does the doM direction agree with the top SVD component?
dom_vs_svd = torch.nn.functional.cosine_similarity(r, svd_dirs[:, 0], dim=-1)

print("\n L   snr    dissim  quality   cos(g,h)  stab    svdvar1  doM~svd1")
print("-" * 72)
for l in range(NL):
    print("%2d  %.4f  %.4f  %.5f   %.4f   %.4f   %.3f    %.3f" %
          (l, snr[l], dissim[l], quality[l], cos[l],
           stab[l - 1] if l > 0 else float("nan"),
           svd_var[l, 0], dom_vs_svd[l]))

best = quality.argmax().item()
print("\npeak quality layer: %d (quality %.5f)" % (best, quality[best]))

# recommended band: exclude first 25% of layers (early = syntax/garbling risk)
lo = max(4, int(round(0.25 * NL)))
cand = [(int(l), float(quality[l])) for l in range(lo, NL)]
cand.sort(key=lambda t: -t[1])
print("band floor = layer %d (25%% of %d); top-8 in band: %s"
      % (lo, NL, cand[:8]))

# inter-layer agreement of r across the whole stack (mean pairwise cos)
Gmat = torch.nn.functional.cosine_similarity(
    r.unsqueeze(1), r.unsqueeze(0), dim=-1)
print("mean pairwise cos(r_l, r_l') across all layers: %.4f" %
      Gmat.mean().item())
print("mean cos within band[%d:]: %.4f" %
      (lo, Gmat[lo:, lo:].mean().item()))

torch.save({"r": r, "mu_h": mu_h, "mu_g": mu_g,
            "snr": snr, "dissim": dissim, "quality": quality,
            "svd_dirs": svd_dirs, "svd_var": svd_var,
            "band_lo": lo, "peak": best}, a.out)
print("saved", a.out)
