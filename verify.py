"""Verify mp_ablate actually removed the target direction, and how well
row norms were preserved (the renormalization step trades exactness for
norm preservation - quantify both)."""
import argparse, json
import torch
from safetensors import safe_open

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", default="/out/dirs.pt")
ap.add_argument("--model", default="/models/MiMo-V2.6-Distill-Qwen-9B")
ap.add_argument("--abl", default="/out/A", help="ablititerated dir")
ap.add_argument("--band", default="16:28")
ap.add_argument("--k", type=int, default=1)
ap.add_argument("--alpha", type=float, default=1.0, help="alpha used for this tensor")
ap.add_argument("--layer", type=int, default=26)
ap.add_argument("--which", default="mlp", choices=["mlp", "attn", "emb"])
a = ap.parse_args()

D = torch.load(a.dirs, weights_only=True)
r_all, mu_g = D["r"], D["mu_g"]
lo, hi = [int(x) for x in a.band.split(":")]
src = a.layer
if a.k == 1:
    V = r_all[src:src + 1]
else:
    V = D["svd_dirs"][src, :a.k].clone()
g = torch.nn.functional.normalize(mu_g[a.layer], dim=-1)
V = V - torch.outer(V @ g, g)
V = torch.nn.functional.normalize(V, dim=-1)
Q, Rr = torch.linalg.qr(V.T)
s = torch.sign(torch.diag(Rr)); s[s == 0] = 1.0
Rr = torch.nn.functional.normalize(Q * s, dim=0).T     # [k, 4096]
print("direction: k=%d  layer=%d  ||R R^T - I|| = %.2e"
      % (Rr.shape[0], a.layer, float((Rr @ Rr.T - torch.eye(Rr.shape[0])).abs().max())))

if a.which == "emb":
    key = "model.language_model.embed_tokens.weight"
    axis = 1
else:
    p = "mlp.down_proj" if a.which == "mlp" else "self_attn.o_proj"
    key = "model.language_model.layers.%d.%s.weight" % (a.layer, p)
    axis = 0


def grab(root, key):
    idx = json.load(open(root + "/model.safetensors.index.json"))["weight_map"]
    with safe_open(root + "/" + idx[key], framework="pt") as f:
        return f.get_tensor(key)


W0 = grab(a.model, key).float()
W1 = grab(a.abl, key).float()
print("tensor %s  %s -> %s" % (key, tuple(W0.shape), tuple(W1.shape)))

if axis == 0:
    # want  W_new^T r = 0  (r in null space => output is always bot r)
    t0 = Rr @ W0               # [k, in]  = r^T W
    t1 = Rr @ W1
    base = float(t0.norm() / W0.norm())
    abl = float(t1.norm() / W1.norm())
    print("rel ||R W||/||W|| (alpha=%.2f): base %.4f -> abl %.4f   (reduction %.1f%%)"
          % (a.alpha, base, abl, 100 * (1 - abl / base)))
    for i in range(Rr.shape[0]):
        b = float((Rr[i] @ W0).norm() / W0.norm())
        c = float((Rr[i] @ W1).norm() / W1.norm())
        print("   dir %d: %.5f -> %.5f  (-%.1f%%)" % (i, b, c, 100 * (1 - c / b)))
else:
    # embeddings: rows live in residual space; want row_proj_onto_R ~ 0
    for i in range(Rr.shape[0]):
        b = float((W0 @ Rr[i]).abs().mean())
        c = float((W1 @ Rr[i]).abs().mean())
        print("   dir %d |row.r| mean: base %.5f -> abl %.5f  (-%.1f%%)"
              % (i, b, c, 100 * (1 - c / b)))

n0 = W0.norm(dim=1)
n1 = W1.norm(dim=1)
rel = (n1 / n0.clamp_min(1e-12) - 1).abs()
print("row-norm preservation: median %.3e  p99 %.3e  max %.3e"
      % (float(rel.median()), float(rel.quantile(0.99)), float(rel.max())))
ang = torch.nn.functional.cosine_similarity(W0, W1, dim=1)
print("cos(base,abl) rows: mean %.4f  min %.4f   (max rotation %.1f deg)"
      % (float(ang.mean()), float(ang.min()),
         float((1 - ang.clamp(-1, 1)).acos().max().rad2deg())))
