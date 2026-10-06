"""Stage 3: norm-preserving biprojected abliteration on residual-stream writers."""
import argparse, json, math, time
import torch
import torch.nn as nn
from safetensors.torch import load_file, save_file

MODEL = "/models/MiMo-V2.6-Distill-Qwen-9B"

FULL_ATTN = {3, 7, 11, 15, 19, 23, 27, 31}


def orthonormalize(vecs):
    """QR on [k, d] -> [k, d] orthonormal basis spanning the same subspace."""
    if vecs.shape[0] == 1:
        return torch.nn.functional.normalize(vecs, dim=-1)
    Q, R = torch.linalg.qr(vecs.T)          # [d, k]
    s = torch.sign(torch.diag(R))
    s[s == 0] = 1.0
    return torch.nn.functional.normalize(Q * s, dim=0).T


def mp_ablate(W, R, alpha):
    """Norm-preserving orthogonal ablation, grimjim MPOA.

    W: [out, in], residual axis = dim 0. R: [k, out] orthonormal.
    Removes R's span from each row's directional component while keeping
    every row's original magnitude.
    """
    W = W.float()
    row_norm = torch.linalg.vector_norm(W, dim=1, keepdim=True)      # [out, 1]
    W_dir = W / row_norm.clamp_min(1e-12)
    # remove span(R) from the residual (output) axis: W - a * R^T R W
    W_dir = W_dir - alpha * (R.T @ R @ W_dir)
    W_dir = W_dir / torch.linalg.vector_norm(W_dir, dim=1, keepdim=True).clamp_min(1e-12)
    return (row_norm * W_dir).contiguous()


def mp_ablate_rows(E, R, alpha):
    """Same, but residual axis = dim 1 (token embeddings, [vocab, d])."""
    E = E.float()
    row_norm = torch.linalg.vector_norm(E, dim=1, keepdim=True)
    E_dir = E / row_norm.clamp_min(1e-12)
    E_dir = E_dir - alpha * ((E_dir @ R.T) @ R)
    E_dir = E_dir / torch.linalg.vector_norm(E_dir, dim=1, keepdim=True).clamp_min(1e-12)
    return (row_norm * E_dir).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", default="/out/dirs.pt")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--band", required=True, help="lo:hi inclusive, e.g. 16:28")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--k", type=int, default=1, help="top-k SVD directions per src layer")
    ap.add_argument("--alpha-attn", type=float, default=1.0)
    ap.add_argument("--alpha-mlp", type=float, default=0.7)
    ap.add_argument("--alpha-emb", type=float, default=1.0)
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--no-biproj", action="store_true")
    a = ap.parse_args()

    lo, hi = [int(x) for x in a.band.split(":")]
    band = list(range(lo, hi + 1))
    D = torch.load(a.dirs, weights_only=True)
    r_all, mu_g, quality = D["r"], D["mu_g"], D["quality"]
    svd_dirs = D["svd_dirs"]                       # [33, k, 4096]
    n_states = r_all.shape[0]

    # hidden_states[l] = resid_pre of layer l (l=0 is embedding output)
    # writers of layer L read resid_pre(L) => source measurement index = L
    chunk_id = lambda L: min(a.chunks - 1, (L - lo) * a.chunks // max(1, len(band)))
    chunks = sorted({chunk_id(L) for L in band})
    src_of = {}
    for c in chunks:
        members = [L for L in band if chunk_id(L) == c]
        src = max(members, key=lambda L: quality[L])
        for L in members:
            src_of[L] = src
    print("band %d..%d (%d layers), chunks=%d" % (lo, hi, len(band), a.chunks))
    print("src layer map:", {c: [L for L in band if chunk_id(L) == c][0]
                             for c in chunks},
          "->", {c: src_of[[L for L in band if chunk_id(L) == c][0]] for c in chunks})

    def direction_for(L):
        src = src_of[L]
        if a.k == 1:
            V = r_all[src:src + 1]                # diff-of-means at src
        else:
            V = svd_dirs[src, :a.k].clone()
        if not a.no_biproj:
            # biprojection: drop the harmless-mean component at the TARGET layer,
            # so the edit does not disturb that layer's harmless direction
            g = torch.nn.functional.normalize(mu_g[L], dim=-1)
            V = V - torch.outer(V @ g, g)
        return orthonormalize(torch.nn.functional.normalize(V, dim=-1))

    dirs = {L: direction_for(L) for L in band}
    print("k=%d  biproj=%s  alpha attn=%.2f mlp=%.2f emb=%.2f"
          % (a.k, not a.no_biproj, a.alpha_attn, a.alpha_mlp, a.alpha_emb))

    # enumerate all shard files
    import os
    idx = json.load(open(a.model + "/model.safetensors.index.json"))["weight_map"]
    shards = sorted(set(idx.values()))
    all_keys = sorted(idx)
    print("%d tensors in %d shards" % (len(all_keys), len(shards)), flush=True)

    t0 = time.time()
    modified = []

    def handle(key, tensor):
        # layer writers
        for L in band:
            p_mlp = f"model.language_model.layers.{L}.mlp.down_proj.weight"
            p_out = f"model.language_model.layers.{L}.linear_attn.out_proj.weight"
            p_o = f"model.language_model.layers.{L}.self_attn.o_proj.weight"
            if key == p_mlp:
                modified.append((key, "mlp", L))
                return mp_ablate(tensor, dirs[L], a.alpha_mlp).to(tensor.dtype)
            if key in (p_out, p_o):
                modified.append((key, "attn", L))
                return mp_ablate(tensor, dirs[L], a.alpha_attn).to(tensor.dtype)
        if key == "model.language_model.embed_tokens.weight" and not a.no_embed:
            # source for embeddings: best-quality layer in band
            src = max(band, key=lambda L: quality[L])
            V = dirs[src]
            modified.append((key, "emb", src))
            return mp_ablate_rows(tensor, V, a.alpha_emb).to(tensor.dtype)
        return tensor

    for shard in shards:
        sd = load_file(a.model + "/" + shard)
        for k in list(sd.keys()):
            sd[k] = handle(k, sd[k])
        save_file(sd, a.out + "/" + shard)
        del sd
        print("  shard %s done  %.0fs  (%d modified so far)"
              % (shard, time.time() - t0, len(modified)), flush=True)

    # copy the rest of the model dir (tokenizer, config, index)
    import shutil
    for f in os.listdir(a.model):
        if f.endswith(".safetensors"):
            continue
        src = a.model + "/" + f
        if os.path.isdir(src):
            shutil.copytree(src, a.out + "/" + f, dirs_exist_ok=True)
        else:
            shutil.copy2(src, a.out + "/" + f)
    assert os.path.exists(a.out + "/model.safetensors.index.json"), "index not copied"

    json.dump({"band": [lo, hi], "chunks": a.chunks, "k": a.k,
               "alpha_attn": a.alpha_attn, "alpha_mlp": a.alpha_mlp,
               "alpha_emb": a.alpha_emb, "embed": not a.no_embed,
               "biproj": not a.no_biproj,
               "src_map": {str(k): int(v) for k, v in src_of.items()},
               "n_modified": len(modified),
               "modified": modified},
              open(a.out + "/ABLIT.json", "w"), indent=1)
    print("modified %d tensors -> %s  (%.0fs)"
          % (len(modified), a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
