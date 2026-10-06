"""Integrity + completeness check for the produced model D (bf16 HF dir)."""
import json, os
from safetensors import safe_open
import torch

root = "/work/out/D"
idx = json.load(open(root + "/model.safetensors.index.json"))["weight_map"]
shards = sorted(set(idx.values()))
abl = json.load(open(root + "/ABLIT.json"))
modkeys = {m[0] for m in abl["modified"]}

# 1) every index key is actually present in some shard
present = set()
for sh in shards:
    with safe_open(root + "/" + sh, framework="pt") as f:
        present.update(f.keys())
missing = [k for k in idx if k not in present]

# 2) NaN/Inf check on the ablated tensors (the only ones that changed)
bad = []
for sh in shards:
    with safe_open(root + "/" + sh, framework="pt") as f:
        keys = set(f.keys())
        for k in modkeys & keys:
            t = f.get_tensor(k).float()
            if torch.isnan(t).any() or torch.isinf(t).any():
                bad.append(k)

# 3) required non-weight files for a loadable / fine-tunable model dir
need = ["config.json", "model.safetensors.index.json", "tokenizer.json",
        "tokenizer_config.json", "generation_config.json", "chat_template.jinja"]
missfiles = [n for n in need if not os.path.exists(root + "/" + n)]

print("shards present      :", len(shards))
print("index keys          :", len(idx), "| all present:", not missing)
print("modified tensors     :", len(modkeys), "| NaN/Inf found:", len(bad))
print("missing index keys   :", missing[:5])
print("missing aux files    :", missfiles)
print("VERDICT:", "OK - complete and clean"
      if not missing and not bad and not missfiles else "PROBLEM (see above)")
