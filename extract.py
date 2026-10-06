"""Stage 1: cache last-token residual activations for harmful vs harmless prompts."""
import argparse, gc, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/MiMo-V2.6-Distill-Qwen-9B"

# This i7 has no AVX-512 BF16: bf16 matmuls are ~1.7x slower than fp32.
# Keep stored weights bf16 (18 GB) but upcast each Linear's matmul to fp32.
_orig_linear_forward = nn.Linear.forward


def _upcast_forward(self, input):
    if self.weight.dtype == torch.bfloat16 and input.dtype != torch.bfloat16:
        return F.linear(input, self.weight, self.bias)
    if self.weight.dtype == torch.bfloat16:
        y = F.linear(input.float(), self.weight.float(),
                     self.bias.float() if self.bias is not None else None)
        return y.to(input.dtype)
    return F.linear(input, self.weight, self.bias)


nn.Linear.forward = _upcast_forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful", default="/data/harmful.json")
    ap.add_argument("--harmless", default="/data/harmless.json")
    ap.add_argument("--n", type=int, default=200, help="prompts per side")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", default="/out/acts.pt")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    tok.pad_token = tok.eos_token

    harmful = json.load(open(args.harmful))[:args.n]
    harmless = json.load(open(args.harmless))[:args.n]

    def render(insts):
        return [tok.apply_chat_template([{"role": "user", "content": t}],
                                        tokenize=False, add_generation_prompt=True)
                for t in insts]

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    print("loaded", flush=True)

    def cache_side(insts, tag):
        texts = render(insts)
        # shuffle-free deterministic order; caller holds out an eval split
        chunks = []
        t0 = time.time()
        for i in range(0, len(texts), args.batch):
            batch = texts[i:i + args.batch]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=args.max_len)
            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)
            # left padding => last position is the final real token for every row
            hs = torch.stack([h[:, -1, :].float() for h in out.hidden_states], dim=1)
            chunks.append(hs.cpu())
            del out, hs
            if i % (args.batch * 2) == 0:
                gc.collect()
            done = min(i + args.batch, len(texts))
            el = time.time() - t0
            print("%s %d/%d  %.0fs  eta %.0fs" % (tag, done, len(texts), el,
                  el / done * (len(texts) - done)), flush=True)
        return torch.cat(chunks, 0)

    acts = {
        "harmful": cache_side(harmful, "harmful"),
        "harmless": cache_side(harmless, "harmless"),
        "meta": {"n": args.n, "max_len": args.max_len,
                 "n_layers": 33, "hidden": 4096},
    }
    torch.save(acts, args.out)
    print("saved %s  harmful%s harmless%s" %
          (args.out, tuple(acts["harmful"].shape), tuple(acts["harmless"].shape)),
          flush=True)


if __name__ == "__main__":
    main()
