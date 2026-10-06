"""Stage 1b: refusal direction from the RESPONSE region (reasoning-model aware).

The original extract.py pooled the single last PROMPT token. On a reasoning
model the refuse/comply decision is made during the <think> stream, so that
position carries a topic-classifier direction that is nearly orthogonal to the
weight outputs (||R W||/||W|| ~ 1.4%) -> ablating it does nothing.

This generates a short greedy completion per prompt and mean-pools the
residual-stream hidden states over the GENERATED tokens, where refusal is
actually expressed. Output format matches acts.pt so score.py/ablate.py are
unchanged.
"""
import argparse, gc, json, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "/models/MiMo-V2.6-Distill-Qwen-9B"

# i7-9700 has no AVX-512 BF16: upcast each Linear matmul to fp32 (~1.7x faster).
def _upcast(self, x):
    if self.weight.dtype == torch.bfloat16 and x.dtype != torch.bfloat16:
        return F.linear(x, self.weight, self.bias)
    if self.weight.dtype == torch.bfloat16:
        y = F.linear(x.float(), self.weight.float(),
                     self.bias.float() if self.bias is not None else None)
        return y.to(x.dtype)
    return F.linear(x, self.weight, self.bias)
nn.Linear.forward = _upcast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harmful", default="/data/harmful.json")
    ap.add_argument("--harmless", default="/data/harmless.json")
    ap.add_argument("--n", type=int, default=96, help="prompts per side")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--resp", type=int, default=20, help="response tokens to pool")
    ap.add_argument("--out", default="/out/acts2.pt")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"
    tok.truncation_side = "left"
    tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id

    harmful = json.load(open(a.harmful))[:a.n]
    harmless = json.load(open(a.harmless))[:a.n]

    def render(ts):
        return [tok.apply_chat_template([{"role": "user", "content": t}],
                                        tokenize=False, add_generation_prompt=True)
                for t in ts]

    print("loading model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model.eval()
    print("loaded", flush=True)

    def side(insts, tag):
        texts = render(insts)
        chunks = []
        t0 = time.time()
        for i in range(0, len(texts), a.batch):
            batch = texts[i:i + a.batch]
            enc = tok(batch, return_tensors="pt", padding=True,
                      truncation=True, max_length=a.max_len)
            P = enc["input_ids"].shape[1]
            with torch.no_grad():
                seqs = model.generate(**enc, max_new_tokens=a.resp,
                                      do_sample=False, use_cache=True,
                                      pad_token_id=pad_id)
                attn = (seqs != pad_id).long()
                out = model(input_ids=seqs, attention_mask=attn,
                            output_hidden_states=True, use_cache=False)
            # pool over the response columns [P, P+resp), excluding pad (post-EOS)
            resp = torch.zeros_like(seqs, dtype=torch.bool)
            resp[:, P:] = seqs[:, P:] != pad_id
            cnt = resp.sum(1).clamp_min(1).unsqueeze(-1).float()      # [b,1]
            m = resp.unsqueeze(-1).float()                            # [b,seq,1]
            per_layer = [((h.float() * m).sum(1) / cnt) for h in out.hidden_states]
            hs = torch.stack(per_layer, dim=1)                        # [b, L+1, d]
            chunks.append(hs.cpu())
            del out, seqs, hs
            gc.collect()
            done = min(i + a.batch, len(texts))
            el = time.time() - t0
            print("%s %d/%d  %.0fs  eta %.0fs" % (tag, done, len(texts), el,
                  el / done * (len(texts) - done)), flush=True)
        return torch.cat(chunks, 0)

    acts = {"harmful": side(harmful, "harmful"),
            "harmless": side(harmless, "harmless"),
            "meta": {"n": a.n, "resp": a.resp, "n_layers": 33, "hidden": 4096}}
    torch.save(acts, a.out)
    print("saved %s  harmful%s harmless%s" % (a.out,
          tuple(acts["harmful"].shape), tuple(acts["harmless"].shape)), flush=True)


if __name__ == "__main__":
    main()
