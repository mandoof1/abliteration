import time, gc, torch, torch.nn.functional as F

MODEL = "/models/MiMo-V2.6-Distill-Qwen-9B"
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
tok.pad_token = tok.eos_token

texts = [
    "Write a function that reverses a linked list in Python with tests.",
    "Explain the difference between TCP and UDP.",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I configure nginx as a reverse proxy?",
] * 8
enc = tok(texts, return_tensors="pt", padding=True)
n = enc["input_ids"].numel()
print("batch", tuple(enc["input_ids"].shape), "tokens", n, flush=True)

m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
m.eval()


def fwd(tag):
    gc.collect()
    with torch.no_grad():
        t0 = time.time()
        m(**enc, use_cache=False)
        dt = time.time() - t0
    print("%s: %.1fs -> %.2f tok/s" % (tag, dt, n / dt), flush=True)
    return dt


fwd("bf16 warmup")
b = fwd("bf16")

# ---- upcast path: bf16 weights, fp32 matmul, cast back ----
orig = F.linear
stats = {"n": 0}


def upcast(x, weight, bias=None):
    if weight is not None and weight.dtype == torch.bfloat16:
        stats["n"] += 1
        y = orig(x.float(), weight.float(), bias.float() if bias is not None else None)
        return y.to(x.dtype)
    return orig(x, weight, bias)


F.linear = upcast
try:
    fwd("upcast warmup")
    u = fwd("upcast")
finally:
    F.linear = orig
print("upcast patched linear calls:", stats["n"], flush=True)
print("SPEEDUP bf16/upcast = %.2fx" % (b / u), flush=True)

# ---- generation speed with cache ----
gen_texts = [
    "Write a function that reverses a linked list in Python with tests.",
    "Explain the difference between TCP and UDP.",
    "Summarize the plot of Hamlet in three sentences.",
    "How do I configure nginx as a reverse proxy?",
] * 4
encg = tok(gen_texts, return_tensors="pt", padding=True)
gc.collect()
with torch.no_grad():
    t0 = time.time()
    m.generate(**encg, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
toks = 32 * len(gen_texts)
print("generate(bf16, cache): %.1fs -> %.2f tok/s" % (dt, toks / dt), flush=True)

F.linear = upcast
try:
    gc.collect()
    with torch.no_grad():
        t0 = time.time()
        m.generate(**encg, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        dt = time.time() - t0
    print("generate(upcast, cache): %.1fs -> %.2f tok/s" % (dt, toks / dt), flush=True)
finally:
    F.linear = orig

print("fla available:", end=" ")
try:
    import fla  # noqa
    print("yes")
except Exception as e:
    print("no", e)
