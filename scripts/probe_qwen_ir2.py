import json
g=json.load(open("/tmp/qwen.uaii.json"))
ts=g.get("tensors") or g.get("graph",{}).get("tensors") or []
print("count", len(ts))
for t in ts:
    n=t.get("name","")
    if "emb" in n or n in ("output.weight","lm_head","tokens","probs","logits"):
        print(n, "dtype=", t.get("dtype"), "quant=", t.get("quant_format"), "shape=", t.get("shape"), "ref=", t.get("weight_ref"))
meta=g.get("metadata") or {}
print("heads", meta.get("n_heads"), "kv", meta.get("n_kv_heads"), "dim", meta.get("embedding_length"), "hd", meta.get("head_dim"))
