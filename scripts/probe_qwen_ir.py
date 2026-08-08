import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen.uaii.json"
g = json.load(open(path, encoding="utf-8"))
tensors = g.get("tensors") or g.get("graph", {}).get("tensors") or []
print("tensor_count", len(tensors))
for t in tensors:
    name = t.get("name", "")
    if any(k in name for k in ("token_embd", "emb", "output", "lm_head")):
        print(json.dumps({k: t.get(k) for k in ("name", "dtype", "shape", "quant_format", "weight_ref") if k in t or True}, ensure_ascii=False))
# also print metadata
meta = g.get("metadata") or g.get("graph", {}).get("metadata") or {}
print("meta", {k: meta.get(k) for k in list(meta)[:20]})
