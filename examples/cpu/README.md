# Phase 3/5 execution examples

## Built-in demos (recommended)

```bash
uaii run --demo toy_mlp
uaii run --demo tiny_block
uaii run --demo parity   # Phase 5: cpu vs cuda under parity policy
```

`toy_mlp` checks a known expected softmax output (`0.25` uniform with all-ones weights).

## Run hand-authored IR

```bash
uaii run examples/ir/toy_mlp.uaii.json \
  --weight-init ones \
  --input x=1,2,3,4 \
  --output y_prob

# Same graph on a GPU backend name (host-fallback; no SDK required)
uaii run examples/ir/toy_mlp.uaii.json \
  --backend cuda --force-host-fallback \
  --weight-init ones \
  --input x=1,2,3,4 \
  --output y_prob
```

Use `--weights-dir` when `weight_ref` files exist as raw f32 blobs.
