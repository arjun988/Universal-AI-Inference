# Phase 3 CPU examples

## Built-in demos (recommended)

```bash
uaii run --demo toy_mlp
uaii run --demo tiny_block
```

`toy_mlp` checks a known expected softmax output (`0.25` uniform with all-ones weights).

## Run hand-authored IR

```bash
uaii run examples/ir/toy_mlp.uaii.json \
  --weight-init ones \
  --input x=1,2,3,4 \
  --output y_prob
```

Use `--weights-dir` when `weight_ref` files exist as raw f32 blobs.
