# UAII IR examples

## `toy_mlp.uaii.json`

Hand-authored UAII IR for a tiny MLP:

`x → MatMul(w1) → Relu → MatMul(w2) → Softmax → y_prob`

### Commands (after building `uaii`)

```bash
uaii validate examples/ir/toy_mlp.uaii.json
uaii inspect examples/ir/toy_mlp.uaii.json
uaii graph examples/ir/toy_mlp.uaii.json --format plan
uaii graph examples/ir/toy_mlp.uaii.json --format dot > toy_mlp.dot
```

### Round-trip to binary (optional)

Use any small program or future `uaii convert`; for now you can load/save via the
C++ API (`uaii::ir::load_graph` / `save_graph`) once you build against `uaii::ir`.
