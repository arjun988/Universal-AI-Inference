# Examples

| Path | Contents |
|---|---|
| `ir/` | Hand-authored UAII IR graphs |
| `cpu/` | Run notes for CPU / multi-backend execution |
| `models/` | Tiny GGUF/Safetensors fixtures (written by Phase 4 demos) |
| `python/` | Phase 7 load → run → profile SDK example |

```bash
uaii doctor --load-plugins
uaii run --demo toy_mlp
uaii run --demo parity
uaii run examples/ir/toy_mlp.uaii.json --backend cuda --force-host-fallback \
  --weight-init ones --input x=1,2,3,4 --output y_prob
```
