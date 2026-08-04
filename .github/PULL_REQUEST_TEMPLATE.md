## Summary

<!-- Why this change exists -->

## Changes

-

## Test plan

- [ ] `cmake -S . -B build && cmake --build build`
- [ ] `uaii doctor --load-plugins`
- [ ] `ctest --test-dir build --output-on-failure` (if core touched)
- [ ] Docs updated (if API / behavior changed)
