## Goal

Train and interpret transformers on Fibonacci-style sequences, i.e. those that generate a new element from the previous two.

## Results

### 1L, 1H, attention-only transformer

May not have the capacity needed to learn addition properly.
Low batch accuracy, low validation accuracy.

### 1L, 1H, transformer with MLP

Better batch accuracy (about 60%) but still low validation accuracy (about 23%).

## Misc TODO

- [x] Just learn addition first
- [x] Wandb integration
- [ ] Fibonacci sequence mod m, to avoid large digit number addition.
- [ ] Better data generation
- [ ] Improve tokenizer