## Fully-Connected Neural Net — Study Notes

*Files*:  
`neural_network.py` ↔ core model / training loops  
`optimization_methods.py` ↔ helper optimizers + LR scheduling

### 1 — What this repo does

I’m experimenting with a plain-vanilla **L-layer dense network** trained on tabular or flattened-image data.  
Features I’ve added so far:

| Capability | Where it lives | Comment |
| :-- | :-- | :-- |
| He-init weight setup | `initialize_parameters` | scales by √(1/fan-in) |
| Forward pass `LINEAR → ReLU` × (L-1) → `LINEAR → Sigmoid` | `model_forward` | optional dropout |
| Cost functions | `compute_cost`, `compute_cost_with_regularization` | cross-entropy + optional L2 |
| Back-prop pipeline | `model_backward*` | supports L2 + dropout |
| Optimizers | in **optimization_methods.py** | GD, Momentum, Adam |
| Mini-batch maker | `random_mini_batches` | shuffles each epoch |
| LR schedulers | `update_lr`, `schedule_lr_decay` | simple inverse-time decay |
| Training loops | `model_1`, `model_2_optimized`, `model_3` | baseline / optimizer sweep / reg sweep |
| Hyper-param grid | `search_params` | loops over optimizer, LR, decay, λ |