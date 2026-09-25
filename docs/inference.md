# Inference and evaluation

Predict with a bundled checkpoint:

```python
import torch
from modpinn.training import load_pretrained

model = load_pretrained("best/amplitude-0.1")
points = torch.tensor([[0.1, 0.2], [1.0, 0.8]], dtype=model.DTYPE)
with torch.no_grad():
    phi, alpha, compactness = model(points)
print(phi, alpha, compactness)
```

Input shape is `(N, 2)`: time, then radius. Outputs are `(N, 1)` tensors:
scalar field, lapse, and compactness `Q = 1 - 1/a²`. Each model fits one
initial amplitude. Reconstruct `a = 1 / sqrt(1 - Q)` only where finite and
physically valid.

For derivatives, set `points.requires_grad_(True)` and omit
`torch.no_grad()`; second derivatives are supported.

Evaluate your own saved run:

```bash
modpinn evaluate datasets/runs/example/model.pt
```

Loading uses CPU mapping, `weights_only=True`, and strict parameter matching.
Bundled checkpoints contain state dictionaries; new runs also store their
configuration. Only load checkpoints from trusted sources.

Bundled data work outside the source checkout. Exact floating-point agreement
across devices or library versions is not guaranteed. See [results](results.md)
for bundled evaluation commands, error definitions, and retained values.
