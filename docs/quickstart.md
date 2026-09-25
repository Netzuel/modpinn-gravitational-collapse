# Minimal training example

Follow [installation](installation.md), then run this Python example from the
repository root:

```python
from modpinn.training import train, evaluate

result = train("best/amplitude-0.1", "datasets/runs/python-example", smoke=True)
metrics = evaluate("datasets/runs/python-example/model.pt")
print("Completed steps:", result["epochs_completed"])
print("Parameter change:", result["parameter_change_l2"])
print("Reloaded errors:", metrics)
```

Or use the equivalent CLI commands:

```bash
modpinn train --experiment best/amplitude-0.1 --smoke --output datasets/runs/cli-example
modpinn evaluate datasets/runs/cli-example/model.pt
```

Use a new output directory each time. Smoke mode performs three SOAP updates
and remeshing operations on an 8-by-8 grid with eight-neuron hidden layers.
Initial and boundary arrays are resampled explicitly.

Expected CPU output: three completed steps, parameter-change norm about
0.00584, and total field error about 8.455. Reloaded errors match the final
evaluation. Smoke mode checks the training and reload workflow.

Outputs: `config.json`, `model.pt`, `history.json`, and `metrics.json`.

For the full configured epoch budget:

```bash
modpinn train --experiment best/amplitude-0.1 --output datasets/runs/full-example
```

This longer run saves its latest state. See [results](results.md) for the
distinction between fresh runs and selected checkpoints. See [training](training.md) for settings,
[inference](inference.md) for predictions, and [notebooks](examples/README.md)
for interactive examples.
