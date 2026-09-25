# Training

Start with the [minimal example](quickstart.md). The CLI also works as
`python -m modpinn.training`.

## Choose a configuration

```bash
modpinn list
modpinn list --family benchmark
```

| Family | Runs | Purpose |
| --- | ---: | --- |
| `best` | 4 | Selected configurations for the four standard amplitudes |
| `benchmark` | 40 | Ten seeds for each standard amplitude |
| `causality` | 100 | Resolution, width/depth, and causal-weight sweeps |
| `remeshing` | 100 | Five remeshing strengths across amplitudes and seeds |
| `scaling` | 4 | Supercritical configurations used in the scaling study |
| `ablation` | 1 | The full ModPINN case from the ablation comparison |

The 249 cases specify seeds, grids, architectures, precision, SOAP settings,
and epoch budgets. Full runs need substantially more work than smoke tests.

```bash
modpinn train --experiment best/amplitude-0.10625 --output datasets/runs/full-case
```

CPU is the default. Use `--device cuda` or `--device mps` on supported systems;
MPS requires float32. `--epochs 10` shortens a run; `--log-every 1000` sets
checkpoint frequency.

## Outputs

| File | Contents |
| --- | --- |
| `config.json` | Effective configuration, including smoke overrides |
| `model.pt` | Latest saved model state and portable configuration |
| `history.json` | Per-step objective, causal area, and pre-step reference errors |
| `metrics.json` | Final full-grid errors, parameter change, and run settings |

Use a new directory for each run. Weights and history are saved at logging
intervals and completion. The checkpoint is the latest state, not the minimum-error
state. Interrupted checkpoints support inference; start a new directory for new training.

## Custom settings

Export a complete configuration:

```python
import json
from pathlib import Path
from modpinn.training import catalogue

config = catalogue()["best/amplitude-0.1"]["config"]
config["training_process"]["parameters"]["random_seed"] = 17
config["training_process"]["parameters"]["epochs"] = 10
Path("custom.json").write_text(json.dumps(config, indent=2))
```

```bash
modpinn train --config custom.json --output datasets/runs/custom
```

For new physical data, add `--reference reference.h5`; see the
[array schema](reference-data.md). The reference is copied into the run.
Changing amplitude requires matching initial and boundary data. Changing grid
size requires matching arrays or `--resample-conditions`.

`number_hidden` counts hidden layers after the first embedding-to-hidden layer.
The embedding has 64 features and 32 RBFs. Use `w_initial` and `w_boundary`
for boundary/initial weights. Legacy `n_freq`, `scale`, `hidden_activation`,
`num_experts`, and per-equation weights are inactive provenance fields.
Configurations are data, not executable Python.

## Reproducibility

The [method](method.md) specifies residuals, weighting, and remeshing.
Reference initial and boundary values enter the loss; full-field errors are
diagnostics. Smoke tests check execution, while [checkpoint results](results.md)
measure selected-state accuracy.

Original POSIX/CUDA launcher settings:

```bash
CUBLAS_WORKSPACE_CONFIG=:16:8 CUDA_LAUNCH_BLOCKING=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  modpinn train --experiment best/amplitude-0.1 --device cuda --output datasets/runs/cuda-case
```

Fixed seeds do not guarantee bitwise agreement across hardware or library versions.
