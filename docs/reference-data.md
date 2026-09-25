# Reference data

The 16 bundled HDF5 files contain four standard amplitudes at three resolutions
and four supercritical cases. No download is needed.

Numerical simulations: Matthew W. Choptuik
([foundational work](https://doi.org/10.1103/PhysRevLett.70.9),
[data release](https://doi.org/10.5281/zenodo.18687035)). Software and trained
examples: Antonio Ferrer-Sánchez and the
[associated paper](https://doi.org/10.1088/2632-2153/ae5459).
Reuse terms: `datasets/LICENSE.txt` (installed: `modpinn/data/LICENSE.txt`).

## Array schema

| Names | Interpretation |
| --- | --- |
| `t_truth`, `r_truth` | Time-major coordinates for full-grid diagnostics |
| `phi_truth`, `alpha_truth`, `Q_truth` | Reference fields, aligned with coordinates |
| `phi_0`, `alpha_0`, `a_0` | Initial fields, one value per radial point |
| `phi_b`, `alpha_b`, `a_b` | Left-boundary fields, one value per time point |

Arrays are flattened in time-major order. Full-grid fields must have equal
lengths; initial and boundary arrays must match `N_x` and `N_t`, respectively.
Changing resolution requires explicit resampling. Training derives compactness
from `a_0` and `a_b`; its moving collocation grid is separate from this reference grid.

Missing interior values remain missing. Evaluation excludes NaN reference
entries and rejects nonfinite predictions; initial endpoints use the
[method guide's rule](method.md).

`datasets/checksums.json` records hashes for all references, four checkpoints,
and the catalogue; `python src/verify.py` checks them. Only the four standard
checkpoints have [reproduced results](results.md). The reference solver and
supercritical trained solutions are not bundled.
