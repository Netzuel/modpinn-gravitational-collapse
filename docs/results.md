# Retained checkpoint results

Evaluate the four bundled states on CPU:

```bash
modpinn evaluate --experiment best/amplitude-0.1
modpinn evaluate --experiment best/amplitude-0.10625
modpinn evaluate --experiment best/amplitude-0.1125
modpinn evaluate --experiment best/amplitude-0.125
```

## Error definition

For field $`u`$:

```math
 \ell_2^u=\left[\frac{\sum_{k\in\mathcal V_u}(u_k-\widehat u_k)^2}
 {\sum_{k\in\mathcal V_u}u_k^2}\right]^{1/2},\qquad
 \ell_2^{\mathrm{total}}=\ell_2^\phi+\ell_2^\alpha+\ell_2^C.
```

$`\mathcal V_u`$ contains non-NaN reference entries; predictions must be finite.
Metric clamps are $`[0,1]`$ for lapse and $`[0,0.999]`$ for compactness; the
scalar field is unclipped. Total error sums three relative norms, without spatial
quadrature or time averaging. Evaluation uses batches of 4096, float64 sums,
and each checkpoint's field dtype.

| Initial amplitude | Grid points | Scalar field | Lapse | Compactness | Total |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 65536 | 0.02781925 | 0.00874562 | 0.01489495 | 0.05145982 |
| 0.10625 | 65536 | 0.02941509 | 0.00780692 | 0.01468216 | 0.05190417 |
| 0.1125 | 16384 | 0.25292825 | 0.02578152 | 0.18578772 | 0.46449749 |
| 0.125 | 16384 | 0.16980352 | 0.00731864 | 0.03357820 | 0.21070035 |

Totals match the original minimum-error histories within $`8\times10^{-8}`$.
Independent values are in `tests/fixtures/pretrained-errors.json`; hashes,
grid sizes, and profile times are in
[the evaluation record](../datasets/figures/evaluation.json).

## Figures

- **Spacetime:** raw predictions as filled contours; white reference isolines
  at scalar levels −0.2, −0.05, 0.05, 0.2 and lapse/compactness levels 0.2, 0.5, 0.8.
  Out-of-range levels have no contour. Axes and color scales are linear and
  shared by field. Charcoal marks scalar zero and lapse/compactness 0.5.
- **Profiles:** earliest reference time of maximum finite compactness;
  solid reference curves and dashed predictions, with shared limits per field.

Both mask missing reference values and show unclipped predictions.

## Scope

The table reports four selected checkpoints. The [paper](https://doi.org/10.1088/2632-2153/ae5459)
contains the comparative study across architectures and training runs.
This package provides ModPINN training configurations and checkpoint evaluation;
smoke mode checks execution. Fresh runs save their final state, so their errors
can differ from the selected checkpoints.
