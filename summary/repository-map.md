# Repository map

| Location | Responsibility | Entry point or consumer |
| --- | --- | --- |
| `src/modpinn/model.py` | Embedding, activations, quadratic layers, ModPINN | Training and checkpoint inference |
| `src/modpinn/physics.py` | Grid, residual loss, radial remeshing | Training and numerical tests |
| `src/modpinn/training.py` | Data loading, SOAP loop, checkpoint evaluation, CLI | `modpinn` |
| `src/figures.py` | Reevaluate four states and draw vector figures | `python src/figures.py` |
| `src/diagrams/` | Shared TikZ architecture source and SVG renderer | `bash src/diagrams/render.sh` |
| `src/verify.py` | Public-file privacy and data-integrity checks | `python src/verify.py` |
| `datasets/experiments.json` | 249 experiment configurations | Training catalogue |
| `datasets/references/` | 16 processed numerical references | Training conditions and evaluation |
| `datasets/pretrained/` | Four selected states | Evaluation and figure reproduction |
| `datasets/figures/` | Four SVGs and computed evaluation record | README and results guide |
| `datasets/figures/architecture/` | Light and dark ModPINN schematic SVGs | README architecture section |
| `datasets/runs/` | Optional generated training outputs, Git-ignored | Created explicitly by the user |
| `docs/` | Scientific formulation, tutorials, provenance, notebooks | README links |
| `docs/examples/` | Two basic Jupyter tutorials and kernel setup | Training/reload and prediction/plotting/derivatives |
| `tests/` | Numerical and workflow verification | `python -m pytest -q` |
| `summary/` | Verification guide and repository map | This index |

The only visible top-level directories are `src`, `datasets`, `docs`, `tests`,
and `summary`. Source installation uses `python -m pip install .`; figure
reproduction additionally requires the optional `figures` extra. The `notebooks`
extra supplies JupyterLab; the `test` extra includes notebook execution checks. The installed
Python import remains `modpinn`, and the console command remains `modpinn`.
