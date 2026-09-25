# Installation

Use Python 3.10 or later; CPU checks used 3.10.20. Later versions must support
the pinned dependencies.

From the repository root:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, create the environment with `py -3.10 -m venv .venv` and activate
it with `.venv\Scripts\Activate.ps1` in PowerShell.

| Package | Version | Purpose |
| --- | --- | --- |
| PyTorch (`torch`) | 2.11.0 | Networks, derivatives, and training |
| NumPy | 2.2.5 | Numerical arrays |
| h5py | 3.16.0 | Bundled HDF5 reference data |
| pytorch-optimizer | 3.10.0 | SOAP optimizer |
| Matplotlib | 3.10.8 | Scientific figures |

Check a bundled checkpoint:

```bash
modpinn evaluate --experiment best/amplitude-0.1
```

Optional installations:

| Command | Purpose |
| --- | --- |
| `python -m pip install .` | Runtime only, without Matplotlib |
| `python -m pip install '.[test]'` | Tests; run `python -m pytest -q` |
| `python -m pip install '.[notebooks]'` | JupyterLab and Python kernel |

Original-grid tests need more memory than the tutorial. Continue with the
[minimal example](quickstart.md) or [notebooks](examples/README.md).
