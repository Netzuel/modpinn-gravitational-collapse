# Jupyter tutorials

| Notebook | What you will do |
| --- | --- |
| [1. Train and reload](quickstart.ipynb) | Run three CPU training steps, save a checkpoint, and check its evaluation |
| [2. Predict, plot, and differentiate](inference.ipynb) | Load a bundled model, plot radial profiles, and compute first and second derivatives |

After [installation](../installation.md), run from the repository root:

```bash
python -m pip install '.[notebooks]'
python -m ipykernel install --user --name modpinn --display-name "Python (ModPINN)"
python -m jupyterlab
```

Open a notebook, select **Python (ModPINN)**, then **Run → Run All Cells**.
Both use CPU and bundled data. Files have no saved outputs.

Training creates a fresh run directory for its three-step workflow check.
Inference reads the bundled model.
