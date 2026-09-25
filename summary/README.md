# Verification

The package includes 249 training configurations, 16 numerical references, four
selected checkpoints, and two tutorials. See [results](../docs/results.md) for
the evaluation protocol and [the repository map](repository-map.md) for file roles.

## Run the checks

```bash
python -m pytest -q
ruff check src tests
ruff format --check src tests
pyright
python src/verify.py
```

Tests cover model fields and derivatives, residuals, remeshing, short SOAP runs,
checkpoint reload, tutorials, and documentation links. Numerical fixtures provide
independent expected values. The verifier checks the 21 bundled data hashes,
configuration consistency, and common sensitive-data patterns, including metadata
in checkpoints, notebooks, and HDF5 files.

These checks validate software behavior and selected-checkpoint evaluation.
Use full training runs for convergence studies. The [paper](https://doi.org/10.1088/2632-2153/ae5459)
reports the broader scientific study.
