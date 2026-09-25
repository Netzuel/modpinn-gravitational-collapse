# Architecture diagram

The README schematic adapts Figure 2 of the
[article](https://doi.org/10.1088/2632-2153/ae5459) to the implemented ModPINN.
See [method](method.md) for feature definitions and the physics loss.

With Tectonic and Poppler's `pdftocairo` installed, run:

```bash
bash src/diagrams/render.sh
```

The shared source, `src/diagrams/modpinn.tex`, generates transparent light/dark
SVGs in `datasets/figures/architecture/`. Temporary PDFs and logs are removed.
The README selects the theme. These tools are unnecessary for training.
