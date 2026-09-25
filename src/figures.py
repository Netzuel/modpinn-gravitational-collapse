"""Reproduce checkpoint errors and transparent vector figures without retraining."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import torch

from modpinn.training import (
    _read_reference,
    _reference_path,
    catalogue,
    data_root,
    load_pretrained,
    reference_metrics,
)

FIELDS = ("phi", "alpha", "Q")
LABELS = (r"$\phi$", r"$\alpha$", r"$C = 2m/r$")
FIELD_CMAP = LinearSegmentedColormap.from_list(
    "blue_charcoal_amber", ["#56b4e9", "#276487", "#272b35", "#94602d", "#efb64c"], N=256
)


def collect():
    """Evaluate all retained states; choose slices by reference compactness alone."""
    records, profiles, spacetime = {}, [], []
    for case, record in catalogue().items():
        if "pretrained" not in record:
            continue
        reference_path = _reference_path(record["config"])
        reference = _read_reference(reference_path)
        model = load_pretrained(case)
        metrics = reference_metrics(model, reference)
        coordinates = np.column_stack((reference["t_truth"], reference["r_truth"]))
        nt = len(np.unique(reference["t_truth"]))
        nr = len(coordinates) // nt
        grid = coordinates.reshape(nt, nr, 2)
        if not np.all(grid[:, :, 0] == grid[:, :1, 0]):
            raise ValueError("Expected time-major reference coordinates")
        with torch.no_grad():
            prediction = np.concatenate(
                [
                    torch.cat(model(torch.tensor(batch, dtype=model.DTYPE)), dim=1).numpy()
                    for batch in np.array_split(coordinates, max(1, len(coordinates) // 4096))
                ]
            )
        if not np.isfinite(prediction).all():
            raise FloatingPointError("Nonfinite spacetime prediction")
        truth_grid = np.column_stack([reference[field + "_truth"] for field in FIELDS])
        spacetime.append((grid, truth_grid.reshape(nt, nr, 3), prediction.reshape(nt, nr, 3)))
        # Use the earliest row containing the global finite maximum of C.
        peak = int(np.nanargmax(reference["Q_truth"]))
        selected_time = reference["t_truth"][peak]
        ix = np.flatnonzero(reference["t_truth"] == selected_time)
        ix = ix[np.argsort(reference["r_truth"][ix])]
        coords = np.column_stack((reference["t_truth"][ix], reference["r_truth"][ix]))
        with torch.no_grad():
            values = model(torch.tensor(coords, dtype=model.DTYPE))
            predictions = torch.cat(values, dim=1).numpy()
        if not np.isfinite(predictions).all():
            raise FloatingPointError("Nonfinite profile prediction")
        truth = np.column_stack([reference[field + "_truth"][ix] for field in FIELDS])
        amplitude = record["config"]["physical"]["parameters"]["phi_0"]
        profiles.append((amplitude, float(selected_time), coords[:, 1], truth, predictions))
        records[case] = {
            **metrics,
            "profile_time": float(selected_time),
            "profile_points": len(ix),
            "spacetime_shape": [nt, nr],
            "reference_nan_counts": np.isnan(truth_grid).sum(axis=0).tolist(),
            "reference": str(reference_path.relative_to(data_root())),
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "checkpoint": record["pretrained"],
            "checkpoint_sha256": hashlib.sha256(
                (data_root() / record["pretrained"]).read_bytes()
            ).hexdigest(),
        }
    return records, profiles, spacetime


def save_vector(fig, path):
    fig.savefig(
        path, transparent=True, metadata={"Date": None, "Creator": "ModPINN figure reproduction"}
    )
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def render(profiles, spacetime, output, theme):
    foreground = "#202124" if theme == "light" else "#E8EAF0"
    prediction_color = "#D55E00" if theme == "light" else "#E69F00"
    style = {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.axisbelow": True,
        "axes.edgecolor": foreground,
        "axes.labelcolor": foreground,
        "text.color": foreground,
        "xtick.color": foreground,
        "ytick.color": foreground,
        "grid.color": foreground,
        "svg.hashsalt": "modpinn-collapse",
    }
    with plt.rc_context(style):
        fig, axes = plt.subplots(3, 4, figsize=(11.2, 6.8), sharex="col", sharey="row")
        fig.subplots_adjust(left=0.10, right=0.985, bottom=0.09, top=0.85, hspace=0.24, wspace=0.13)
        for col, (amplitude, time, radii, truth, predictions) in enumerate(profiles):
            for row, label in enumerate(LABELS):
                ax = axes[row, col]
                ax.set_axisbelow(True)
                ax.grid(alpha=0.15, linewidth=0.5, zorder=0)
                ax.plot(radii, truth[:, row], color=foreground, lw=1.4, label="Numerical reference")
                ax.plot(
                    radii,
                    predictions[:, row],
                    color=prediction_color,
                    lw=1.3,
                    ls="--",
                    label="ModPINN checkpoint",
                )
                ax.set_xlim(float(radii.min()), float(radii.max()))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3), useMathText=True)
                if col == 0:
                    ax.set_ylabel(label, rotation=0, labelpad=26, va="center")
                if row == 0:
                    ax.set_title(rf"$\phi_0={amplitude:g}$" + "\n" + rf"$t={time:.3f}$")
                if row == 2:
                    ax.set_xlabel(r"$r$")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.99)
        )
        save_vector(fig, output / f"profiles-{theme}.svg")

        fig = plt.figure(figsize=(11.6, 9.0))
        layout = fig.add_gridspec(3, 1, left=0.065, right=0.985, bottom=0.07, top=0.90, hspace=0.48)
        field_titles = (r"Scalar field $\phi$", r"Lapse $\alpha$", r"Compactness $C=2m/r$")
        scalar_limit = (
            np.ceil(
                max(
                    np.nanmax(np.abs(x[:, :, 0]))
                    for _, truth, pred in spacetime
                    for x in (truth, pred)
                )
                * 10
            )
            / 10
        )
        for row, label in enumerate(LABELS):
            section = layout[row].subgridspec(
                2, 4, height_ratios=[0.065, 1], hspace=0.40, wspace=0.14
            )
            bounds = layout[row].get_position(fig)
            fig.text(bounds.x0, bounds.y1 - 0.005, field_titles[row], fontsize=12, va="center")
            levels = (
                np.linspace(-scalar_limit, scalar_limit, 49) if row == 0 else np.linspace(0, 1, 49)
            )
            isolines = [-0.2, -0.05, 0.05, 0.2] if row == 0 else [0.2, 0.5, 0.8]
            for col, ((grid, truth, pred), profile) in enumerate(zip(spacetime, profiles)):
                ax = fig.add_subplot(section[1, col])
                ax.set_axisbelow(True)
                r, t = grid[:, :, 1], grid[:, :, 0]
                invalid = ~np.isfinite(truth[:, :, row])
                field = np.ma.array(pred[:, :, row], mask=invalid)
                fill = ax.contourf(
                    r,
                    t,
                    field,
                    levels=levels,
                    cmap=FIELD_CMAP,
                    antialiased=False,
                    corner_mask=False,
                )
                # Cover vector-renderer seams without adding mesh boundaries.
                fill.set_edgecolor("face")
                fill.set_linewidth(0.2)
                contours = ax.contour(
                    r,
                    t,
                    np.ma.masked_invalid(truth[:, :, row]),
                    levels=isolines,
                    colors="white",
                    linewidths=0.7,
                    linestyles="solid",
                    corner_mask=False,
                )
                contours.set_path_effects(
                    [
                        path_effects.Stroke(linewidth=1.05, foreground="#333333"),
                        path_effects.Normal(),
                    ]
                )
                ax.set_xlim(float(r.min()), float(r.max()))
                ax.set_ylim(float(t.min()), float(t.max()))
                ax.set_xticks([float(r.min()), 5, float(r.max())], ["0.01", "5", "10.01"])
                ax.set_yticks([0, 5, 10])
                ax.tick_params(length=3, width=0.65, pad=4)
                for spine in ax.spines.values():
                    spine.set_linewidth(0.65)
                ax.minorticks_off()
                ax.get_xticklabels()[0].set_horizontalalignment("left")
                ax.get_xticklabels()[-1].set_horizontalalignment("right")
                if row == 0:
                    box = ax.get_position()
                    fig.text(
                        (box.x0 + box.x1) / 2,
                        0.96,
                        rf"$\phi_0={profile[0]:g}$",
                        ha="center",
                        fontsize=13,
                    )
                if row == 2:
                    ax.set_xlabel(r"$r$")
                else:
                    ax.tick_params(labelbottom=False)
                if col == 0:
                    ax.set_ylabel(r"$t$", rotation=0, labelpad=12, va="center")
                else:
                    ax.tick_params(labelleft=False)
            colorbar = fig.colorbar(
                fill, cax=fig.add_subplot(section[0, 1:3]), orientation="horizontal"
            )
            colorbar.set_ticks(
                [-scalar_limit, -scalar_limit / 2, 0, scalar_limit / 2, scalar_limit]
                if row == 0
                else [0, 0.25, 0.5, 0.75, 1]
            )
            colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
            colorbar.ax.tick_params(length=2.5, width=0.6, pad=3, labelsize=10)
            colorbar.ax.spines["outline"].set_linewidth(0.6)
            if colorbar.solids is not None:
                colorbar.solids.set_rasterized(False)
                colorbar.solids.set_edgecolor("face")
        save_vector(fig, output / f"spacetime-{theme}.svg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/figures"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    records, profiles, spacetime = collect()
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {
        "scope": "CPU evaluation of retained checkpoints; no retraining or multi-seed statistics",
        "profile_selection": "Earliest reference time containing the maximum finite compactness",
        "profile_values": "Raw network fields; diagnostic clipping applies only to error metrics",
        "spacetime_values": "Raw network fields on reference coordinates; reference NaNs masked",
        "spacetime_axes": "Horizontal: radius on linear scale; vertical: time",
        "reference_contours": {
            "phi": [-0.2, -0.05, 0.05, 0.2],
            "alpha": [0.2, 0.5, 0.8],
            "Q": [0.2, 0.5, 0.8],
        },
        "cases": records,
    }
    (args.output / "evaluation.json").write_text(json.dumps(payload, indent=2) + "\n")
    for theme in ("light", "dark"):
        render(profiles, spacetime, args.output, theme)
    print(json.dumps({key: record["relative_l2_sum"] for key, record in records.items()}, indent=2))


if __name__ == "__main__":
    main()
