"""Portable entry points for paper ModPINN training."""

from pathlib import Path
import json
from pytorch_optimizer.optimizer.soap import SOAP
import argparse
import copy
import hashlib
import math
import numpy as np
import h5py
import torch
from .model import ModPINN, DTYPES
from .physics import make_grid, loss, remesh


def data_root():
    package = Path(__file__).resolve().parent / "data"
    source = Path(__file__).resolve().parents[2] / "datasets"
    for directory in (package, source):
        if (directory / "experiments.json").is_file():
            return directory
    raise FileNotFoundError("Bundled ModPINN data are missing; reinstall the complete package")


def catalogue():
    return json.loads((data_root() / "experiments.json").read_text())["experiments"]


def make_optimizer(model, config):
    return SOAP(model.parameters(), lr=config["training_process"]["parameters"]["learning_rate"])


def validate_config(config):
    """Validate public numeric configuration; never evaluate Python expressions."""
    c = copy.deepcopy(config)
    try:
        p = c["physical"]["parameters"]
        n = c["neural_parameters"]["general_parameters"]
        t = c["training_process"]
        if t["DTYPE"] not in DTYPES:
            raise ValueError("DTYPE must be torch.float32 or torch.float64")
        for key in ("N_t", "N_x"):
            value = p[key]
            if isinstance(value, bool) or str(value) != str(int(value)) or int(value) < 2:
                raise ValueError(f"{key} must be an integer of at least 2")
            p[key] = str(int(value))
        for key in ("t_range", "x_range"):
            bounds = p[key]
            if (
                len(bounds) != 2
                or not all((math.isfinite(float(v)) for v in bounds))
                or bounds[1] <= bounds[0]
            ):
                raise ValueError(f"{key} must contain increasing finite bounds")
        if p["x_range"][0] < 0:
            raise ValueError("The radial domain must be nonnegative")
        for key in ("eps_causality", "eps_spatial", "domain_step"):
            if not math.isfinite(float(p[key])) or p[key] < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        for key in ("number_hidden", "number_neurons"):
            if isinstance(n[key], bool) or int(n[key]) != n[key] or n[key] < 1:
                raise ValueError(f"{key} must be a positive integer")
        if n["num_inputs"] != 2 or n["num_outputs"] != 3:
            raise ValueError("ModPINN requires two inputs and three outputs")
        params = t["parameters"]
        if params["optimizer"] != "SOAP":
            raise ValueError("The paper configurations use SOAP")
        if not math.isfinite(float(params["learning_rate"])) or params["learning_rate"] <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if (
            isinstance(params["epochs"], bool)
            or int(params["epochs"]) != params["epochs"]
            or params["epochs"] < 1
        ):
            raise ValueError("epochs must be a positive integer")
        for key in ("update_from", "update_each"):
            value = t["domain_updating"][key]
            if (
                isinstance(value, bool)
                or int(value) != value
                or value < (1 if key == "update_each" else 0)
            ):
                raise ValueError(f"{key} is invalid")
        weights = c["neural_parameters"]["loss_function_parameters"]
        if len(weights["w_initial"]) != 4 or len(weights["w_boundary"]) != 2:
            raise ValueError("Expected four initial and two boundary weights")
        if any(
            (
                not math.isfinite(float(v)) or v < 0
                for v in weights["w_initial"] + weights["w_boundary"]
            )
        ):
            raise ValueError("Loss weights must be finite and nonnegative")
        if t["import"]["load_weights"]:
            raise ValueError(
                "Use evaluate for saved states; training starts from its configured seed"
            )
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Invalid or incomplete ModPINN configuration") from exc
    return c


def _reference_path(config):
    path = Path(config["training_process"]["import"]["data_path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("data_path must name a bundled reference; custom data use --reference")
    resolved = data_root() / path
    if not resolved.is_file():
        raise FileNotFoundError("Reference data not found: " + str(path))
    return resolved


def _read_reference(path):
    required = (
        "t_truth",
        "r_truth",
        "phi_truth",
        "alpha_truth",
        "Q_truth",
        "phi_0",
        "alpha_0",
        "a_0",
        "phi_b",
        "alpha_b",
        "a_b",
    )
    with h5py.File(path, "r") as handle:
        missing = set(required) - set(handle)
        if missing:
            raise ValueError("Missing reference arrays: " + ", ".join(sorted(missing)))
        arrays = {key: np.asarray(handle[key]).reshape(-1) for key in required}
    groups = (required[:5], required[5:8], required[8:])
    for names in groups:
        lengths = {len(arrays[key]) for key in names}
        if len(lengths) != 1 or 0 in lengths:
            raise ValueError("Reference field groups must have equal nonzero lengths")
    if any(not np.isfinite(arrays[key]).all() for key in required[:2]):
        raise ValueError("Reference coordinates must be finite")
    return arrays


def _conditions(reference, config, *, resample=False):
    p = config["physical"]["parameters"]
    (dtype, device) = (
        DTYPES[config["training_process"]["DTYPE"]],
        config["training_process"]["device"],
    )

    def tensor(key, count):
        values = reference[key]
        if len(values) != count:
            if not resample:
                raise ValueError(
                    "Reference boundary/initial grid does not match configuration; use --resample-conditions explicitly"
                )
            values = np.interp(np.linspace(0, 1, count), np.linspace(0, 1, len(values)), values)
        return torch.tensor(values, dtype=dtype, device=device).reshape(-1, 1)

    def fields(suffix, count):
        (phi, alpha, a) = [tensor(key + suffix, count) for key in ("phi", "alpha", "a")]
        return torch.cat((phi, alpha, 1 - 1 / a.square()), dim=1)

    return (fields("_0", int(p["N_x"])), fields("_b", int(p["N_t"])))


def reference_metrics(model, reference, *, limit=None):
    """Paper relative-L2 diagnostics. They do not enter the residual objective."""
    size = len(reference["t_truth"])
    indices = (
        np.arange(size) if limit is None else np.linspace(0, size - 1, min(size, limit), dtype=int)
    )
    numerator = torch.zeros(3, dtype=torch.float64)
    denominator = torch.zeros(3, dtype=torch.float64)
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            ix = indices[start : start + 4096]
            coords = np.column_stack((reference["t_truth"][ix], reference["r_truth"][ix]))
            x = torch.tensor(coords, dtype=model.DTYPE, device=model.device)
            (phi, alpha, q) = model(x)
            if not all(torch.isfinite(field).all() for field in (phi, alpha, q)):
                raise FloatingPointError("Nonfinite prediction during reference evaluation")
            pred = torch.cat((phi, alpha.clamp(0, 1), q.clamp(0, 0.999)), dim=1)
            truth = torch.tensor(
                np.column_stack(
                    [reference[key][ix] for key in ("phi_truth", "alpha_truth", "Q_truth")]
                ),
                dtype=model.DTYPE,
                device=model.device,
            )
            numerator += (truth - pred).square().nansum(dim=0).cpu().double()
            denominator += truth.square().nansum(dim=0).cpu().double()
    relative = (numerator / denominator).sqrt().tolist()
    if not all((math.isfinite(value) for value in relative)):
        raise FloatingPointError(
            "Reference metrics are nonfinite; check predictions and reference norms"
        )
    return dict(
        zip(("relative_l2_phi", "relative_l2_alpha", "relative_l2_Q"), relative),
        relative_l2_sum=sum(relative),
        points=len(indices),
    )


def _save_model(path, model, config, experiment, completed):
    torch.save(
        {
            "schema_version": 1,
            "config": config,
            "experiment": experiment,
            "epochs_completed": completed,
            "state_dict": model.state_dict(),
        },
        path,
    )


def train(
    experiment,
    output,
    *,
    smoke=False,
    epochs=None,
    device="cpu",
    config=None,
    reference=None,
    resample_conditions=False,
    log_every=100,
):
    """Run one case into a new directory; never overwrite retained examples."""
    if isinstance(log_every, bool) or not isinstance(log_every, int) or log_every < 1:
        raise ValueError("log_every must be a positive integer")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    bundled = data_root().resolve()
    if output.is_relative_to(bundled) and not output.is_relative_to(bundled / "runs"):
        raise ValueError("Inside bundled data, training output must be under runs/")
    record = (
        catalogue()[experiment]
        if config is None
        else {"config": config, "update_order": "before_step"}
    )
    c = copy.deepcopy(record["config"])
    c["training_process"]["device"] = device
    if device not in ("cpu", "cuda", "mps"):
        raise ValueError("device must be cpu, cuda or mps")
    if device == "cuda" and (not torch.cuda.is_available()):
        raise ValueError("CUDA is not available")
    if device == "mps" and (
        not torch.backends.mps.is_available() or c["training_process"]["DTYPE"] == "torch.float64"
    ):
        raise ValueError("MPS requires an available device and float32")
    if smoke:
        c["physical"]["parameters"].update(N_t="8", N_x="8")
        c["neural_parameters"]["general_parameters"].update(number_hidden=1, number_neurons=8)
        c["training_process"]["parameters"]["epochs"] = 3
    if epochs is not None:
        c["training_process"]["parameters"]["epochs"] = epochs
    c = validate_config(c)
    path = Path(reference).resolve() if reference is not None else _reference_path(c)
    ref = _read_reference(path)
    (initial, boundary) = _conditions(ref, c, resample=smoke or resample_conditions)
    seed = c["training_process"]["parameters"]["random_seed"]
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    grid = make_grid(c)
    model = ModPINN(config=c).to(device=device, dtype=DTYPES[c["training_process"]["DTYPE"]])
    (model.N_t, model.N_x) = (
        int(c["physical"]["parameters"]["N_t"]),
        int(c["physical"]["parameters"]["N_x"]),
    )
    model.AUC_hist = []
    (model.xmin, model.xmax) = (float(np.nanmin(ref["r_truth"])), float(np.nanmax(ref["r_truth"])))
    optimizer = make_optimizer(model, c)
    initial_parameters = torch.cat([p.detach().flatten().cpu() for p in model.parameters()])
    output.mkdir(parents=True, exist_ok=False)
    saved_config = copy.deepcopy(c)
    if reference is not None:
        import shutil

        shutil.copyfile(path, output / "reference.h5")
        saved_config["training_process"]["import"]["data_path"] = "reference.h5"
    (output / "config.json").write_text(json.dumps(saved_config, indent=2) + "\n")
    history = []
    adaptive_updates = 0
    total = c["training_process"]["parameters"]["epochs"]
    update = c["training_process"]["domain_updating"]
    for epoch in range(total):
        diagnostics = reference_metrics(model, ref, limit=64 if smoke else None)
        optimizer.zero_grad()
        grid.grad = None
        objective = loss(model, grid, initial, boundary)
        if not torch.isfinite(objective):
            raise FloatingPointError(f"Nonfinite objective at epoch {epoch}")
        change_grid = epoch >= update["update_from"] and epoch % update["update_each"] == 0
        if change_grid and record["update_order"] == "before_step":
            grid = remesh(c, model, grid)
            adaptive_updates += 1
        objective.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if change_grid and record["update_order"] == "after_step":
            grid = remesh(c, model, grid)
            adaptive_updates += 1
        history.append(
            {
                "epoch": epoch,
                "loss": objective.item(),
                "causal_area": model.AUC_hist[-1],
                **diagnostics,
            }
        )
        if (epoch + 1) % log_every == 0 or epoch + 1 == total:
            print(
                json.dumps({"epoch": epoch + 1, "epochs": total, "loss": objective.item()}),
                flush=True,
            )
            _save_model(output / "model.pt", model, saved_config, experiment, epoch + 1)
            (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    final_metrics = reference_metrics(model, ref)
    delta = torch.linalg.vector_norm(
        torch.cat([p.detach().flatten().cpu() for p in model.parameters()]) - initial_parameters
    ).item()
    result = {
        "experiment": experiment,
        "epochs_completed": total,
        "smoke": smoke,
        "adaptive_updates": adaptive_updates,
        "parameter_change_l2": delta,
        "reference_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "update_order": record["update_order"],
        **final_metrics,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def evaluate(checkpoint, *, reference=None):
    """Evaluate an emitted checkpoint, or use a catalogue record for example weights."""
    checkpoint = Path(checkpoint)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = copy.deepcopy(bundle["config"])
    config["training_process"]["device"] = "cpu"
    model = ModPINN(config=config).to(dtype=DTYPES[config["training_process"]["DTYPE"]])
    model.load_state_dict(bundle["state_dict"], strict=True)
    if reference is not None:
        path = Path(reference)
    elif config["training_process"]["import"]["data_path"] == "reference.h5":
        path = checkpoint.parent / "reference.h5"
    else:
        path = _reference_path(config)
    return reference_metrics(model, _read_reference(path))


def load_pretrained(experiment):
    """Load one of the four bundled example states on CPU."""
    record = catalogue()[experiment]
    if "pretrained" not in record:
        raise ValueError("This experiment has no bundled pretrained example")
    config = copy.deepcopy(record["config"])
    config["training_process"]["device"] = "cpu"
    model = ModPINN(config=config).to(dtype=DTYPES[config["training_process"]["DTYPE"]])
    state = torch.load(data_root() / record["pretrained"], map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def evaluate_pretrained(experiment):
    record = catalogue()[experiment]
    return reference_metrics(
        load_pretrained(experiment), _read_reference(_reference_path(record["config"]))
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List every expanded paper training case")
    listing.add_argument(
        "--family", choices=["best", "benchmark", "scaling", "ablation", "remeshing", "causality"]
    )
    fitting = commands.add_parser("train", help="Train one case into a new output directory")
    source = fitting.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment")
    source.add_argument("--config", type=Path)
    fitting.add_argument("--output", required=True, type=Path)
    fitting.add_argument("--smoke", action="store_true")
    fitting.add_argument("--epochs", type=int)
    fitting.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    fitting.add_argument("--reference", type=Path)
    fitting.add_argument("--resample-conditions", action="store_true")
    fitting.add_argument("--log-every", type=int, default=100)
    evaluation = commands.add_parser("evaluate", help="Evaluate a training checkpoint")
    evaluation.add_argument("checkpoint", type=Path, nargs="?")
    evaluation.add_argument("--experiment")
    evaluation.add_argument("--reference", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "list":
            print(
                "\n".join(
                    (
                        key
                        for (key, case) in catalogue().items()
                        if args.family is None or case["family"] == args.family
                    )
                )
            )
        elif args.command == "train":
            if args.log_every < 1:
                raise ValueError("log-every must be positive")
            config = json.loads(args.config.read_text()) if args.config else None
            result = train(
                args.experiment or "custom",
                args.output,
                smoke=args.smoke,
                epochs=args.epochs,
                device=args.device,
                config=config,
                reference=args.reference,
                resample_conditions=args.resample_conditions,
                log_every=args.log_every,
            )
            print(json.dumps(result, indent=2))
        else:
            if bool(args.checkpoint) == bool(args.experiment):
                raise ValueError("Provide a checkpoint or --experiment")
            if args.experiment and args.reference:
                raise ValueError("--reference applies to a checkpoint, not a bundled example")
            result = (
                evaluate_pretrained(args.experiment)
                if args.experiment
                else evaluate(args.checkpoint, reference=args.reference)
            )
            print(json.dumps(result, indent=2))
    except (ValueError, KeyError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
