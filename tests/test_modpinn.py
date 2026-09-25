"""Public API tests and independent pre-refactor numerical regression fixtures."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from modpinn.model import ModPINN
from modpinn.physics import make_grid, loss, remesh
from modpinn.training import catalogue, make_optimizer

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_legacy_numerical_parity(precision):
    torch.set_num_threads(1)
    config = json.loads((ROOT / "tests/fixtures/golden-config.json").read_text())
    config["training_process"]["DTYPE"] = "torch." + precision
    expected = np.load(ROOT / "tests/fixtures/golden.npz")
    prefix = precision + "__"
    torch.manual_seed(7)
    np.random.seed(7)
    model = ModPINN(config=config).to(dtype=getattr(torch, precision))
    model.N_t = model.N_x = 8
    model.AUC_hist = []
    model.xmin, model.xmax = 0.0, 10.0
    grid = make_grid(config)
    initial = torch.tensor(expected[prefix + "initial"])
    boundary = torch.tensor(expected[prefix + "boundary"])
    np.testing.assert_array_equal(grid.detach().numpy(), expected[prefix + "grid"])
    fields = torch.cat(model(grid), dim=1)
    np.testing.assert_allclose(
        fields.detach().numpy(), expected[prefix + "fields"], rtol=2e-6, atol=1e-8
    )
    fields.sum().backward()
    assert grid.grad is not None
    np.testing.assert_allclose(
        grid.grad.numpy(), expected[prefix + "input_gradient"], rtol=2e-6, atol=1e-8
    )
    model.zero_grad()
    grid.grad = None
    objective = loss(model, grid, initial.clone(), boundary.clone())
    objective.backward(retain_graph=True)
    gradients = torch.cat(
        [
            p.grad.flatten() if p.grad is not None else torch.zeros_like(p).flatten()
            for p in model.parameters()
        ]
    )
    np.testing.assert_allclose(objective.item(), expected[prefix + "loss"], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(
        gradients.detach().numpy(), expected[prefix + "gradient"], rtol=2e-6, atol=1e-8
    )
    np.testing.assert_allclose(
        remesh(config, model, grid).detach().numpy(),
        expected[prefix + "remeshed"],
        rtol=2e-6,
        atol=1e-8,
    )
    optimizer = make_optimizer(model, config)
    losses = []
    for _ in range(3):
        optimizer.zero_grad()
        objective = loss(model, grid, initial.clone(), boundary.clone())
        grid = remesh(config, model, grid)
        objective.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(objective.item())
    np.testing.assert_allclose(losses, expected[prefix + "three_losses"], rtol=2e-6, atol=1e-8)
    np.testing.assert_allclose(
        torch.cat([p.detach().flatten() for p in model.parameters()]).numpy(),
        expected[prefix + "after_steps"],
        rtol=2e-6,
        atol=1e-8,
    )


def test_complete_expanded_paper_catalogue():
    cases = catalogue()
    assert len(cases) == 249
    assert len({case["source_experiment"] for case in cases.values()}) == 81
    assert all(
        case["config"]["training_process"]["parameters"]["optimizer"] == "SOAP"
        for case in cases.values()
    )


def test_smoke_train_reload_and_refuse_overwrite(tmp_path):
    from modpinn.training import train, evaluate

    output = tmp_path / "run"
    result = train("best/amplitude-0.1", output, smoke=True)
    assert result["epochs_completed"] == 3
    assert len(json.loads((output / "history.json").read_text())) == 3
    assert result["parameter_change_l2"] > 0
    assert result["adaptive_updates"] == 3
    metrics = evaluate(output / "model.pt")
    assert metrics["points"] > 0
    assert all(
        np.isfinite(metrics[key])
        for key in ("relative_l2_phi", "relative_l2_alpha", "relative_l2_Q")
    )
    before = (output / "model.pt").read_bytes()
    with pytest.raises(FileExistsError):
        train("best/amplitude-0.1", output, smoke=True)
    assert (output / "model.pt").read_bytes() == before
    for name in ("config.json", "history.json", "metrics.json"):
        text = (output / name).read_text()
        assert str(ROOT) not in text and str(tmp_path) not in text


def test_cli_works_from_unrelated_directory(tmp_path):
    import subprocess
    import os

    result = subprocess.run(
        [sys.executable, "-m", "modpinn.training", "list", "--family", "best"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert len(result.stdout.strip().splitlines()) == 4


def test_invalid_configuration_cannot_execute_code(tmp_path):
    from modpinn.training import validate_config

    config = copy.deepcopy(next(iter(catalogue().values()))["config"])
    config["physical"]["parameters"]["N_t"] = "__import__('os').system('echo unsafe')"
    with pytest.raises(ValueError):
        validate_config(config)
    config = copy.deepcopy(next(iter(catalogue().values()))["config"])
    config["training_process"]["DTYPE"] = '__import__("os")'
    with pytest.raises(ValueError):
        validate_config(config)


def test_all_bundled_pretrained_models_load_and_evaluate():
    from modpinn.training import load_pretrained, evaluate_pretrained

    for key, record in catalogue().items():
        if "pretrained" not in record:
            continue
        model = load_pretrained(key)
        fields = model(torch.tensor([[0.1, 0.2], [1.0, 0.8]], requires_grad=True))
        assert len(fields) == 3 and all(torch.isfinite(field).all() for field in fields)
        metrics = evaluate_pretrained(key)
        assert metrics["points"] > 1000
        assert np.isfinite(metrics["relative_l2_sum"])
        expected = json.loads((ROOT / "tests/fixtures/pretrained-errors.json").read_text())
        archived = expected["cases"][key]["archived_errors"]
        np.testing.assert_allclose(
            [metrics["relative_l2_" + field] for field in ("phi", "alpha", "Q", "sum")],
            [archived[field] for field in ("l2_φ", "l2_α", "l2_Q", "l2")],
            rtol=2e-5,
            atol=1e-7,
        )


def test_public_audit_detects_plain_and_compressed_private_paths(tmp_path):
    import gzip
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_audit", ROOT / "src/verify.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    private = "/" + "Users" + "/example-person/project"
    (tmp_path / "note.txt").write_text(private)
    (tmp_path / "nested.json.gz").write_bytes(gzip.compress(json.dumps({"path": private}).encode()))
    report = module.audit(tmp_path)
    assert {item["path"] for item in report["findings"]} == {"note.txt", "nested.json.gz"}


def test_public_audit_and_data_contract():
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_audit", ROOT / "src/verify.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.audit(ROOT)["findings"] == []
    result = module.verify_data(ROOT)
    assert result["errors"] == []
    assert result["references"] == 16
    assert result["experiments"] == 249


def test_every_expanded_case_constructs_with_finite_derivatives():
    from modpinn.training import validate_config

    torch.set_num_threads(1)
    for case in catalogue().values():
        config = validate_config(case["config"])
        torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
        model = ModPINN(config=config)
        points = torch.tensor([[0.1, 0.2], [1.0, 0.8]], requires_grad=True)
        values = torch.cat(model(points), dim=1)
        assert torch.isfinite(values).all()
        values.sum().backward()
        assert points.grad is not None and torch.isfinite(points.grad).all()


def test_ablation_after_step_branch_runs(tmp_path):
    from modpinn.training import train

    result = train("ablation/modpinn", tmp_path / "ablation", smoke=True)
    assert result["update_order"] == "after_step"
    assert result["adaptive_updates"] == 3
    assert result["parameter_change_l2"] > 0


def test_installed_data_directory_takes_precedence(tmp_path, monkeypatch):
    import modpinn.training as training

    package = tmp_path / "installed" / "modpinn"
    (package / "data").mkdir(parents=True)
    (package / "data" / "experiments.json").write_text("{}")
    monkeypatch.setattr(training, "__file__", str(package / "training.py"))
    assert training.data_root() == package / "data"


def test_nonfinite_predictions_cannot_report_zero_error():
    from modpinn.training import reference_metrics

    class InvalidModel:
        DTYPE = torch.float32
        device = "cpu"

        def __call__(self, x):
            return (torch.full((len(x), 1), float("nan")),) * 3

    reference = {
        key: np.ones(2) for key in ("t_truth", "r_truth", "phi_truth", "alpha_truth", "Q_truth")
    }
    with pytest.raises(FloatingPointError, match="prediction"):
        reference_metrics(InvalidModel(), reference)


@pytest.mark.parametrize(
    "malformation", ["unaligned", "empty", "nonfinite_coordinates", "condition_mismatch"]
)
def test_custom_reference_rejects_malformed_arrays(tmp_path, malformation):
    import h5py
    from modpinn.training import _read_reference

    keys = (
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
    arrays = {key: np.ones(4) for key in keys}
    if malformation == "unaligned":
        arrays["phi_truth"] = np.ones(5)
    elif malformation == "empty":
        arrays["phi_0"] = np.ones(0)
    elif malformation == "condition_mismatch":
        arrays["alpha_b"] = np.ones(3)
    else:
        arrays["t_truth"][0] = float("nan")
    path = tmp_path / "bad.h5"
    with h5py.File(path, "w") as handle:
        for key, values in arrays.items():
            handle.create_dataset(key, data=values)
    with pytest.raises(ValueError):
        _read_reference(path)


def test_api_rejects_bad_log_interval_before_creating_output(tmp_path):
    from modpinn.training import train

    with pytest.raises(ValueError, match="log_every"):
        train("best/amplitude-0.1", tmp_path / "run", smoke=True, log_every=0)
    assert not (tmp_path / "run").exists()


def test_generated_runs_stay_in_data_layout(tmp_path, monkeypatch):
    import modpinn.training as training

    cases = training.catalogue()
    reference = training._reference_path(cases["best/amplitude-0.1"]["config"])
    bundled = tmp_path / "datasets"
    bundled.mkdir()
    monkeypatch.setattr(training, "data_root", lambda: bundled)
    monkeypatch.setattr(training, "catalogue", lambda: cases)
    output = bundled / "runs" / "example"
    result = training.train("best/amplitude-0.1", output, smoke=True, reference=reference)
    assert result["epochs_completed"] == 3
    with pytest.raises(ValueError, match="bundled"):
        training.train("best/amplitude-0.1", bundled / "references" / "new", smoke=True)


def test_delivered_figure_numbers_and_vector_format():
    import xml.etree.ElementTree as ET
    from modpinn.training import evaluate_pretrained

    directory = ROOT / "datasets/figures"
    report = json.loads((directory / "evaluation.json").read_text())
    for case, stored in report["cases"].items():
        actual = evaluate_pretrained(case)
        for key in ("relative_l2_phi", "relative_l2_alpha", "relative_l2_Q", "relative_l2_sum"):
            np.testing.assert_allclose(actual[key], stored[key], rtol=2e-5, atol=1e-7)
    for path in directory.glob("*.svg"):
        svg = ET.parse(path).getroot()
        assert not svg.findall(".//{http://www.w3.org/2000/svg}image")
        background = svg.find('.//{http://www.w3.org/2000/svg}g[@id="patch_1"]')
        assert background is not None
        assert all("fill: none" in child.attrib.get("style", "") for child in background)
    assert {p.name for p in directory.glob("*.svg")} == {
        "profiles-light.svg",
        "profiles-dark.svg",
        "spacetime-light.svg",
        "spacetime-dark.svg",
    }


def test_minimal_tutorial_runs_as_written(tmp_path, monkeypatch):
    import re

    text = (ROOT / "docs/quickstart.md").read_text()
    snippet = re.findall(r"```python\n(.*?)```", text, re.S)[0]
    monkeypatch.chdir(tmp_path)
    namespace = {}
    exec(compile(snippet, "quickstart-example", "exec"), namespace)
    result, metrics = namespace["result"], namespace["metrics"]
    assert result["epochs_completed"] == result["adaptive_updates"] == 3
    np.testing.assert_allclose(result["parameter_change_l2"], 0.0058383867, rtol=2e-5)
    np.testing.assert_allclose(result["relative_l2_sum"], 8.4549951299, rtol=2e-5)
    for key in ("phi", "alpha", "Q", "sum"):
        assert metrics["relative_l2_" + key] == result["relative_l2_" + key]
    history = json.loads((tmp_path / "datasets/runs/python-example/history.json").read_text())
    assert len(history) == 3
    assert all(np.isfinite(item["loss"]) for item in history)


def test_spacetime_fields_use_reference_coordinates_and_raw_predictions():
    from figures import collect
    from modpinn.training import load_pretrained, _read_reference, _reference_path

    records, profiles, spacetime = collect()
    assert len(records) == len(profiles) == len(spacetime) == 4
    for (case, record), (grid, truth, prediction) in zip(records.items(), spacetime):
        reference = _read_reference(_reference_path(catalogue()[case]["config"]))
        expected_coordinates = np.column_stack((reference["t_truth"], reference["r_truth"]))
        np.testing.assert_array_equal(grid.reshape(-1, 2), expected_coordinates)
        expected_fields = np.column_stack([reference[f + "_truth"] for f in ("phi", "alpha", "Q")])
        np.testing.assert_array_equal(truth.reshape(-1, 3), expected_fields)
        assert list(grid.shape[:2]) == record["spacetime_shape"]
        assert np.isnan(truth).sum(axis=(0, 1)).tolist() == record["reference_nan_counts"]
        indices = [0, len(expected_coordinates) // 2, len(expected_coordinates) - 1]
        model = load_pretrained(case)
        with torch.no_grad():
            direct = torch.cat(
                model(torch.tensor(expected_coordinates[indices], dtype=model.DTYPE)), dim=1
            )
        np.testing.assert_allclose(
            prediction.reshape(-1, 3)[indices], direct.numpy(), rtol=2e-5, atol=1e-6
        )
