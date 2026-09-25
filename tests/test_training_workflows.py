"""Training coverage beyond the small-grid smoke override."""

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from modpinn.training import (
    _conditions,
    _read_reference,
    _reference_path,
    catalogue,
    evaluate,
    train,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]
NATIVE_CASES = [
    "best/amplitude-0.1",
    "best/amplitude-0.125",
    "ablation/modpinn",
    "benchmark/seed-0/amplitude-0.1",
    "scaling/log-offset--04.45",
    "causality/amplitude-0.1/grid-64/depth-10-width-75/epsilon-100.0",
    "remeshing/seed-0/amplitude-0.10625/lambda-0.0",
]


def test_every_catalogue_configuration_matches_its_conditions():
    for record in catalogue().values():
        config = validate_config(record["config"])
        config["training_process"]["device"] = "cpu"
        initial, boundary = _conditions(_read_reference(_reference_path(config)), config)
        physical = config["physical"]["parameters"]
        assert initial.shape == (int(physical["N_x"]), 3)
        assert boundary.shape == (int(physical["N_t"]), 3)
        assert torch.isfinite(initial[:-1]).all()
        assert torch.isfinite(boundary).all()


@pytest.mark.parametrize("case", NATIVE_CASES)
def test_original_grid_and_architecture_train_and_reload(case, tmp_path):
    torch.set_num_threads(1)
    output = tmp_path / "run"
    result = train(case, output, epochs=3, log_every=1)
    assert result["epochs_completed"] == 3
    assert result["smoke"] is False
    assert result["parameter_change_l2"] > 0
    assert result["update_order"] == catalogue()[case]["update_order"]
    assert evaluate(output / "model.pt")["relative_l2_sum"] == result["relative_l2_sum"]
    expected = copy.deepcopy(catalogue()[case]["config"])
    expected["training_process"]["device"] = "cpu"
    expected["training_process"]["parameters"]["epochs"] = 3
    assert json.loads((output / "config.json").read_text()) == expected
    history = json.loads((output / "history.json").read_text())
    assert len(history) == 3
    assert all(np.isfinite(item["loss"]) for item in history)
    update = expected["training_process"]["domain_updating"]
    assert result["adaptive_updates"] == sum(
        epoch >= update["update_from"] and epoch % update["update_each"] == 0 for epoch in range(3)
    )


def test_cli_custom_float64_training_with_portable_reference(tmp_path):
    config = copy.deepcopy(catalogue()["best/amplitude-0.1"]["config"])
    config["physical"]["parameters"].update(N_t="8", N_x="8")
    config["neural_parameters"]["general_parameters"].update(number_hidden=1, number_neurons=8)
    config["training_process"]["parameters"]["epochs"] = 3
    config["training_process"]["DTYPE"] = "torch.float64"
    config_path = tmp_path / "custom.json"
    config_path.write_text(json.dumps(config))
    reference = tmp_path / "input.h5"
    reference.write_bytes(_reference_path(config).read_bytes())
    output = tmp_path / "custom-run"
    args = [
        sys.executable,
        "-m",
        "modpinn.training",
        "train",
        "--config",
        str(config_path),
        "--reference",
        str(reference),
        "--output",
        str(output),
        "--resample-conditions",
        "--log-every",
        "1",
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1"}
    process = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90
    )
    assert process.returncode == 0, process.stderr
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["epochs_completed"] == 3 and metrics["parameter_change_l2"] > 0
    assert hashlib.sha256(reference.read_bytes()).hexdigest() == metrics["reference_sha256"]
    reference.unlink()
    assert evaluate(output / "model.pt")["relative_l2_sum"] == metrics["relative_l2_sum"]
    checkpoint = torch.load(output / "model.pt", weights_only=True)
    assert all(value.dtype == torch.float64 for value in checkpoint["state_dict"].values())
    before = (output / "model.pt").read_bytes()
    repeated = subprocess.run(
        args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30
    )
    assert repeated.returncode == 2
    assert "new output directory" in repeated.stderr
    assert (output / "model.pt").read_bytes() == before
