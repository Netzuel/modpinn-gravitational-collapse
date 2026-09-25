"""Audit the public source payload and verify bundled numerical identities.

This scans the deliverable, not Git history, installed dependencies, or private
build/test caches. It reports locations and categories without disclosing values.
"""

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import zipfile

import h5py
import numpy as np

EXCLUDED = {
    ".git",
    ".cleanup",
    ".venv",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    "build",
    "dist",
    "runs",
}
PATTERNS = {
    "private-home-path": re.compile(
        rb"(?:/(?:Users|home)/[^\s\"\'<>]+|[A-Za-z]:\\(?:Users|Documents and Settings)\\)"
    ),
    "email-address": re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "personal-hostname": re.compile(
        rb"(?i)\b(?:[a-z0-9_-]+[-_])?(?:MacBook|DESKTOP|LAPTOP)[-_][a-z0-9_-]+"
    ),
    "private-key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access-token": re.compile(
        rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|AKIA[0-9A-Z]{16}|sk-[A-Za-z0-9_-]{40,})\b"
    ),
}


def public_files(root):
    for directory, folders, names in os.walk(root):
        folders[:] = sorted(
            name for name in folders if name not in EXCLUDED and not name.endswith(".egg-info")
        )
        for name in sorted(names):
            yield Path(directory) / name


def audit(root):
    root = Path(root).resolve()
    findings = []
    counts = {"files": 0, "hdf5": 0, "zip_containers": 0, "gzip": 0, "notebooks": 0}

    def scan(data, path):
        for label, pattern in PATTERNS.items():
            count = len(pattern.findall(data))
            if count:
                findings.append({"path": path, "category": label, "count": count})

    for path in public_files(root):
        relative = path.relative_to(root).as_posix()
        counts["files"] += 1
        if path.is_symlink():
            findings.append({"path": relative, "category": "symlink-not-portable", "count": 1})
            continue
        if (
            (
                path.suffix.lower() in {".tex", ".bib", ".cls", ".bst", ".pdf", ".eps"}
                and relative != "src/diagrams/modpinn.tex"
            )
            or "zenodo" in path.parts
            or path.name.startswith(".env")
            or path.name == ".DS_Store"
        ):
            findings.append({"path": relative, "category": "excluded-public-artifact", "count": 1})
        try:
            if path.suffix == ".h5":
                counts["hdf5"] += 1
                with h5py.File(path, "r") as handle:

                    def inspect(name, obj):
                        scan(name.encode(), relative)
                        for key, value in obj.attrs.items():
                            scan((str(key) + str(value)).encode(), relative)
                        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "OSU":
                            scan(str(np.asarray(obj)).encode(), relative)

                    inspect("/", handle)
                    handle.visititems(inspect)
            elif path.suffix == ".gz":
                counts["gzip"] += 1
                scan(gzip.decompress(path.read_bytes()), relative)
            elif zipfile.is_zipfile(path):
                counts["zip_containers"] += 1
                with zipfile.ZipFile(path) as archive:
                    for member in archive.infolist():
                        scan(member.filename.encode(), relative)
                        # Parse metadata without executing checkpoint pickle contents.
                        if not re.search(r"/(?:data)/\d+$", member.filename):
                            scan(archive.read(member), relative)
            else:
                data = path.read_bytes()
                scan(data, relative)
                if path.suffix == ".ipynb":
                    counts["notebooks"] += 1
                    notebook = json.loads(data)
                    if any(
                        cell.get("outputs") or cell.get("execution_count") is not None
                        for cell in notebook["cells"]
                    ):
                        findings.append(
                            {"path": relative, "category": "saved-notebook-output", "count": 1}
                        )
        except Exception as exc:
            findings.append(
                {"path": relative, "category": "unreadable-" + type(exc).__name__, "count": 1}
            )
    return {
        "scope": "public source payload; excludes Git history and local build/test caches",
        "counts": counts,
        "findings": findings,
    }


def verify_data(root):
    root = Path(root)
    directory = root / "datasets"
    errors = []
    manifest = json.loads((directory / "checksums.json").read_text())
    for name, expected in manifest.items():
        path = directory / name
        if not path.is_file():
            errors.append(name + ": missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(name + ": checksum mismatch")
    cases = json.loads((directory / "experiments.json").read_text())["experiments"]
    references = set()
    shapes = {}
    for key, case in cases.items():
        config = case["config"]
        name = config["training_process"]["import"]["data_path"]
        references.add(name)
        if name not in shapes:
            with h5py.File(directory / name, "r") as h:
                shapes[name] = (np.asarray(h["phi_0"]).size, np.asarray(h["phi_b"]).size)
        expected = (
            int(config["physical"]["parameters"]["N_x"]),
            int(config["physical"]["parameters"]["N_t"]),
        )
        if shapes[name] != expected:
            errors.append(key + ": initial/boundary shape mismatch")
        if case["update_order"] not in ("before_step", "after_step"):
            errors.append(key + ": unknown update order")
        if case["config"]["training_process"]["parameters"]["optimizer"] != "SOAP":
            errors.append(key + ": wrong optimizer")
        if "pretrained" in case and case["pretrained"] not in manifest:
            errors.append(key + ": unrecorded state")
    if len(cases) != 249 or len({case["source_experiment"] for case in cases.values()}) != 81:
        errors.append("Incomplete paper catalogue")
    return {
        "artifacts": len(manifest),
        "references": len(references),
        "experiments": len(cases),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = {"privacy": audit(args.root), "integrity": verify_data(args.root)}
    print(json.dumps(report, indent=2))
    return int(bool(report["privacy"]["findings"] or report["integrity"]["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
