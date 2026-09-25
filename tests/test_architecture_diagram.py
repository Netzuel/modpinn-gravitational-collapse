"""Validate the README diagram assets and the narrow TikZ-source exception."""

from pathlib import Path
import xml.etree.ElementTree as ET

from verify import audit

ROOT = Path(__file__).resolve().parents[1]


def test_architecture_themes_are_vector_and_linked():
    readme = (ROOT / "README.md").read_text()
    for theme in ("light", "dark"):
        path = Path(f"datasets/figures/architecture/modpinn-{theme}.svg")
        assert str(path) in readme
        svg = ET.parse(ROOT / path).getroot()
        assert svg.tag == "{http://www.w3.org/2000/svg}svg"
        assert svg.findall(".//{http://www.w3.org/2000/svg}path")
        assert not svg.findall(".//{http://www.w3.org/2000/svg}image")
        assert not svg.findall(".//{http://www.w3.org/2000/svg}script")
    directory = ROOT / "datasets/figures/architecture"
    assert (directory / "modpinn-light.svg").read_bytes() != (
        directory / "modpinn-dark.svg"
    ).read_bytes()


def test_audit_allows_diagram_source_but_rejects_manuscripts(tmp_path):
    source = tmp_path / "src/diagrams/modpinn.tex"
    source.parent.mkdir(parents=True)
    source.write_text((ROOT / "src/diagrams/modpinn.tex").read_text())
    assert not audit(tmp_path)["findings"]
    (tmp_path / "paper.tex").write_text("Manuscript placeholder")
    findings = audit(tmp_path)["findings"]
    assert any(
        finding["path"] == "paper.tex" and finding["category"] == "excluded-public-artifact"
        for finding in findings
    )
