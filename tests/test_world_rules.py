"""The pilot must not be able to see ground truth (SPEC section 2)."""

import ast
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1] / "kestrel"
GROUND_TRUTH_MODULES = {"kestrel.sim.oracle", "kestrel.sim.course", "kestrel.sim.gates"}
BLIND_PACKAGES = ["pilot", "perception"]


def _imported_modules(path: Path) -> set[str]:
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


def test_pilot_and_perception_never_import_ground_truth():
    offenders = []
    for package in BLIND_PACKAGES:
        for path in (PACKAGE / package).rglob("*.py"):
            leaked = _imported_modules(path) & GROUND_TRUTH_MODULES
            if leaked:
                offenders.append(f"{path.relative_to(PACKAGE)} imports {sorted(leaked)}")
    assert not offenders, offenders


def test_import_scanner_catches_both_import_forms(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("import kestrel.sim.oracle\nfrom kestrel.sim import course\n")
    assert {"kestrel.sim.oracle", "kestrel.sim.course"} <= _imported_modules(sample)
