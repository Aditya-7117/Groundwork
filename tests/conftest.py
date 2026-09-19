"""Shared test configuration and fixtures."""

import hashlib
import os
import zipfile
from pathlib import Path

import pytest
from hypothesis import settings

# CI runs a fixed set of generated examples so that a failure reproduces on the next run. Local
# runs keep Hypothesis's default random exploration, which finds new cases over time.
settings.register_profile("ci", derandomize=True, database=None, print_blob=True)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

TINY_BEIR = Path(__file__).parent / "fixtures" / "tiny-beir"


@pytest.fixture
def tiny_beir_archive(tmp_path: Path) -> tuple[Path, str]:
    """The fixture corpus zipped in the BEIR distribution layout, and the archive's SHA-256."""
    path = tmp_path / "tiny-beir.zip"
    with zipfile.ZipFile(path, "w") as archive:
        for file in sorted(TINY_BEIR.rglob("*")):
            if file.is_file():
                archive.write(file, Path("tiny-beir") / file.relative_to(TINY_BEIR))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()
