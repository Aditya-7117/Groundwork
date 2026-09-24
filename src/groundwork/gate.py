"""The regression gate: the golden slice re-run on every push, failing if retrieval got worse.

Two setups are checked, BM25 and MiniLM on CPU (decision 36), on passage recall@10 and nDCG@10.
Each has a committed baseline and a tolerance, and the gate fails when a metric falls below its
baseline by more than the tolerance. Each tolerance comes from measured noise, recorded beside it
in the baseline file: BM25 is exact arithmetic and scores identically on every run, while an
embedding model's floating-point results differ slightly between processors.

When a change is meant to move the numbers, the baseline is re-recorded with --record and the new
values are committed with the change, so every move of the baseline is visible in the history.
"""

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from groundwork.config import load_config
from groundwork.evaluation import load_built_corpus
from groundwork.experiment import ModelProvider, run_experiment

logger = logging.getLogger(__name__)

GATE_METRICS = ("passage.recall@10", "passage.ndcg@10")


class GateError(ValueError):
    """The baseline file is missing something the gate needs."""


@dataclass(frozen=True, slots=True, kw_only=True)
class Check:
    """One metric of one setup against its baseline."""

    setup: str
    metric: str
    baseline: float
    measured: float
    tolerance: float

    @property
    def passed(self) -> bool:
        """Whether the metric stayed within tolerance of the baseline, or improved."""
        return self.measured >= self.baseline - self.tolerance


def run_gate(baseline_path: Path, models: ModelProvider) -> tuple[Check, ...]:
    """Run every setup in the baseline file over its golden slice and compare.

    Raises:
        GateError: If the baseline file lacks a setup's config, metrics or tolerance.
    """
    baseline = _read(baseline_path)
    corpus = load_built_corpus(Path(str(baseline["corpus"])))
    checks = []
    for name, entry in _setups(baseline).items():
        config = load_config(Path(str(entry["config"])))
        result = run_experiment(config, corpus, models=models)
        metrics = entry.get("metrics")
        tolerance = entry.get("tolerance")
        if not isinstance(metrics, Mapping) or not isinstance(tolerance, int | float):
            raise GateError(f"{baseline_path}: {name} needs metrics and a tolerance")
        for metric in GATE_METRICS:
            checks.append(
                Check(
                    setup=name,
                    metric=metric,
                    baseline=float(metrics[metric]),
                    measured=result.aggregate[metric],
                    tolerance=float(tolerance),
                )
            )
    for check in checks:
        logger.log(
            logging.INFO if check.passed else logging.ERROR,
            "gate check",
            extra={
                "setup": check.setup,
                "metric": check.metric,
                "baseline": check.baseline,
                "measured": check.measured,
                "tolerance": check.tolerance,
                "passed": check.passed,
            },
        )
    return tuple(checks)


def record_baseline(baseline_path: Path, models: ModelProvider) -> dict[str, dict[str, float]]:
    """Re-run every setup and write its metrics into the baseline file, keeping the tolerances.

    Raises:
        GateError: If the baseline file lacks a setup's config.
    """
    baseline = _read(baseline_path)
    corpus = load_built_corpus(Path(str(baseline["corpus"])))
    recorded = {}
    for name, entry in _setups(baseline).items():
        config = load_config(Path(str(entry["config"])))
        result = run_experiment(config, corpus, models=models)
        recorded[name] = {metric: result.aggregate[metric] for metric in GATE_METRICS}
        entry["metrics"] = recorded[name]
    baseline_path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    return recorded


def _read(path: Path) -> dict[str, object]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise GateError(f"{path}: baseline file not found") from error
    if not isinstance(document, dict) or "corpus" not in document:
        raise GateError(f"{path}: expected an object naming the golden corpus")
    return document


def _setups(baseline: Mapping[str, object]) -> dict[str, dict[str, object]]:
    setups = baseline.get("setups")
    if not isinstance(setups, dict) or not setups:
        raise GateError("the baseline file names no setups")
    for name, entry in setups.items():
        if not isinstance(entry, dict) or "config" not in entry:
            raise GateError(f"setup {name} has no config")
    return setups
