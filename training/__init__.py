from .evaluation import EvaluationSuite
from .trajectory import Trajectory, TrajectoryDataset

__all__ = [
    "EvaluationSuite",
    "PaperBenchmarkResult",
    "Trajectory",
    "TrajectoryDataset",
    "run_paper_benchmark",
    "select_literature_scenario",
]


def __getattr__(name: str):
    if name in {
        "PaperBenchmarkResult",
        "run_paper_benchmark",
        "select_literature_scenario",
    }:
        from .literature_benchmark import (
            PaperBenchmarkResult,
            run_paper_benchmark,
            select_literature_scenario,
        )

        exports = {
            "PaperBenchmarkResult": PaperBenchmarkResult,
            "run_paper_benchmark": run_paper_benchmark,
            "select_literature_scenario": select_literature_scenario,
        }
        return exports[name]
    raise AttributeError(f"module 'training' has no attribute {name!r}")
