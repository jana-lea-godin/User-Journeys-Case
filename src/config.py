from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Paths:
    data_raw: Path = PROJECT_ROOT / "data" / "raw"
    data_interim: Path = PROJECT_ROOT / "data" / "interim"
    data_processed: Path = PROJECT_ROOT / "data" / "processed"

    results_figures: Path = PROJECT_ROOT / "results" / "figures"
    results_tables: Path = PROJECT_ROOT / "results" / "tables"
    results_segments: Path = PROJECT_ROOT / "results" / "segments"
    results_drift: Path = PROJECT_ROOT / "results" / "drift_reports"

    reports: Path = PROJECT_ROOT / "reports"