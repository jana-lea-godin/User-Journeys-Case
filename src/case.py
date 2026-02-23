from __future__ import annotations

from .config import Paths
from .data_loader import DataLoader
from .content_classifier import ContentClassifier
from .session_builder import SessionBuilder


def ensure_dirs() -> Paths:
    paths = Paths()
    for p in [
        paths.data_interim,
        paths.data_processed,
        paths.results_figures,
        paths.results_tables,
        paths.results_segments,
        paths.results_drift,
    ]:
        p.mkdir(parents=True, exist_ok=True)
    return paths


def run_pipeline() -> None:
    paths = ensure_dirs()

    data = DataLoader(paths.data_raw).load_all()

    classifier = ContentClassifier()
    classified = classifier.add_labels(data.events, data.items)

    sessionizer = SessionBuilder()
    out = sessionizer.build(classified.events)

    # Save interim artifacts
    out.events.to_parquet(paths.data_interim / "events_labeled_sessionized.parquet", index=False)
    out.sessions.to_parquet(paths.data_interim / "sessions.parquet", index=False)

    # quick sanity tables
    out.sessions.head(50).to_csv(paths.results_tables / "sessions_head.csv", index=False)
    (
        out.events["content_type"]
        .value_counts(dropna=False)
        .head(30)
        .to_frame("n_events")
        .to_csv(paths.results_tables / "content_type_event_counts.csv")
    )

    print("✅ Pipeline run complete")
    print(f"Interim: {paths.data_interim}")
    print(f"Tables: {paths.results_tables}")


if __name__ == "__main__":
    run_pipeline()