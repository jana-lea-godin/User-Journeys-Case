from __future__ import annotations

from .config import Paths
from .data_loader import DataLoader
from .content_classifier import ContentClassifier
from .session_builder import SessionBuilder
from .evaluation import DataQualityChecker
from .feature_store import FeatureStore
from .segmenter import Segmenter
from .conversion_analysis import ConversionAnalyzer
import pandas as pd

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
    # Data Quality Check
    dq = DataQualityChecker()
    dq_report = dq.check_events(data.events, data.items, data.users)

    dq_report.summary.to_csv(
        paths.results_tables / "data_quality_summary.csv",
        index=False,
    )

    for name, df in dq_report.details.items():
        df.to_csv(
            paths.results_tables / f"data_quality_{name}.csv",
            index=False,
        )

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


    # -------------------------
    # Feature Store
    # -------------------------
    fs = FeatureStore()
    user_features = fs.build_user_features(
        events=out.events,
        sessions=out.sessions,
        users=data.users,
    )

    user_features.user_features.to_parquet(
        paths.data_processed / "user_features.parquet",
        index=False,
    )

    user_features.user_features.head(50).to_csv(
        paths.results_tables / "user_features_head.csv",
        index=False,
    )



    # -------------------------
    # Segmentation
    # -------------------------
    uf = user_features.user_features.copy()

    share_cols = [c for c in uf.columns if c.startswith("share_content_")]
    base_cols = [
        "n_sessions",
        "n_events",
        "active_days",
        "purchase_rate_sessions",
        "add_to_cart_rate_sessions",
        "mean_session_duration",
        "mean_events_per_session",
        "mean_unique_items_per_session",
        "content_entropy",
    ]

    feature_cols = [c for c in base_cols if c in uf.columns] + share_cols

    seg = Segmenter(
        k_range=range(3, 8),   # 3..7  -> für 7er Story
        outlier_threshold=0.0,
    )

    seg_res = seg.segment_users(
        uf,
        feature_columns=feature_cols,
        min_sessions=1,
    )

    # Save segment outputs
    seg_res.assignments.to_csv(paths.results_segments / "user_segments.csv", index=False)
    seg_res.profile.to_csv(paths.results_segments / "segment_profile.csv", index=False)
    seg_res.model_info.to_csv(paths.results_tables / "k_selection_metrics.csv", index=False)

        # --------------------
    # Conversion micro-journey analysis (session-based, robust)
    # --------------------
    sessions_path = paths.data_interim / "sessions.parquet"
    segments_path = paths.results_segments / "user_segments.csv"

    if sessions_path.exists() and segments_path.exists():
        sessions_df = pd.read_parquet(paths.data_interim / "sessions.parquet")
        assignments = pd.read_csv(paths.results_segments / "user_segments.csv")

        conv = ConversionAnalyzer()
        conv_art = conv.run(
            sessions=sessions_df,
            assignments=assignments,
            results_tables=paths.results_tables,
            results_figures=paths.results_figures,
        )

        conv_art.table.head(50).to_csv(paths.results_tables / "segment_conversion_head.csv", index=False)

        print(f"✅ Conversion analysis saved: {conv_art.table_path}")

        if conv_art.conversion_fig_path:
            print(f"✅ Conversion figure saved: {conv_art.conversion_fig_path}")
        else:
            print("⚠️ matplotlib not available; skipped conversion efficiency figure")

        if conv_art.quadrant_fig_path:
            print(f"✅ Quadrant figure saved: {conv_art.quadrant_fig_path}")
        else:
            print("⚠️ matplotlib not available; skipped quadrant figure")

        if conv_art.leakage_size_fig_path:
            print(f"✅ Leakage vs size figure saved: {conv_art.leakage_size_fig_path}")
        else:
            print("⚠️ matplotlib not available; skipped leakage vs size figure")
                

        print(f"✅ Segmentation complete (chosen_k={seg_res.chosen_k})")


    print("✅ Pipeline run complete")
    print(f"Interim: {paths.data_interim}")
    print(f"Tables: {paths.results_tables}")


if __name__ == "__main__":
    run_pipeline()