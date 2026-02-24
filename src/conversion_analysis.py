from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


@dataclass(frozen=True)
class ConversionArtifacts:
    table: pd.DataFrame
    table_path: Path
    figure_path: Path | None


class ConversionAnalyzer:
    """
    Segment-level conversion micro-journey analysis based on *session facts*.

    Inputs:
      - sessions: session-level table containing has_add_to_cart, has_purchase
      - assignments: user_id -> segment_id
    """

    def __init__(
        self,
        id_col: str = "user_id",
        seg_col: str = "segment_id",
        has_atc_col: str = "has_add_to_cart",
        has_purchase_col: str = "has_purchase",
    ):
        self.id_col = id_col
        self.seg_col = seg_col
        self.has_atc_col = has_atc_col
        self.has_purchase_col = has_purchase_col

    def run(
        self,
        sessions: pd.DataFrame,
        assignments: pd.DataFrame,
        results_tables: Path,
        results_figures: Path,
        filename_table: str = "segment_conversion_analysis.csv",
        filename_fig: str = "segment_conversion_efficiency.png",
    ) -> ConversionArtifacts:
        results_tables.mkdir(parents=True, exist_ok=True)
        results_figures.mkdir(parents=True, exist_ok=True)

        # Validate columns
        req_sessions = {self.id_col, self.has_atc_col, self.has_purchase_col}
        req_assign = {self.id_col, self.seg_col}
        miss_sess = req_sessions - set(sessions.columns)
        miss_ass = req_assign - set(assignments.columns)
        if miss_sess:
            raise ValueError(f"sessions missing required columns: {sorted(miss_sess)}")
        if miss_ass:
            raise ValueError(f"assignments missing required columns: {sorted(miss_ass)}")

        # Merge session -> segment
        df = sessions[[self.id_col, self.has_atc_col, self.has_purchase_col]].merge(
            assignments[[self.id_col, self.seg_col]],
            on=self.id_col,
            how="left",
        )

        # Keep only clustered segments (>=0); treat -1 as outlier
        df = df[df[self.seg_col].notna()].copy()
        df[self.seg_col] = df[self.seg_col].astype(int)
        df = df[df[self.seg_col] >= 0].copy()

        # Ensure boolean-ish
        df[self.has_atc_col] = df[self.has_atc_col].astype(bool)
        df[self.has_purchase_col] = df[self.has_purchase_col].astype(bool)

        # Aggregate per segment
        out = (
            df.groupby(self.seg_col, as_index=False)
            .agg(
                n_sessions=(self.has_atc_col, "size"),
                atc_rate=(self.has_atc_col, "mean"),
                purchase_rate=(self.has_purchase_col, "mean"),
            )
        )

        # Users per segment
        users_per_seg = (
            assignments[assignments[self.seg_col] >= 0]
            .groupby(self.seg_col)[self.id_col]
            .nunique()
            .rename("n_users")
            .reset_index()
        )
        out = users_per_seg.merge(out, on=self.seg_col, how="left")

        # Efficiency and leakage bounded in [0,1]
        denom = out["atc_rate"].replace(0, pd.NA)
        out["conversion_efficiency"] = (out["purchase_rate"] / denom).fillna(0.0)
        out["conversion_efficiency"] = out["conversion_efficiency"].clip(lower=0.0, upper=1.0)
        out["conversion_leakage"] = (1.0 - out["conversion_efficiency"]).clip(lower=0.0, upper=1.0)

        out = out.sort_values(["conversion_efficiency", "n_users"], ascending=[False, False]).reset_index(drop=True)

        table_path = results_tables / filename_table
        out.to_csv(table_path, index=False)

        fig_path = None
        if plt is not None:
            fig_path = results_figures / filename_fig
            self._plot_efficiency(out, fig_path)

        return ConversionArtifacts(table=out, table_path=table_path, figure_path=fig_path)

    @staticmethod
    def _plot_efficiency(df: pd.DataFrame, out_path: Path) -> None:
        plt.figure()
        x = df["segment_id"].astype(str)
        y = df["conversion_efficiency"].astype(float)
        plt.bar(x, y)
        plt.xlabel("Segment ID")
        plt.ylabel("Conversion Efficiency (Purchase | ATC sessions)")
        plt.title("Segment Conversion Efficiency (ATC → Purchase)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()