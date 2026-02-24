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
    conversion_fig_path: Path | None
    quadrant_fig_path: Path | None
    leakage_size_fig_path: Path | None


class ConversionAnalyzer:
    """
    Segment-level conversion analysis based on session facts.

    Produces:
      - segment_conversion_analysis.csv
      - segment_conversion_efficiency.png
      - segment_atc_vs_purchase_quadrant.png
      - segment_leakage_vs_size.png
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
        filename_eff_fig: str = "segment_conversion_efficiency.png",
        filename_quadrant_fig: str = "segment_atc_vs_purchase_quadrant.png",
        filename_leakage_size_fig: str = "segment_leakage_vs_size.png",
    ) -> ConversionArtifacts:
        results_tables.mkdir(parents=True, exist_ok=True)
        results_figures.mkdir(parents=True, exist_ok=True)


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


        df[self.has_atc_col] = df[self.has_atc_col].astype(bool)
        df[self.has_purchase_col] = df[self.has_purchase_col].astype(bool)

        # Segment aggregates (session basis)
        seg_sessions = (
            df.groupby(self.seg_col, as_index=False)
            .agg(
                n_sessions=(self.has_atc_col, "size"),
                atc_rate=(self.has_atc_col, "mean"),
                purchase_rate=(self.has_purchase_col, "mean"),
            )
        )

        # Users per segment (assignment basis)
        users_per_seg = (
            assignments[assignments[self.seg_col] >= 0]
            .groupby(self.seg_col)[self.id_col]
            .nunique()
            .rename("n_users")
            .reset_index()
        )

        out = users_per_seg.merge(seg_sessions, on=self.seg_col, how="left")
        

        denom = out["atc_rate"].replace(0, pd.NA)
        
        out["conversion_efficiency"] = (out["purchase_rate"] / denom).fillna(0.0)
        out["conversion_efficiency"] = out["conversion_efficiency"].clip(lower=0.0, upper=1.0)
        out["conversion_leakage"] = (1.0 - out["conversion_efficiency"]).clip(lower=0.0, upper=1.0)

        out = out.sort_values(["conversion_efficiency", "n_users"], ascending=[False, False]).reset_index(drop=True)

        table_path = results_tables / filename_table
        out.to_csv(table_path, index=False)

        eff_fig = None
        quad_fig = None
        leak_fig = None

        if plt is not None:
            eff_fig = results_figures / filename_eff_fig
            self._plot_efficiency(out, eff_fig)

            quad_fig = results_figures / filename_quadrant_fig
            self._plot_quadrant(out, quad_fig, results_tables)

            leak_fig = results_figures / filename_leakage_size_fig
            self._plot_leakage_vs_size(out, leak_fig, results_tables)

        return ConversionArtifacts(
            table=out,
            table_path=table_path,
            conversion_fig_path=eff_fig,
            quadrant_fig_path=quad_fig,
            leakage_size_fig_path=leak_fig,
        )

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

    @staticmethod
    def _plot_quadrant(df: pd.DataFrame, out_path: Path, results_tables: Path) -> None:
        """
        Scatter of ATC rate vs Purchase rate with median lines (quadrants).
        Saves underlying table for reproducibility.
        """
        quad_table = df[["segment_id", "n_users", "n_sessions", "atc_rate", "purchase_rate"]].copy()
        quad_table_path = results_tables / "segment_atc_purchase_table.csv"
        quad_table.to_csv(quad_table_path, index=False)

        x = quad_table["atc_rate"].astype(float)
        y = quad_table["purchase_rate"].astype(float)
        sizes = quad_table["n_users"].astype(float)

        x_med = float(x.median())
        y_med = float(y.median())

        plt.figure()
        plt.scatter(x, y, s=(sizes / sizes.max()) * 600 + 40)  # size scaling (no color choice)

        # label points
        for _, r in quad_table.iterrows():
            plt.text(float(r["atc_rate"]), float(r["purchase_rate"]), str(int(r["segment_id"])), fontsize=9)

        # quadrant lines
        plt.axvline(x_med, linewidth=1)
        plt.axhline(y_med, linewidth=1)

        plt.xlabel("ATC Rate (share of sessions with add_to_cart)")
        plt.ylabel("Purchase Rate (share of sessions with purchase)")
        plt.title("Segments by Funnel Position (ATC vs Purchase)")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    @staticmethod
    def _plot_leakage_vs_size(df: pd.DataFrame, out_path: Path, results_tables: Path) -> None:
        """
        Scatter of leakage vs segment size (n_users).
        Saves underlying table for reproducibility.
        """
        tab = df[["segment_id", "n_users", "conversion_leakage", "conversion_efficiency", "atc_rate", "purchase_rate"]].copy()
        tab_path = results_tables / "segment_leakage_size_table.csv"
        tab.to_csv(tab_path, index=False)

        x = tab["n_users"].astype(float)
        y = tab["conversion_leakage"].astype(float)

        plt.figure()
        plt.scatter(x, y, s=120)

        for _, r in tab.iterrows():
            plt.text(float(r["n_users"]), float(r["conversion_leakage"]), str(int(r["segment_id"])), fontsize=9)

        plt.xlabel("Segment Size (n_users)")
        plt.ylabel("Leakage (1 - efficiency)")
        plt.title("Leakage vs Segment Size")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()