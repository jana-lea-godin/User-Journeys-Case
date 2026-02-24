from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd


# ---------- Data Structures ----------

@dataclass(frozen=True)
class JourneyArtifacts:
    journeys: pd.DataFrame
    transition_matrix: pd.DataFrame
    top_paths: pd.DataFrame
    funnel: pd.DataFrame
    segment_path_lift: pd.DataFrame


# ---------- Core ----------

class JourneyBuilder:
    """
    Builds user/session journeys from event logs and produces journey analytics artifacts:
      - session-level ordered journeys (sequence of event "steps")
      - transition matrix (Markov-style counts & probabilities)
      - top paths (most common sequences)
      - funnel (step conversion)
      - segment path lift (which segments over-index on which paths)

    Assumptions about events dataframe:
      - user_id
      - ga_session_id
      - date (datetime)
      - type (e.g. 'view', 'add_to_cart', 'purchase')
      - content_type (categorical label added by ContentClassifier)
      - session_key (optional; will be created if missing)
    """

    def __init__(
        self,
        step_mode: str = "event_type",  # "event_type" or "content_type" or "hybrid"
        max_steps: int = 30,
        min_session_events: int = 2,
        drop_repeats: bool = True,
    ):
        if step_mode not in {"event_type", "content_type", "hybrid"}:
            raise ValueError("step_mode must be one of: event_type, content_type, hybrid")

        self.step_mode = step_mode
        self.max_steps = int(max_steps)
        self.min_session_events = int(min_session_events)
        self.drop_repeats = bool(drop_repeats)

    # ---------- Public API ----------

    def build(
        self,
        events: pd.DataFrame,
        assignments: Optional[pd.DataFrame] = None,  # user_id, segment_id (optional)
        results_tables: Optional[Path] = None,
    ) -> JourneyArtifacts:
        df = events.copy()

        # Ensure basics
        required = {"user_id", "ga_session_id", "date", "type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"events missing required columns: {sorted(missing)}")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"]).copy()

        if "session_key" not in df.columns:
            df["session_key"] = df["user_id"].astype("string") + "_" + df["ga_session_id"].astype("string")

        # Step column
        df["step"] = self._make_step(df)

        # Build session journeys
        journeys = self._build_session_journeys(df)

        # Transition matrix
        transition_matrix = self._transition_matrix(journeys)

        # Top paths
        top_paths = self._top_paths(journeys)

        # Funnel
        funnel = self._funnel(journeys)

        # Segment lifts (optional)
        segment_path_lift = self._segment_path_lift(journeys, assignments)

        artifacts = JourneyArtifacts(
            journeys=journeys,
            transition_matrix=transition_matrix,
            top_paths=top_paths,
            funnel=funnel,
            segment_path_lift=segment_path_lift,
        )

        if results_tables:
            results_tables.mkdir(parents=True, exist_ok=True)
            self._write_tables(artifacts, results_tables)

        return artifacts

    # ---------- Internals ----------

    def _make_step(self, df: pd.DataFrame) -> pd.Series:
        if self.step_mode == "event_type":
            return df["type"].astype("string")

        if self.step_mode == "content_type":
            if "content_type" not in df.columns:
                raise ValueError("step_mode='content_type' requires events['content_type']")
            return df["content_type"].astype("string").fillna("unknown_content")

        # hybrid
        if "content_type" not in df.columns:
            raise ValueError("step_mode='hybrid' requires events['content_type']")
        return (df["type"].astype("string") + "::" + df["content_type"].astype("string").fillna("unknown_content"))

    def _build_session_journeys(self, df: pd.DataFrame) -> pd.DataFrame:
        # sort within session
        df = df.sort_values(["session_key", "date"]).copy()

        # group and build sequences
        rows: List[Dict] = []
        for (user_id, ga_session_id, session_key), g in df.groupby(["user_id", "ga_session_id", "session_key"], sort=False):
            steps = g["step"].astype("string").tolist()

            if self.drop_repeats:
                steps = self._collapse_consecutive_repeats(steps)

            if len(steps) < self.min_session_events:
                continue

            steps = steps[: self.max_steps]
            path = " > ".join(steps)

            rows.append(
                {
                    "user_id": int(user_id),
                    "ga_session_id": int(ga_session_id),
                    "session_key": str(session_key),
                    "session_start": g["date"].min(),
                    "session_end": g["date"].max(),
                    "n_events": int(len(g)),
                    "n_steps": int(len(steps)),
                    "path": path,
                    "steps": steps,  # keep as list for further analytics
                    "has_add_to_cart": bool((g["type"] == "add_to_cart").any()),
                    "has_purchase": bool((g["type"] == "purchase").any()),
                }
            )

        journeys = pd.DataFrame(rows)
        if journeys.empty:
            # return empty but well-typed frames
            return pd.DataFrame(
                columns=[
                    "user_id",
                    "ga_session_id",
                    "session_key",
                    "session_start",
                    "session_end",
                    "n_events",
                    "n_steps",
                    "path",
                    "steps",
                    "has_add_to_cart",
                    "has_purchase",
                ]
            )

        return journeys.sort_values(["session_start", "user_id"]).reset_index(drop=True)

    @staticmethod
    def _collapse_consecutive_repeats(steps: List[str]) -> List[str]:
        out = []
        prev = None
        for s in steps:
            if prev is None or s != prev:
                out.append(s)
            prev = s
        return out

    def _transition_matrix(self, journeys: pd.DataFrame) -> pd.DataFrame:
        if journeys.empty:
            return pd.DataFrame()

        pairs = []
        for steps in journeys["steps"]:
            for a, b in zip(steps[:-1], steps[1:]):
                pairs.append((a, b))

        if not pairs:
            return pd.DataFrame()

        trans = pd.DataFrame(pairs, columns=["from", "to"])
        counts = trans.groupby(["from", "to"], as_index=False).size().rename(columns={"size": "count"})

        # Probabilities per "from"
        counts["p"] = counts["count"] / counts.groupby("from")["count"].transform("sum")

        # wide matrix (probabilities)
        mat = counts.pivot_table(index="from", columns="to", values="p", fill_value=0.0)
        mat = mat.sort_index().sort_index(axis=1)

        return mat.reset_index()

    def _top_paths(self, journeys: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
        if journeys.empty:
            return pd.DataFrame(columns=["path", "n_sessions", "purchase_rate", "add_to_cart_rate", "avg_steps"])

        vc = journeys["path"].value_counts().head(top_n).rename_axis("path").reset_index(name="n_sessions")

        agg = (
            journeys.groupby("path", as_index=False)
            .agg(
                purchase_rate=("has_purchase", "mean"),
                add_to_cart_rate=("has_add_to_cart", "mean"),
                avg_steps=("n_steps", "mean"),
            )
        )

        out = vc.merge(agg, on="path", how="left").sort_values("n_sessions", ascending=False)
        return out.reset_index(drop=True)

    def _funnel(self, journeys: pd.DataFrame) -> pd.DataFrame:
        """
        Simple funnel on session level:
          sessions -> add_to_cart -> purchase
        """
        if journeys.empty:
            return pd.DataFrame(columns=["step", "n_sessions", "rate_from_start", "rate_from_prev"])

        n = len(journeys)
        n_cart = int(journeys["has_add_to_cart"].sum())
        n_purchase = int(journeys["has_purchase"].sum())

        rows = [
            {"step": "sessions", "n_sessions": n, "rate_from_start": 1.0, "rate_from_prev": 1.0},
            {"step": "add_to_cart", "n_sessions": n_cart, "rate_from_start": n_cart / n if n else 0.0, "rate_from_prev": n_cart / n if n else 0.0},
            {"step": "purchase", "n_sessions": n_purchase, "rate_from_start": n_purchase / n if n else 0.0, "rate_from_prev": n_purchase / n_cart if n_cart else 0.0},
        ]
        return pd.DataFrame(rows)

    def _segment_path_lift(self, journeys: pd.DataFrame, assignments: Optional[pd.DataFrame]) -> pd.DataFrame:
        """
        Measures which segments over-index on which paths.
        Lift = P(path | segment) / P(path overall)
        """
        if journeys.empty or assignments is None or assignments.empty:
            return pd.DataFrame(columns=["segment_id", "path", "n_sessions", "lift", "p_seg", "p_all"])

        if not {"user_id", "segment_id"}.issubset(assignments.columns):
            raise ValueError("assignments must have columns: user_id, segment_id")

        j = journeys.merge(assignments[["user_id", "segment_id"]], on="user_id", how="left")
        j = j.dropna(subset=["segment_id"]).copy()
        j["segment_id"] = j["segment_id"].astype(int)

        # overall path probs
        all_counts = j["path"].value_counts()
        p_all = (all_counts / all_counts.sum()).to_dict()

        rows = []
        for seg_id, g in j.groupby("segment_id"):
            seg_counts = g["path"].value_counts()
            total = seg_counts.sum()

            for path, c in seg_counts.head(50).items():  # cap for readability
                p_seg = c / total if total else 0.0
                p0 = p_all.get(path, 1e-12)
                lift = p_seg / p0 if p0 > 0 else np.nan
                rows.append(
                    {
                        "segment_id": int(seg_id),
                        "path": path,
                        "n_sessions": int(c),
                        "p_seg": float(p_seg),
                        "p_all": float(p0),
                        "lift": float(lift),
                    }
                )

        out = pd.DataFrame(rows)
        if out.empty:
            return out

        return out.sort_values(["segment_id", "lift"], ascending=[True, False]).reset_index(drop=True)

    @staticmethod
    def _write_tables(artifacts: JourneyArtifacts, out_dir: Path) -> None:
        # journeys table (don’t write raw list column; write a readable path summary)
        journeys_out = artifacts.journeys.copy()
        if "steps" in journeys_out.columns:
            journeys_out = journeys_out.drop(columns=["steps"])

        journeys_out.to_csv(out_dir / "journeys_sessions.csv", index=False)
        artifacts.top_paths.to_csv(out_dir / "journeys_top_paths.csv", index=False)
        artifacts.funnel.to_csv(out_dir / "journeys_funnel.csv", index=False)

        # transition matrix can be large; still useful as CSV
        artifacts.transition_matrix.to_csv(out_dir / "journeys_transition_matrix.csv", index=False)

        # segment lift
        artifacts.segment_path_lift.to_csv(out_dir / "journeys_segment_path_lift.csv", index=False)