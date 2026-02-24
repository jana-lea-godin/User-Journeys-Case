from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class SegmentationResult:
    assignments: pd.DataFrame          # user_id, segment_id
    profile: pd.DataFrame              # segment-level summary
    model_info: pd.DataFrame           # k, silhouette, inertia
    feature_columns: List[str]
    chosen_k: int
    outlier_threshold: float


class Segmenter:
    """
    KMeans segmentation with:
      - StandardScaler
      - k selection: choose the k with the highest silhouette score within k_range
      - optional outlier policy: clusters below outlier_threshold share => segment_id = -1
      - optional renumbering to contiguous segment IDs
    """

    def __init__(
        self,
        k_range: range = range(3, 9),
        random_state: int = 42,
        max_iter: int = 300,
        n_init: int = 20,
        outlier_threshold: float = 0.0,  # set >0 to enable outlier cluster policy
    ):
        self.k_range = k_range
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_init = n_init
        self.outlier_threshold = outlier_threshold

    def segment_users(
        self,
        user_features: pd.DataFrame,
        feature_columns: List[str],
        id_col: str = "user_id",
        min_sessions: int = 1,
        renumber_segments: bool = True,
    ) -> SegmentationResult:
        df = user_features.copy()

        # Low-activity filter (default min_sessions=1 for ecommerce realism)
        eligible = df[df["n_sessions"] >= min_sessions].copy()
        ineligible = df[df["n_sessions"] < min_sessions].copy()

        X = (
            eligible[feature_columns]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # 1) Fit models for each k
        rows = []
        models: Dict[int, KMeans] = {}

        for k in self.k_range:
            model = KMeans(
                n_clusters=int(k),
                random_state=self.random_state,
                n_init=self.n_init,
                max_iter=self.max_iter,
            )
            labels = model.fit_predict(Xs)

            sil = float(silhouette_score(Xs, labels))
            rows.append({"k": int(k), "silhouette": sil, "inertia": float(model.inertia_)})
            models[int(k)] = model

        model_info = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)

        # 2) Choose k: maximize silhouette within k_range
        chosen_k = int(model_info.sort_values("silhouette", ascending=False).iloc[0]["k"])
        chosen_model = models[chosen_k]

        # 3) Assign labels for eligible
        eligible_labels = chosen_model.predict(Xs).astype(int)

        eligible_assignments = eligible[[id_col]].copy()
        eligible_assignments["segment_id"] = eligible_labels

        # 4) Optional: mark tiny clusters as outliers
        if self.outlier_threshold and self.outlier_threshold > 0:
            counts = eligible_assignments["segment_id"].value_counts()
            total = len(eligible_assignments)
            share = (counts / total).to_dict()
            tiny_clusters = {seg for seg, s in share.items() if s < self.outlier_threshold}
            if tiny_clusters:
                eligible_assignments.loc[
                    eligible_assignments["segment_id"].isin(tiny_clusters),
                    "segment_id",
                ] = -1

        # 5) Append ineligible users as -1
        if len(ineligible) > 0:
            ineligible_assignments = ineligible[[id_col]].copy()
            ineligible_assignments["segment_id"] = -1
            assignments = pd.concat([eligible_assignments, ineligible_assignments], ignore_index=True)
        else:
            assignments = eligible_assignments

        # 6) Optional renumbering (keeps -1 as outlier/low-activity)
        if renumber_segments:
            assignments = self._renumber(assignments, id_col=id_col)

        # 7) Profiling (exclude -1)
        prof_df = df.merge(assignments, on=id_col, how="left")
        prof_df = prof_df[prof_df["segment_id"] >= 0].copy()

        profile_cols = self._profile_columns(feature_columns)

        profile = (
            prof_df.groupby("segment_id", as_index=False)[profile_cols]
            .agg(["mean", "median"])
        )
        profile.columns = ["_".join([c for c in col if c]) for col in profile.columns.to_flat_index()]
        profile = profile.reset_index(drop=True)

        seg_sizes = (
            prof_df["segment_id"]
            .value_counts()
            .rename_axis("segment_id")
            .reset_index(name="n_users")
        )
        profile = seg_sizes.merge(profile, on="segment_id", how="left").sort_values("segment_id")

        return SegmentationResult(
            assignments=assignments.sort_values(id_col).reset_index(drop=True),
            profile=profile.reset_index(drop=True),
            model_info=model_info,
            feature_columns=feature_columns,
            chosen_k=chosen_k,
            outlier_threshold=float(self.outlier_threshold),
        )

    @staticmethod
    def _renumber(assignments: pd.DataFrame, id_col: str) -> pd.DataFrame:
        df = assignments.copy()
        non_out = sorted([s for s in df["segment_id"].unique() if int(s) >= 0])
        mapping: Dict[int, int] = {old: new for new, old in enumerate(non_out)}

        df["segment_id"] = df["segment_id"].apply(lambda x: mapping.get(int(x), -1)).astype(int)
        return df

    @staticmethod
    def _profile_columns(feature_columns: List[str]) -> List[str]:
        core = [
            "n_sessions",
            "n_events",
            "purchase_rate_sessions",
            "add_to_cart_rate_sessions",
            "mean_session_duration",
            "active_days",
            "content_entropy",
        ]
        shares = [c for c in feature_columns if c.startswith("share_content_")]
        cols = []
        for c in core + shares:
            if c in feature_columns:
                cols.append(c)
        return cols