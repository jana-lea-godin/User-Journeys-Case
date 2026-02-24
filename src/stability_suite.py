from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StabilityArtifacts:
    temporal_segment_shares: pd.DataFrame
    temporal_drift: pd.DataFrame
    bootstrap_stability: pd.DataFrame
    stability_summary: pd.DataFrame


class StabilitySuite:
    """
    Segment stability evaluation (portfolio-grade).

    1) Temporal stability:
       - segment share per period (month/week)
       - drift metrics between consecutive periods

    2) Bootstrap stability:
       - refit KMeans on bootstrap samples
       - compare assignments via Adjusted Rand Index (ARI)
       - silhouette dispersion

    Inputs expected:
      - events: needs user_id, date
      - assignments: user_id, segment_id
      - user_features: user_id + feature columns used for segmentation
    """

    def __init__(
        self,
        period: str = "M",  # "M" monthly, "W" weekly
        bootstrap_runs: int = 20,
        bootstrap_sample_frac: float = 0.8,
        random_state: int = 42,
    ):
        self.period = period
        self.bootstrap_runs = int(bootstrap_runs)
        self.bootstrap_sample_frac = float(bootstrap_sample_frac)
        self.random_state = int(random_state)

    # ---------- Public API ----------

    def run(
        self,
        events: pd.DataFrame,
        assignments: pd.DataFrame,
        user_features: pd.DataFrame,
        feature_columns: List[str],
        chosen_k: int,
        results_tables: Optional[Path] = None,
    ) -> StabilityArtifacts:
        # Validate
        if not {"user_id", "date"}.issubset(events.columns):
            raise ValueError("events must contain: user_id, date")
        if not {"user_id", "segment_id"}.issubset(assignments.columns):
            raise ValueError("assignments must contain: user_id, segment_id")
        if "user_id" not in user_features.columns:
            raise ValueError("user_features must contain: user_id")

        ev = events.copy()
        ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
        ev = ev.dropna(subset=["date"]).copy()

        asg = assignments.copy()
        asg["segment_id"] = asg["segment_id"].astype(int)

        # ---- temporal
        temporal_segment_shares = self._temporal_segment_shares(ev, asg)
        temporal_drift = self._temporal_drift(temporal_segment_shares)

        # ---- bootstrap stability
        bootstrap_stability = self._bootstrap_stability(
            user_features=user_features,
            assignments=asg,
            feature_columns=feature_columns,
            chosen_k=int(chosen_k),
        )

        # ---- summary
        stability_summary = self._summary(temporal_drift, bootstrap_stability)

        artifacts = StabilityArtifacts(
            temporal_segment_shares=temporal_segment_shares,
            temporal_drift=temporal_drift,
            bootstrap_stability=bootstrap_stability,
            stability_summary=stability_summary,
        )

        if results_tables:
            results_tables.mkdir(parents=True, exist_ok=True)
            temporal_segment_shares.to_csv(results_tables / "stability_temporal_segment_shares.csv", index=False)
            temporal_drift.to_csv(results_tables / "stability_temporal_drift.csv", index=False)
            bootstrap_stability.to_csv(results_tables / "stability_bootstrap.csv", index=False)
            stability_summary.to_csv(results_tables / "stability_summary.csv", index=False)

        return artifacts

    # ---------- Temporal Stability ----------

    def _temporal_segment_shares(self, events: pd.DataFrame, assignments: pd.DataFrame) -> pd.DataFrame:
        # define "active users per period" from events
        df = events[["user_id", "date"]].copy()
        df["period"] = df["date"].dt.to_period(self.period).dt.to_timestamp()

        # user active in period
        active = df.drop_duplicates(["period", "user_id"])
        active = active.merge(assignments[["user_id", "segment_id"]], on="user_id", how="left")

        # drop unknown segment ids if any
        active = active.dropna(subset=["segment_id"]).copy()
        active["segment_id"] = active["segment_id"].astype(int)

        counts = (
            active.groupby(["period", "segment_id"], as_index=False)
            .size()
            .rename(columns={"size": "n_users_active"})
        )

        totals = (
            active.groupby("period", as_index=False)
            .size()
            .rename(columns={"size": "n_users_active_total"})
        )

        out = counts.merge(totals, on="period", how="left")
        out["share"] = out["n_users_active"] / out["n_users_active_total"]
        out = out.sort_values(["period", "segment_id"]).reset_index(drop=True)
        return out

    def _temporal_drift(self, shares: pd.DataFrame) -> pd.DataFrame:
        """
        Drift metrics between consecutive periods:
          - L1 distance (sum |p_t - p_{t-1}|)
          - PSI-like (Population Stability Index style) over segment shares
        """
        if shares.empty:
            return pd.DataFrame(columns=["period", "prev_period", "l1_drift", "psi_drift"])

        pivot = shares.pivot_table(index="period", columns="segment_id", values="share", fill_value=0.0).sort_index()
        periods = list(pivot.index)

        rows = []
        eps = 1e-8

        for i in range(1, len(periods)):
            p_prev = pivot.loc[periods[i - 1]].values.astype(float)
            p_now = pivot.loc[periods[i]].values.astype(float)

            l1 = float(np.abs(p_now - p_prev).sum())

            # PSI: sum((now-prev) * ln(now/prev))
            psi = float(np.sum((p_now - p_prev) * np.log((p_now + eps) / (p_prev + eps))))

            rows.append(
                {
                    "period": periods[i],
                    "prev_period": periods[i - 1],
                    "l1_drift": l1,
                    "psi_drift": psi,
                }
            )

        return pd.DataFrame(rows)

    # ---------- Bootstrap Stability ----------

    def _bootstrap_stability(
        self,
        user_features: pd.DataFrame,
        assignments: pd.DataFrame,
        feature_columns: List[str],
        chosen_k: int,
    ) -> pd.DataFrame:
        """
        Refit KMeans on bootstrapped user samples and compare clustering vs baseline (ARI).
        Baseline labels taken from assignments.

        Returns rows per run:
          run, ari_vs_baseline, silhouette
        """
        uf = user_features.copy()
        uf = uf.merge(assignments[["user_id", "segment_id"]], on="user_id", how="left")
        uf = uf.dropna(subset=["segment_id"]).copy()
        uf["segment_id"] = uf["segment_id"].astype(int)

        X = (
            uf[feature_columns]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        y_base = uf["segment_id"].astype(int).values

        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        rng = np.random.default_rng(self.random_state)
        n = len(uf)
        sample_n = int(np.floor(self.bootstrap_sample_frac * n))

        rows = []
        for run in range(1, self.bootstrap_runs + 1):
            idx = rng.choice(n, size=sample_n, replace=True)
            Xb = Xs[idx]
            yb_base = y_base[idx]

            model = KMeans(
                n_clusters=int(chosen_k),
                random_state=int(self.random_state + run),
                n_init=20,
                max_iter=300,
            )
            yb_pred = model.fit_predict(Xb)

            ari = float(adjusted_rand_score(yb_base, yb_pred))

            # silhouette needs >1 cluster
            sil = float(silhouette_score(Xb, yb_pred)) if len(np.unique(yb_pred)) > 1 else np.nan

            rows.append({"run": run, "ari_vs_baseline": ari, "silhouette": sil})

        return pd.DataFrame(rows)

    # ---------- Summary ----------

    @staticmethod
    def _summary(temporal_drift: pd.DataFrame, bootstrap: pd.DataFrame) -> pd.DataFrame:
        rows = []

        if not temporal_drift.empty:
            rows.append({"metric": "temporal_l1_mean", "value": float(temporal_drift["l1_drift"].mean())})
            rows.append({"metric": "temporal_l1_max", "value": float(temporal_drift["l1_drift"].max())})
            rows.append({"metric": "temporal_psi_mean", "value": float(temporal_drift["psi_drift"].mean())})
            rows.append({"metric": "temporal_psi_max", "value": float(temporal_drift["psi_drift"].max())})
        else:
            rows.append({"metric": "temporal_l1_mean", "value": np.nan})
            rows.append({"metric": "temporal_l1_max", "value": np.nan})
            rows.append({"metric": "temporal_psi_mean", "value": np.nan})
            rows.append({"metric": "temporal_psi_max", "value": np.nan})

        if not bootstrap.empty:
            rows.append({"metric": "bootstrap_ari_mean", "value": float(bootstrap["ari_vs_baseline"].mean())})
            rows.append({"metric": "bootstrap_ari_min", "value": float(bootstrap["ari_vs_baseline"].min())})
            rows.append({"metric": "bootstrap_silhouette_mean", "value": float(bootstrap["silhouette"].mean())})
            rows.append({"metric": "bootstrap_silhouette_std", "value": float(bootstrap["silhouette"].std())})
        else:
            rows.append({"metric": "bootstrap_ari_mean", "value": np.nan})
            rows.append({"metric": "bootstrap_ari_min", "value": np.nan})
            rows.append({"metric": "bootstrap_silhouette_mean", "value": np.nan})
            rows.append({"metric": "bootstrap_silhouette_std", "value": np.nan})

        return pd.DataFrame(rows)