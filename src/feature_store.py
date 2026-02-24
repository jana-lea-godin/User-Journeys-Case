from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class UserFeatures:
    user_features: pd.DataFrame


class FeatureStore:

    def build_user_features(
        self,
        events: pd.DataFrame,
        sessions: pd.DataFrame,
        users: pd.DataFrame,
    ) -> UserFeatures:

        # -------------------------
        # SESSION-LEVEL AGGREGATION
        # -------------------------
        session_agg = (
            sessions.groupby("user_id", as_index=False)
            .agg(
                n_sessions=("session_key", "count"),
                mean_session_duration=("session_duration_sec", "mean"),
                purchase_rate_sessions=("has_purchase", "mean"),
                add_to_cart_rate_sessions=("has_add_to_cart", "mean"),
                mean_events_per_session=("n_events", "mean"),
                mean_unique_items_per_session=("n_unique_items", "mean"),
            )
        )

        # -------------------------
        # EVENT-LEVEL AGGREGATION
        # -------------------------
        event_agg = (
            events.groupby("user_id", as_index=False)
            .agg(
                n_events=("type", "count"),
                purchase_rate_events=("type", lambda s: (s == "purchase").mean()),
                active_days=("date", lambda s: s.dt.date.nunique()),
                first_seen=("date", "min"),
                last_seen=("date", "max"),
            )
        )

        # -------------------------
        # CONTENT MIX
        # -------------------------
        content_counts = (
            events.groupby(["user_id", "content_type"])
            .size()
            .unstack(fill_value=0)
        )

        content_shares = content_counts.div(content_counts.sum(axis=1), axis=0)

        content_shares.columns = [
            f"share_content_{c.lower()}" for c in content_shares.columns
        ]

        content_shares = content_shares.reset_index()

        # -------------------------
        # CONTENT ENTROPY
        # -------------------------
        def entropy(row):
            probs = row[row > 0]
            return -(probs * np.log(probs)).sum()

        content_entropy = content_shares.set_index("user_id")
        content_entropy["content_entropy"] = content_entropy.apply(entropy, axis=1)
        content_entropy = content_entropy[["content_entropy"]].reset_index()

        # -------------------------
        # MERGE EVERYTHING
        # -------------------------
        user_features = (
            session_agg
            .merge(event_agg, on="user_id", how="inner")
            .merge(content_shares, on="user_id", how="left")
            .merge(content_entropy, on="user_id", how="left")
            .merge(users[["id", "ltv"]], left_on="user_id", right_on="id", how="left")
            .drop(columns=["id"])
        )

        user_features = user_features.fillna(0)

        return UserFeatures(user_features=user_features)