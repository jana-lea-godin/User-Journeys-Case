from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Sessionized:
    events: pd.DataFrame
    sessions: pd.DataFrame


class SessionBuilder:
    """
    Builds session identifiers and session-level aggregates from event logs.
    Uses (user_id, ga_session_id) as primary session key.
    """

    def build(self, events: pd.DataFrame) -> Sessionized:
        df = events.copy()

        # Stable session key
        df["session_key"] = df["user_id"].astype("string") + "_" + df["ga_session_id"].astype("string")

        sessions = (
            df.groupby(["user_id", "ga_session_id", "session_key"], as_index=False)
            .agg(
                session_start=("date", "min"),
                session_end=("date", "max"),
                n_events=("type", "size"),
                n_unique_items=("item_id", "nunique"),
                n_purchases=("type", lambda s: (s == "purchase").sum()),
                n_add_to_cart=("type", lambda s: (s == "add_to_cart").sum()),
                country=("country", "first"),
                device=("device", "first"),
            )
        )

        sessions["session_duration_sec"] = (
            sessions["session_end"] - sessions["session_start"]
        ).dt.total_seconds().clip(lower=0)

        sessions["has_purchase"] = sessions["n_purchases"] > 0
        sessions["has_add_to_cart"] = sessions["n_add_to_cart"] > 0

        sessions = sessions.sort_values(["user_id", "session_start"]).reset_index(drop=True)

        return Sessionized(events=df, sessions=sessions)