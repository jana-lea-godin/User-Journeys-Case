from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass(frozen=True)
class DataQualityReport:
    summary: pd.DataFrame
    details: Dict[str, pd.DataFrame]


class DataQualityChecker:
    """
    Data quality checks for the event log + referential integrity to items/users.

    Produces:
      - summary: one-row-per-metric table
      - details: named tables for drill-down (nulls, missing joins, etc.)

    Notes:
      - We normalize the null-report to include the column name explicitly,
        because otherwise CSV export loses the index and you only see counts.
      - We compute both:
          * duplicate_event_rows (full-row duplicates)
          * duplicate_events_by_key (key-based duplicates on a meaningful subset)
    """

    def check_events(
        self,
        events: pd.DataFrame,
        items: pd.DataFrame,
        users: pd.DataFrame,
    ) -> DataQualityReport:
        report_rows = []
        details: Dict[str, pd.DataFrame] = {}

        # --------------------
        # Basic shape
        # --------------------
        report_rows.append({"metric": "n_events", "value": len(events)})

        # --------------------
        # Duplicate events (full-row duplicates)
        # --------------------
        dup_events_full_row = int(events.duplicated().sum())
        report_rows.append({"metric": "duplicate_event_rows", "value": dup_events_full_row})

        # --------------------
        # Duplicate events (key-based)
        # --------------------
        dup_events_by_key = int(
            events.duplicated(subset=["user_id", "ga_session_id", "item_id", "date", "type"]).sum()
        )
        report_rows.append({"metric": "duplicate_events_by_key", "value": dup_events_by_key})

        # --------------------
        # Null checks
        # --------------------
        null_counts = events.isnull().sum()
        null_series = null_counts[null_counts > 0]

        null_df = (
            null_series.rename_axis("column")
            .reset_index(name="null_count")
            .sort_values("null_count", ascending=False)
            .reset_index(drop=True)
        )

        report_rows.append({"metric": "columns_with_nulls", "value": len(null_df)})
        details["event_nulls"] = null_df

        # --------------------
        # Item integrity
        # --------------------
        missing_items = events[~events["item_id"].isin(items["id"])]
        report_rows.append({"metric": "events_with_unknown_item", "value": int(len(missing_items))})
        details["missing_items"] = missing_items.head(50).reset_index(drop=True)

        # --------------------
        # User integrity
        # --------------------
        missing_users = events[~events["user_id"].isin(users["id"])]
        report_rows.append({"metric": "events_with_unknown_user", "value": int(len(missing_users))})
        details["missing_users"] = missing_users.head(50).reset_index(drop=True)

        # --------------------
        # Timestamp sanity
        # --------------------
        if "date" in events.columns:
            min_date = events["date"].min()
            max_date = events["date"].max()
            report_rows.append({"metric": "min_event_date", "value": str(min_date)})
            report_rows.append({"metric": "max_event_date", "value": str(max_date)})

        # --------------------
        # Negative durations (if sessionized and present at event level)
        # --------------------
        if "session_duration_sec" in events.columns:
            negative_duration = int((events["session_duration_sec"] < 0).sum())
            report_rows.append({"metric": "negative_session_duration", "value": negative_duration})

        # Build summary *after* all metrics were appended
        summary = pd.DataFrame(report_rows)

        return DataQualityReport(summary=summary, details=details)