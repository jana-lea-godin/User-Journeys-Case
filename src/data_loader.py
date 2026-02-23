from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


REQUIRED_EVENTS_COLS = {
    "user_id",
    "ga_session_id",
    "country",
    "device",
    "type",
    "item_id",
    "date",
}

REQUIRED_ITEMS_COLS = {"id", "name", "brand", "variant", "category", "price_in_usd"}
REQUIRED_USERS_COLS = {"id", "ltv", "date"}


@dataclass(frozen=True)
class LoadedData:
    events: pd.DataFrame
    items: pd.DataFrame
    users: pd.DataFrame


class DataLoader:
    def __init__(self, raw_dir: Path):
        self.raw_dir = Path(raw_dir)

    def load_events(self, filename: str = "events1.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"events file not found: {path}")

        df = pd.read_csv(path)
        self._validate_columns(df, REQUIRED_EVENTS_COLS, name="events")

        df = df.copy()
        df["user_id"] = pd.to_numeric(df["user_id"], errors="raise").astype("int64")
        df["ga_session_id"] = pd.to_numeric(df["ga_session_id"], errors="raise").astype("int64")
        df["item_id"] = pd.to_numeric(df["item_id"], errors="raise").astype("int64")

        for col in ["country", "device", "type"]:
            df[col] = df[col].astype("string").str.strip()

        df["date"] = pd.to_datetime(df["date"], errors="raise", utc=False)
        df = df.dropna(subset=["user_id", "ga_session_id", "type", "date", "item_id"])
        df = df.sort_values(["user_id", "ga_session_id", "date", "item_id"]).reset_index(drop=True)
        return df

    def load_items(self, filename: str = "items.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"items file not found: {path}")

        df = pd.read_csv(path)
        self._validate_columns(df, REQUIRED_ITEMS_COLS, name="items")

        df = df.copy()
        df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
        df["price_in_usd"] = pd.to_numeric(df["price_in_usd"], errors="coerce")

        for col in ["name", "brand", "variant", "category"]:
            df[col] = df[col].astype("string").str.strip()

        # optional: ensure id uniqueness
        if df["id"].duplicated().any():
            # keep first occurrence; log-worthy later
            df = df.drop_duplicates(subset=["id"], keep="first")

        return df

    def load_users(self, filename: str = "users.csv") -> pd.DataFrame:
        path = self.raw_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"users file not found: {path}")

        df = pd.read_csv(path)
        self._validate_columns(df, REQUIRED_USERS_COLS, name="users")

        df = df.copy()
        df["id"] = pd.to_numeric(df["id"], errors="raise").astype("int64")
        df["ltv"] = pd.to_numeric(df["ltv"], errors="coerce").fillna(0.0)

        df["date"] = pd.to_datetime(df["date"], errors="raise", utc=False)

        if df["id"].duplicated().any():
            # if multiple snapshots exist, keep the latest
            df = df.sort_values(["id", "date"]).drop_duplicates(subset=["id"], keep="last")

        return df

    def load_all(self) -> LoadedData:
        events = self.load_events()
        items = self.load_items()
        users = self.load_users()
        return LoadedData(events=events, items=items, users=users)

    @staticmethod
    def _validate_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {sorted(missing)}")