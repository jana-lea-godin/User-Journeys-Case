from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class ClassifiedEvents:
    events: pd.DataFrame
    items: pd.DataFrame


class ContentClassifier:
    """
    Adds interpretable content/product labels to event logs.
    Uses items.category + optional heuristics from items.name.
    """

    DEFAULT_MAP = {
        "apparel": "Apparel",
        "drinkware": "Drinkware",
        "bags": "Bags",
        "electronics accessories": "Electronics_Accessories",
        "lifestyle": "Lifestyle",
        "small goods": "Small_Goods",
        "campus collection": "Campus_Collection",
        "new": "New_Arrivals",
        "clearance": "Clearance",
        "shop by brand": "Shop_By_Brand",
        "black lives matter": "Cause_Collection",
        "uncategorized items": "Uncategorized",
    }

    def __init__(self, category_map: dict[str, str] | None = None):
        self.category_map = category_map or self.DEFAULT_MAP

    def add_labels(self, events: pd.DataFrame, items: pd.DataFrame) -> ClassifiedEvents:
        ev = events.copy()
        it = items.copy()

        # Normalize category key
        it["category_key"] = (
            it["category"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        it["content_type"] = it["category_key"].map(self.category_map).fillna("Other")

        # Price buckets (useful for segmentation)
        it["price_bucket"] = pd.cut(
            it["price_in_usd"],
            bins=[-0.01, 10, 25, 50, 100, float("inf")],
            labels=["<=10", "10-25", "25-50", "50-100", "100+"],
        ).astype("string")

        # Join onto events
        ev = ev.merge(
            it[["id", "content_type", "price_in_usd", "price_bucket", "brand", "variant", "category"]],
            left_on="item_id",
            right_on="id",
            how="left",
            validate="m:1",
        ).drop(columns=["id"])

        # If some items are missing in items table
        ev["content_type"] = ev["content_type"].fillna("Unknown_Item")
        ev["price_bucket"] = ev["price_bucket"].fillna("Unknown_Price")

        return ClassifiedEvents(events=ev, items=it)