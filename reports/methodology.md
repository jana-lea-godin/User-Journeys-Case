# Methodology

## Objective
Build a portfolio-grade, reproducible analytics pipeline that answers:

1) Which user groups behave similarly in the conversion funnel?  
2) Which segments are stable over time?  
3) Where does the funnel leak per segment?

## Data Model
### Inputs (raw)
- `users.csv`
- `items.csv`
- `events1.csv`

### Processing layers
- `data/raw/` – immutable source files
- `data/interim/` – labeled + sessionized parquet outputs
- `data/processed/` – final feature tables for modeling/segmentation

## Preprocessing
- type casting
- integrity checks (unknown users/items)
- deduplication (exact duplicates and structured key duplicates)

## Sessionization
Session key:
- `(user_id, ga_session_id)`

Session aggregates:
- duration
- number of events
- unique items
- binary flags: `has_add_to_cart`, `has_purchase`

## Content Labeling
Items are mapped into a small set of content/product categories.
Event rows inherit `content_type` via `item_id`.

## Feature Engineering
User features combine:
- volume (n_sessions, n_events)
- intensity (events/session, unique_items/session)
- conversion rates (session-based and event-based)
- engagement timing (active days, first/last seen)
- content distribution + entropy

## Segmentation
Algorithm:
- KMeans on standardized numerical features

Model selection:
- evaluate k in a defined range
- compute silhouette and inertia
- pick business-sensible k (tradeoff: stability + interpretability)

Outputs:
- user assignments
- segment profiles (mean + median)

## Conversion Efficiency
We compute conversion metrics from session-facts:
- `atc_rate`
- `purchase_rate`
- `conversion_efficiency = purchase_rate / atc_rate` (clipped to [0,1])
- leakage = `1 - efficiency`

This ensures:
- interpretable, bounded metrics
- robustness against feature aggregation artifacts

## Reproducibility
- deterministic seeds
- artifacts written to `results/`
- no notebooks required; pipeline is runnable via `python -m src.case`