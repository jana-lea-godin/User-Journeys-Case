# Analysis Walkthrough

## Scope & Dataset Reality Check
This project analyzes **lower-funnel ecommerce behavior** from the Google Merchandise Store logs.
The raw event table contains primarily:
- `add_to_cart`
- `purchase`
and derived session-level constructs enable micro-journey insights.

There are no browsing/pageview events, therefore the focus is:
> **Conversion micro-journeys, funnel leakage, and segment-specific behavior**  
> (instead of full clickstream pathing).

---

## Data Quality
Key checks are exported to `results/tables/`:
- `data_quality_summary.csv`
- `data_quality_event_nulls.csv`

Notable:
- duplicate events were detected and removed during preprocessing (`Deduplicated events: removed 39,498 rows`).

---

## Feature Engineering
User-level features are built from:
- session aggregates (duration, events/session, sessions/user)
- conversion rates (`has_add_to_cart`, `has_purchase`)
- content affinity shares (`share_content_*`)
- entropy to quantify content diversity (`content_entropy`)

Output:
- `data/processed/user_features.parquet`
- sample: `results/tables/user_features_head.csv`

---

## Segmentation
Clustering method:
- KMeans + StandardScaler
- k selection via silhouette metrics (exported)

Artifacts:
- `results/tables/k_selection_metrics.csv`
- `results/segments/user_segments.csv`
- `results/segments/segment_profile.csv`

The chosen solution:
- **k = 7 clusters**
- segments are interpretable and stable enough for business usage

---

## Conversion Micro-Journey Analysis
We compute segment conversion on **session-facts** (robust and bounded):
- `atc_rate` = share of sessions with add-to-cart
- `purchase_rate` = share of sessions with purchase
- `conversion_efficiency` = purchase_rate / atc_rate (clipped to [0,1])
- `conversion_leakage` = 1 - conversion_efficiency

Output:
- table: `results/tables/segment_conversion_analysis.csv`
- figure: `results/figures/segment_conversion_efficiency.png`

### Key patterns
- Some segments have **very high ATC rates** but low purchase rates → strong leakage.
- At least one segment shows **purchase without recorded ATC** (purchase_rate high while atc_rate low),
  indicating incomplete event coverage for upstream steps or behavior bypassing the tracked ATC event.

See:
- `segment_conversion_analysis.csv`
- `segment_conversion_efficiency.png`

---

## What this enables next
- segment-specific intervention strategies (checkout UX, discount targeting)
- stability monitoring over time (traffic mix drift)
- content-type response modeling per segment