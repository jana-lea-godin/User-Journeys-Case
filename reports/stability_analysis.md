# Stability Analysis

## Why stability matters
Segments are only useful if they are:
- consistent under sampling noise
- not overly sensitive to small shifts in data
- trackable across time windows

## Bootstrap stability
We estimate segmentation robustness via repeated bootstraps:
- sample users (fraction)
- re-fit clustering
- compare assignment similarity via ARI

Key metric:
- ARI (Adjusted Rand Index)
  - 1.0 = identical clustering
  - 0.0 = random agreement

Artifacts:
- written to `results/tables/` by StabilitySuite

Interpretation:
- ARI ~0.7 indicates strong stability for a behavioral segmentation
- some variance is expected due to sparse activity users

## Temporal drift
We track segment shares across monthly periods and measure drift:
- distribution shift across segment proportions
- L1 distance and PSI as summary metrics

Artifacts:
- `results/tables/temporal_segment_shares.csv`
- `results/tables/temporal_drift.csv`

Interpretation:
- drift suggests changes in traffic mix / campaign composition
- stable segments + drifting mix is normal in ecommerce seasonality

## Recommendation
Use segments operationally with:
- periodic drift monitoring (monthly)
- re-training thresholds if drift exceeds a business-defined tolerance