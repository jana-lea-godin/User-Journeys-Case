# Executive Summary

## What we built
A fully reproducible Python pipeline (no notebooks) that:
- cleans and sessionizes ecommerce events
- creates user-level behavioral features
- segments users into interpretable clusters (k=7)
- quantifies funnel leakage and conversion efficiency per segment
- exports tables + figures for GitHub-ready reporting

## Key insight: Funnel leakage is segment-specific
Segment conversion efficiency (ATC → Purchase) varies strongly:
- some segments show near-universal add-to-cart but low purchase → checkout friction / price sensitivity
- at least one segment shows purchase without tracked add-to-cart → event coverage gaps or alternative paths

## Output artifacts
- Segment profile: `results/segments/segment_profile.csv`
- Conversion analysis: `results/tables/segment_conversion_analysis.csv`
- Conversion plot: `results/figures/segment_conversion_efficiency.png`

## Business implication
Different segments require different actions:
- High leakage → checkout UX + discount experiments
- High efficiency → upsell/cross-sell + retention
- “Purchase without ATC” → instrumentation audit + improved funnel tracking

Additional visuals:
- `results/figures/segment_atc_vs_purchase_quadrant.png` (quadrant: ATC vs Purchase)
- `results/figures/segment_leakage_vs_size.png` (leakage vs segment size)