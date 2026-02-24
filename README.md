# User Journey Intelligence — Google Merchandise Store

End-to-end analytics pipeline transforming raw ecommerce events into:

- Sessionized behavioral data
- User-level feature modeling
- Interpretable user segmentation (k-means)
- Conversion leakage diagnostics
- Stability validation (bootstrap + temporal drift)
- Reproducible executive-ready artifacts

Focus: lower-funnel behavior (`add_to_cart`, `begin_checkout`, `purchase`).

---

## Why this project matters

This project demonstrates the ability to:

- Engineer behavioral features from raw event logs
- Build modular, reproducible pipelines (no notebooks)
- Apply unsupervised learning with defensible model selection
- Quantify funnel leakage using bounded, interpretable metrics
- Validate segmentation stability
- Translate analytics into business impact

It bridges Data Engineering, Analytics Engineering, and Machine Learning.

---

## Key Insights

1️⃣ The largest segment (~5,300 users) shows 99% cart interaction but only 9% purchase → ~91% leakage after add-to-cart.

2️⃣ Two dominant segments (~9,700 users combined) exhibit structural checkout friction (high ATC, low purchase).

3️⃣ A premium segment (~1,100 users) shows >54% purchase rate → ideal for upsell & loyalty targeting.

4️⃣ Purchase without recorded add-to-cart exists → instrumentation gap / alternative purchase path.

5️⃣ Segments are behaviorally distinct (entropy, session depth, leakage structure) — not cosmetic clusters.

---

## Architecture Overview

The pipeline follows a layered data design:

RAW → INTERIM → PROCESSED → MODEL → ARTIFACTS



