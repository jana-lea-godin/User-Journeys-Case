# Behavioral Segmentation Analysis

## Methodology

We applied KMeans clustering on standardized user-level behavioral features:

### Feature Categories

- Session metrics
- Event frequency
- Conversion ratios
- Content distribution shares
- Behavioral entropy

### Preprocessing

- StandardScaler normalization
- Removal of infinite values
- NaN handling via zero-imputation
- Low-activity filtering (min_sessions ≥ 1)

---

## Model Selection

We evaluated k ∈ [3, 8] using:

- Silhouette score
- Inertia
- Business interpretability

Final selection logic:

Choose the smallest k within tolerance of the best silhouette score.

Final model:

**k = 7**

---

## Segment Size Distribution

| Segment | Users |
|--------|-------|
| 0 | 2,061 |
| 1 | 5,368 |
| 2 | 3 |
| 3 | 1,433 |
| 4 | 4,352 |
| 5 | 1,101 |
| 6 | 383 |

---

## Behavioral Profiles

### High Engagement Cluster (Segment 5)
- Highest event depth
- Highest session duration
- High purchase rate
- High entropy

Represents core revenue drivers.

---

### Conversion-Heavy Low-Engagement (Segment 0)
- Few interactions
- Very high conversion
- Efficient buyers

Represents performance scaling opportunity.

---

### Cart-Dominant Segment (Segment 1)
- Near-universal add-to-cart
- Low purchase completion
- Funnel leakage potential

High ROI optimization target.

---

### Research-Oriented Segment (Segment 3)
- High browsing intensity
- Moderate purchase rate
- Long decision cycles

Content & UX leverage segment.

---

### Broad Explorers (Segment 4)
- High category diversity
- Moderate engagement
- Medium conversion

Personalization opportunity.

---

### Category Specialists (Segment 6)
- Dominant bag category affinity
- Niche targeting potential

---

## Observations

- Revenue is likely concentrated in Segments 5 and 0.
- Largest optimization opportunity lies in Segment 1.
- Behavioral entropy strongly correlates with engagement depth.
- Micro segment (Segment 2) negligible in business terms.

---

## Next Steps

- Add revenue & LTV overlay
- Develop segment-based campaign strategies
- Implement automated CRM routing by segment
- Deploy lookalike audiences for high-value clusters