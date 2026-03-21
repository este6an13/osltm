# CSV Outputs Reference

> Every analysis script generates CSV files alongside its plots so that conclusions can be drawn and analyses reconstructed without relying on images.

---

## Naming Convention

```
{analysis_type}_{parameter_values}.csv
```

Parameters are encoded directly in the filename so different runs produce distinct files. Common parameters:

| Token | Values | Example |
|-------|--------|---------|
| `count_type` | `checkins`, `checkouts` | `fano_factor_checkins.csv` |
| `day_type` | `WD`, `SA`, `SU`, `HO` | `clustering_shape_WD_checkins_4clusters.csv` |
| `envelope_type` | `std`, `quantile` | `mean_envelope_checkins_quantile.csv` |
| `n_clusters` | integer | `clustering_shape_scale_WD_checkouts_3clusters.csv` |

---

## Profile Analysis CSVs

### FPCA Across Stations

**Purpose:** Shows inter-station variability by projecting all (station, day) profiles into a shared principal component space.

| File | Columns | What It Captures |
|------|---------|------------------|
| `fpca_across_stations_{count_type}.csv` | `station_code`, `station_name`, `date_type`, `date_str`, `PC1`, `PC2`, ... | PC scores for every profile — enables measuring spread/overlap between stations in PC space |
| `fpca_across_stations_variance_{count_type}.csv` | `component`, `explained_variance_ratio`, `cumulative_explained_variance` | How much variance each PC captures |

**Conclusions it supports:** Quantify inter-station separation by computing cluster distances or overlap in PC space. Identify which stations behave similarly.

---

### FPCA Per Station

**Purpose:** Tests whether day types (WD/SA/SU/HO) separate in PC space within each station.

| File | Columns | What It Captures |
|------|---------|------------------|
| `fpca_per_station_{count_type}.csv` | `station_code`, `station_name`, `date_type`, `date_str`, `PC1`, `PC2`, ... | Combined scores for all stations |
| `fpca_per_station_variance_{count_type}.csv` | `station_code`, `PC1_explained_variance`, `PC2_explained_variance`, ... | Explained variance per PC per station |

**Conclusions it supports:** Measure within-station day-type separation by computing distances between day-type clusters in PC space.

---

### Heatmap Station Profiles

**Purpose:** Mean profiles matrix (stations × time bins) for a fixed day type — quick cross-station comparison.

| File | Columns | What It Captures |
|------|---------|------------------|
| `heatmap_profiles_{day_type}_{count_type}.csv` | `station_code`, `station_name`, `t_400`, `t_415`, ..., `t_2245` | Mean count per 15-min bin per station |

**Conclusions it supports:** Identify peak hours, compare intensity levels, and spot stations with unusual temporal profiles.

---

### Mean Envelope Plots (per station)

**Purpose:** Typical pattern + day-to-day variability for each station, broken down by day type.

| File | Columns | What It Captures |
|------|---------|------------------|
| `mean_envelope_{count_type}_{envelope_type}.csv` | `station_code`, `station_name`, `day_type`, `time_bin_index`, `time_bin`, `mean`, `lower`, `upper`, `n_days` | Mean, lower, and upper envelope bounds per time bin |

**Conclusions it supports:** Quantify variability (envelope width) per station and day type. Compare regularity of patterns across day types.

---

### Mean Envelope Stations (cross-station)

**Purpose:** Cross-station comparison within each day type — overlays multiple stations' envelopes.

| File | Columns | What It Captures |
|------|---------|------------------|
| `mean_envelope_stations_{count_type}_{envelope_type}.csv` | `day_type`, `station_code`, `time_bin_index`, `time_bin`, `mean`, `lower`, `upper`, `n_days` | Same as above but organized by day type first |

**Conclusions it supports:** Compare station profiles side-by-side within each day type. Identify stations with similar/different temporal patterns.

---

### Clustering — Shape Based

**Purpose:** Groups stations with similar profile *shapes* (normalized to proportions), regardless of scale.

| File | Columns | What It Captures |
|------|---------|------------------|
| `clustering_shape_{day_type}_{count_type}_{n}clusters.csv` | `station_code`, `station_name`, `cluster`, `cluster_size`, `PC1_explained_variance`, ... | Cluster assignment per station + FPCA explained variance |

**Conclusions it supports:** Identify groups of stations sharing similar temporal shapes. Assess cluster balance.

---

### Clustering — Shape + Scale

**Purpose:** Groups stations similar in both shape *and* volume (log-transformed profiles).

| File | Columns | What It Captures |
|------|---------|------------------|
| `clustering_shape_scale_{day_type}_{count_type}_{n}clusters.csv` | `station_code`, `station_name`, `cluster`, `cluster_size` | Cluster assignment per station |

**Conclusions it supports:** Compare to shape-only clustering to see how volume information changes groupings.

---

### Clustering Label Alignment

**Purpose:** Tests whether unsupervised clusters match the WD/SA/SU/HO day-type taxonomy.

| File | Columns | What It Captures |
|------|---------|------------------|
| `clustering_summary_{n}clusters_{count_type}.csv` | `station_code`, `station_name`, `ari`, `purity`, `entropy`, confusion matrix columns | Adjusted Rand Index, purity, and entropy per station |

**Conclusions it supports:** Quantify how well day-type labels align with data-driven clusters. ARI close to 1 = strong alignment.

---

### Within–Between Distances

**Purpose:** Ratio R = mean(within-group dist) / mean(between-group dist). R < 1 → good separation of day types.

| File | Columns | What It Captures |
|------|---------|------------------|
| `distance_ratios_{count_type}.csv` | `station_code`, `station_name`, `ratio`, `within_mean`, `between_mean`, `within_std`, `between_std`, `within_median`, `between_median`, `n_within`, `n_between`, `n_days`, `interpretation` | Summary statistics + qualitative interpretation (strong/moderate/weak/poor) |
| `distance_details_{count_type}.csv` | `station_code`, `station_name`, `group`, `mean`, `std`, `count` | Per day-type pair breakdown (e.g., `WD_within`, `WD_vs_SA`) |

**Conclusions it supports:** Determine which stations show clear day-type separation and which pairs of day types are hardest to distinguish.

---

## Intensity Analysis CSVs

### Fano Factor (across days)

**Purpose:** Fano = Var(N)/E(N) across days per time bin. Poisson → Fano ≈ 1. Fano > 1 → overdispersion.

| File | Columns | What It Captures |
|------|---------|------------------|
| `fano_factor_{count_type}.csv` | `station_code`, `day_type`, `time_bin`, `fano_factor` | Raw Fano factor per station × day type × time bin |
| `fano_factor_median_{count_type}.csv` | `day_type`, `time_bin`, `median_fano_factor` | Median Fano across stations per time bin |

**Conclusions it supports:** Identify time-of-day and day-type patterns in overdispersion. Determine whether a Poisson assumption holds.

---

### Fano Factor (within bins)

**Purpose:** Fano factor within each 15-min bin using sub-bin subdivision. Tests Poisson at finer temporal resolution.

| File | Columns | What It Captures |
|------|---------|------------------|
| `fano_factor_within_bins_{count_type}.csv` | `day_type`, `time_bin`, `median`, `lower`, `upper` | Median ± envelope of Fano factor across stations |

**Conclusions it supports:** Assess whether overdispersion persists at sub-bin resolution.

---

### Time Rescaling QQ Plots

**Purpose:** If the estimated intensity is correct, rescaled inter-event times → Uniform(0,1). KS test validates this.

| File | Columns | What It Captures |
|------|---------|------------------|
| `time_rescaling_ks_stats_checkins.csv` | `station_code`, `station_name`, `day_type`, `n_events`, `ks_statistic`, `ks_pvalue` | Kolmogorov–Smirnov test results per station and day type |

**Conclusions it supports:** Identify which stations/day types pass or fail the uniformity test (p-value significance), validating the intensity model.
