# OSLTM — Experiments Playbook

> Reproducible experiment commands covering the full pipeline, utilities, and all analysis scripts.
> Based on actual command history and the sampled station set.

---

## Your Sampled Stations (20)

From `src/workflow/data/sampled_stations.csv`:

| Code | Name | Notes |
|------|------|-------|
| **03000** | Portal Suba | ⭐ Primary test station (used in most experiments) |
| **07112** | Comuneros | Frequently paired with 03000 |
| **06000** | Portal El Dorado | Portal — high volume |
| **07000** | Portal Sur JFK | Portal — high volume |
| **05105** | Pradera – Plaza Central | Used in envelope comparisons |
| **09116** | Avenida 39 | Caracas corridor |
| 02303 | Calle 85 - GATO DUMAS | Northern corridor |
| 02104 | Calle 146 | Northern corridor |
| 02105 | Calle 142 | Northern corridor |
| 02205 | Calle 106 | Northern corridor |
| 04002 | Carrera 90 | Calle 80 corridor |
| 05100 | Banderas P. Central | Américas corridor |
| 07200 | Tygua-San José | Southern extension |
| 07105 | Movistar Arena | Events venue |
| 08002 | Biblioteca | Tunal area |
| 09100 | Calle 40 Sur | Caracas corridor |
| 09108 | Hospital | Caracas corridor |
| 09121 | Flores – Areandina | Caracas corridor |
| 10001 | Country Sur | 20 de Julio corridor |
| 40000 | Cable Portal Tunal | Cable car |

### Station Groups for Experiments

```bash
# Single station (quick test)
SOLO="03000"

# Core trio (portals + intermediate)
TRIO="03000 07112 06000"

# Core quad (most used multi-station group)
QUAD="03000 07112 06000 07000"

# Envelope comparison set (from history)
ENVELOPE="03000 05105 09116"

# All 20 sampled stations (omit --stations flag)
```

---

## Part 1 — Data Pipeline

### 1.1 Full Pipeline (fresh run)

```bash
# Run all 4 steps with current params
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps all
```

### 1.2 Individual Steps

```bash
# Step 1 only — re-sample dates
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1

# Steps 2–3 only — download + sample stations (skip date re-sampling)
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 2 3

# Step 4 only — re-populate DB (useful after schema changes)
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 4
```

---

## Part 2 — Utilities

### 2.1 Drop Unused Columns

Strips CSV files down to only the columns used by step4. Saves ~60–85% disk space.

```bash
# Process a specific downloaded file
uv run python -m src.workflow.utils.drop_unused_columns \
  --params src/workflow/params.json --files 20260105

# Process a date range
uv run python -m src.workflow.utils.drop_unused_columns \
  --params src/workflow/params.json --date-start 20240625 --date-end 20240630

# Process ALL downloaded files
uv run python -m src.workflow.utils.drop_unused_columns \
  --params src/workflow/params.json
```

### 2.2 Data Loader (quick DB check)

Loads all data from the database and prints shape — useful to verify step4 worked.

```bash
uv run python -m src.workflow.data_loader
```

---

## Part 3 — Profile Analysis Experiments

### 3.1 FPCA Per Station

Visualize how daily profiles separate in PC space, colored by day type.

```bash
# Quick: single station
uv run python -m src.workflow.scripts.profiles.fpca_per_station \
  --stations 03000 --n_components 3

# Multi-station: compare PC structure
uv run python -m src.workflow.scripts.profiles.fpca_per_station \
  --stations 03000 07112 06000 07000

# All stations (no --stations flag), checkins
uv run python -m src.workflow.scripts.profiles.fpca_per_station \
  --count_type checkins

# Checkouts comparison
uv run python -m src.workflow.scripts.profiles.fpca_per_station \
  --stations 03000 07112 06000 07000 --count_type checkouts
```

### 3.2 FPCA Across Stations

All (station, day) profiles projected together — see inter-station variability.

```bash
# Single station across day types
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000

# Trio — compare portals
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000 07112 06000

# Quad — standard comparison set
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000 07112 06000 07000

# Focus on weekdays only
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000 07112 06000 --day_types WD

# Weekdays vs Saturdays
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000 07112 06000 --day_types WD SA
```

### 3.3 Within vs Between Distances

Ratio R < 1 = good day-type separation. Higher R = less separation.

```bash
# Single station
uv run python -m src.workflow.scripts.profiles.within_between_distances \
  --stations 03000

# All stations (generates a summary CSV)
uv run python -m src.workflow.scripts.profiles.within_between_distances

# Checkouts
uv run python -m src.workflow.scripts.profiles.within_between_distances \
  --count_type checkouts

# No plots, just CSV (faster)
uv run python -m src.workflow.scripts.profiles.within_between_distances \
  --no_plot
```

### 3.4 Mean Envelope Plots (Per Station)

Mean ± variability band for each day type, one plot per station.

```bash
# Envelope comparison set
uv run python -m src.workflow.scripts.profiles.mean_envelope_plots \
  --stations 03000 05105 09116

# Use quantile envelope instead of std
uv run python -m src.workflow.scripts.profiles.mean_envelope_plots \
  --stations 03000 --envelope_type quantile --quantile_low 0.1 --quantile_high 0.9
```

### 3.5 Mean Envelope Stations (Cross-Station)

Multiple stations overlaid, faceted by day type.

```bash
# Quad set — all day types
uv run python -m src.workflow.scripts.profiles.mean_envelope_stations \
  --stations 03000 07112 06000 07000

# Trio — weekdays only
uv run python -m src.workflow.scripts.profiles.mean_envelope_stations \
  --stations 03000 07112 06000 --day_types WD

# All stations, weekdays only (good overview)
uv run python -m src.workflow.scripts.profiles.mean_envelope_stations \
  --day_types WD
```

### 3.6 Heatmap of Station Profiles

Quick visual: rows = stations, columns = time bins.

```bash
# Weekdays (most common)
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles \
  --day_type WD

# Holidays (check for heterogeneity)
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles \
  --day_type HO

# Subset of stations
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles \
  --stations 03000 07112 06000 --day_type WD

# Custom colormap
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles \
  --day_type WD --cmap viridis
```

### 3.7 Clustering — Label Alignment

KMeans vs day-type labels. ARI score and confusion matrix.

```bash
# Single station, k=3
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment \
  --stations 03000

# Single station, k=4 (one cluster per day type)
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment \
  --stations 03000 --n-clusters 4

# Multiple stations
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment \
  --stations 03000 07112 06000
```

> **Note:** This script uses `--count-type in` / `out` (not `checkins`/`checkouts`).

### 3.8 Cluster Stations by Shape

Groups stations with similar profile *shapes* (ignores volume). Uses normalize-by-total → FPCA → Ward.

```bash
# Weekdays, 4 clusters
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape \
  --day_type WD

# Saturdays, 3 clusters
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape \
  --day_type SA --n_clusters 3
```

### 3.9 Cluster Stations by Shape + Scale

Groups stations similar in both shape **and** volume. Uses log-transform → Ward.

```bash
# Weekdays, 4 clusters
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape_scale \
  --day_type WD

# Compare: run both shape-only and shape+scale
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape \
  --day_type WD --n_clusters 4
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape_scale \
  --day_type WD --n_clusters 4
```

---

## Part 4 — Intensity Analysis Experiments

### 4.1 Time Rescaling QQ Plots

Tests if estimated intensity is correct. Checkins only.

```bash
# Single station, all dates (slow)
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots \
  --stations 03000

# 10% date sample (fast, recommended for iteration)
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots \
  --stations 03000 --date_percentage 0.1

# 30% date sample (balance speed/accuracy)
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots \
  --stations 03000 --date_percentage 0.3
```

> ⚠️ Checkouts not supported (no raw timestamps).

### 4.2 Fano Factor Analysis (Across Days)

Fano = Var/Mean per 15-min bin, across days. Poisson → Fano ≈ 1.

```bash
# Single station — checkins
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis \
  --stations 03000

# All stations (no --stations)
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis

# Checkouts
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis \
  --count_type checkouts
```

### 4.3 Fano Factor Within Bins

Subdivides each 15-min bin into δ-minute sub-bins and computes Fano within.

```bash
# Standard: 15-min outer bins, 1-min sub-bins, 10% dates
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000 --date_percentage 0.1

# Checkins with all dates (slow)
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000

# Checkouts — same params
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000 --date_percentage 0.1 --count_type checkouts

# Coarser bins: 30-min outer, 15-min sub-bins (checkouts)
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000 --date_percentage 0.1 --count_type checkouts \
  --time_step 30 --delta_minutes 15

# Even coarser: 60-min outer, 15-min sub-bins (checkouts)
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000 --date_percentage 0.1 --count_type checkouts \
  --time_step 60 --delta_minutes 15
```

### 4.4 Hawkes Process

Modeling event dependencies with a self-exciting point process.

| Script | Description |
|---|---|
| `hawkes_fit` | Fit continuous-time Hawkes process to exact check-in timestamps |
| `hawkes_diagnostics` | Branching ratio boxplots and time-rescaling GOF tests for Hawkes fits |
| `hawkes_simulate` | Simulate raw and binned synthetic traffic exact timestamps from the Hawkes process |

```bash
# Fit Hawkes process for a single station with 10% date sample
uv run python -m src.workflow.scripts.models.hawkes.step1_fit \
  --stations 03000 --date_percentage 0.1

# Run diagnostics for all fitted Hawkes models (no --stations needed)
uv run python -m src.workflow.scripts.models.hawkes.step2_diagnostics

# Simulate 10 synthetic baseline days using the median fitted parameters
uv run python -m src.workflow.scripts.models.hawkes.step3_simulate \
  --stations 03000 --n_days 10
```

---

## Experiment Progression (Suggested Order)

For resuming or starting fresh, follow this order:

```
1. Pipeline     → Run full pipeline (Part 1.1)
2. Quick check  → Verify DB with data_loader (Part 2.2)
3. Profile EDA  → Heatmap (3.6) → Envelope plots (3.4) → FPCA per station (3.1)
4. Day-type Q   → Distances (3.3) → Clustering label alignment (3.7)
5. Station Q    → FPCA across (3.2) → Cross-station envelopes (3.5)
6. Station groups → Shape clustering (3.8) → Shape+Scale clustering (3.9)
7. Poisson test → Fano across days (4.2) → Fano within bins (4.3)
8. Intensity    → Time rescaling QQ (4.1)
9. Hawkes       → Hawkes fit (4.4) → Hawkes diagnostics (4.4)
```

### Quick Smoke Test (runs fast)

```bash
uv run python -m src.workflow.data_loader
uv run python -m src.workflow.scripts.profiles.fpca_per_station --stations 03000 --n_components 3
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles --day_type WD
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis --stations 03000
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins --stations 03000 --date_percentage 0.1
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots --stations 03000 --date_percentage 0.1

# Hawkes modeling
uv run python -m src.workflow.scripts.models.hawkes.step1_fit --stations 03000 --date_percentage 0.1
uv run python -m src.workflow.scripts.models.hawkes.step2_diagnostics
```
