# Experiment Pipeline Guide

> Step-by-step reference for running the full analysis pipeline — from data ingestion to intensity diagnostics.

---

## Prerequisites

- Database configured and accessible
- `uv` installed with the project virtualenv
- All commands run from the repository root

---

## Quick Start (Full Pipeline)

```bash
# 1. Data pipeline — sample dates, download files, sample stations, populate DB
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1,2,3,4

# 2. Profile analysis
uv run python -m src.workflow.scripts.profiles.fpca_per_station --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.profiles.fpca_across_stations --stations 03000 07112 06000 40000 02205 09121 --day_types WD SA
uv run python -m src.workflow.scripts.profiles.within_between_distances --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.profiles.mean_envelope_plots --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.profiles.mean_envelope_stations --stations 03000 07112 06000 40000 02205 09121 --day_types WD SA
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles --stations 03000 07112 06000 40000 02205 09121 --day_type WD
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment --stations 03000 07112 06000 40000 02205 09121 --n-clusters 4
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape --stations 03000 07112 06000 40000 02205 09121 --day_type WD --n_clusters 3
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape_scale --stations 03000 07112 06000 40000 02205 09121 --day_type WD --n_clusters 3

# 3. Intensity analysis
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins --stations 03000 07112 06000 40000 02205 09121 --date_percentage 0.1
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots --stations 03000 07112 06000 40000 02205 09121 --date_percentage 0.1
```

---

## Pipeline Stages

### Stage 1 — Data Pipeline

Fetches and processes raw data into the database. Run once (or when parameters change).

```bash
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1,2,3,4
```

| Step | What It Does |
|------|-------------|
| 1 | Stratified date sampling (WD/SA/SU/HO) |
| 2 | Downloads daily check-in/checkout CSVs |
| 3 | Scans files and samples stations |
| 4 | Computes 15-min bin counts and populates DB |

Steps can be run individually (e.g., `--steps 3,4` to re-sample stations without re-downloading).

> [!TIP]
> After running, verify that `src/workflow/data/sampled_dates.csv` and `src/workflow/data/sampled_stations.csv` exist.

---

### Stage 2 — Profile Analysis

These scripts operate on 15-min count profiles stored in the database. They produce both images and CSV files (see [CSV_OUTPUTS_REFERENCE.md](file:///d:/dequi/repositories/osltm/src/workflow/docs/CSV_OUTPUTS_REFERENCE.md)).

#### 2a. Dimensionality reduction (FPCA)

```bash
# Per-station FPCA — do day types separate within each station?
uv run python -m src.workflow.scripts.profiles.fpca_per_station \
  --stations 03000 07112 06000 40000 02205 09121

# Cross-station FPCA — inter-station variability
uv run python -m src.workflow.scripts.profiles.fpca_across_stations \
  --stations 03000 07112 06000 40000 02205 09121 \
  --day_types WD SA
```

#### 2b. Distance & separation

```bash
# Within vs between day-type distances — R < 1 = good separation
uv run python -m src.workflow.scripts.profiles.within_between_distances \
  --stations 03000 07112 06000 40000 02205 09121
```

#### 2c. Mean profiles & envelopes

```bash
# Per-station envelopes — pattern + variability
uv run python -m src.workflow.scripts.profiles.mean_envelope_plots \
  --stations 03000 07112 06000 40000 02205 09121

# Cross-station comparison within day types
uv run python -m src.workflow.scripts.profiles.mean_envelope_stations \
  --stations 03000 07112 06000 40000 02205 09121 \
  --day_types WD SA

# Heatmap overview
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles \
  --stations 03000 07112 06000 40000 02205 09121 \
  --day_type WD
```

#### 2d. Clustering

```bash
# Label alignment — do clusters match WD/SA/SU/HO?
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment \
  --stations 03000 07112 06000 40000 02205 09121
uv run python -m src.workflow.scripts.profiles.clustering_label_alignment \
  --stations 03000 07112 06000 40000 02205 09121 \
  --n-clusters 4

# Shape-based station clustering
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape \
  --stations 03000 07112 06000 40000 02205 09121 \
  --day_type WD --n_clusters 3

# Shape + scale station clustering
uv run python -m src.workflow.scripts.profiles.cluster_stations_shape_scale \
  --stations 03000 07112 06000 40000 02205 09121 \
  --day_type WD --n_clusters 3
```

---

### Stage 3 — Intensity Analysis

Tests whether arrival processes follow Poisson assumptions.

```bash
# Fano factor across days — Poisson → Fano ≈ 1
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis \
  --stations 03000 07112 06000 40000 02205 09121

# Fano factor within bins (sub-bin resolution)
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins \
  --stations 03000 07112 06000 40000 02205 09121 \
  --date_percentage 0.1

# Time rescaling QQ-plots + KS test
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots \
  --stations 03000 07112 06000 40000 02205 09121 \
  --date_percentage 0.1
```

> [!TIP]
> Use `--date_percentage 0.1` for intensity scripts to speed up experiments while maintaining day-type balance. Increase for final results.

---

## Outputs

All scripts save results to `src/workflow/results/` in subdirectories by analysis type:

```
src/workflow/results/
├── clustering_results/     # clustering CSVs and dendrograms
├── distance_results/       # distance ratio CSVs and distributions
├── envelope_results/       # mean/envelope CSVs and plots
├── fano_factor/            # Fano factor CSVs and plots
├── fano_factor_within_bins/# within-bin Fano CSVs and plots
├── fpca_results/           # FPCA score CSVs and scatter plots
├── heatmap_results/        # heatmap profile CSVs and images
└── time_rescaling/         # KS test CSVs and QQ-plots
```

Each directory contains **both images and CSV files**. See [CSV_OUTPUTS_REFERENCE.md](file:///d:/dequi/repositories/osltm/src/workflow/docs/CSV_OUTPUTS_REFERENCE.md) for detailed column descriptions and what conclusions each CSV supports.

---

## Common Variations

| What you want | How to do it |
|---|---|
| Analyze checkouts instead of checkins | Add `--count_type checkouts` |
| Focus on specific day types | Add `--day_types WD SA` (where supported) |
| Try different cluster counts | Add `--n_clusters 3` or `--n-clusters 4` |
| Use quantile envelopes instead of std | Add `--envelope_type quantile` |
| Speed up intensity scripts | Add `--date_percentage 0.1` |
| Analyze all stations (not a subset) | Omit the `--stations` argument |
| Redirect output | Add `--output_dir path/to/dir` |
