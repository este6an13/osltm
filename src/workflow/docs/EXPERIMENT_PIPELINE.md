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
# 1. Data pipeline — sample dates, download files, sample stations, populate DB, generate network
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1,2,3,4,5

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
uv run python -m src.workflow.scripts.models.hawkes.step1_fit --stations 03000 07112 06000 40000 02205 09121 --date_percentage 0.1
```

---

## Pipeline Stages

### Stage 1 — Data Pipeline

Fetches and processes raw data into the database. Run once (or when parameters change).

```bash
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1,2,3,4,5
```

| Step | What It Does |
|------|-------------|
| 1 | Stratified date sampling (WD/SA/SU/HO) |
| 2 | Downloads daily check-in/checkout CSVs |
| 3 | Scans files and samples stations |
| 4 | Computes 15-min bin counts and populates DB |
| 5 | Generates the static topological and geographical network structure |

Steps can be run individually (e.g., `--steps 3,4` to re-sample stations without re-downloading, or `--steps 5` to just build the network).

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

### Stage 4 — Cox Process Diagnostics

Once Fano factors confirm overdispersion (Fano > 1), these scripts quantify and model the extra-Poisson variability.

#### 4a. Negative Binomial per-bin fit

```bash
# Fit Poisson vs NegBin per (station, day_type, time_bin) — AIC comparison
uv run python -m src.workflow.scripts.intensity.negbin_fit \
  --stations 03000 07112 06000 40000 02205 09121
```

| Output | What it shows |
|---|---|
| `negbin_fit_checkins.csv` | Per-bin: r̂, p̂, Fano, log-likelihoods, AICs, preferred model |
| `negbin_summary_checkins.csv` | Per day-type: % bins preferring NegBin, median r̂, median ΔAIC |
| `negbin_dispersion_checkins.png` | 1/r̂ vs time-of-day (higher = more overdispersion) |
| `negbin_aic_comparison_checkins.png` | ΔAIC vs time-of-day (positive = NegBin preferred) |

#### 4b. LGCP two-stage estimation

```bash
# Fit GP kernels (SE + Matérn 3/2) to log-residual covariance + Gaussianity validation
uv run python -m src.workflow.scripts.models.lgcp.step1_twostage \
  --stations 03000 07112 06000 40000 02205 09121
```

| Output | What it shows |
|---|---|
| `lgcp_kernel_params_checkins.csv` | Per (s,d): σ̂², ℓ̂, η̂², AIC/BIC for SE and Matérn, selected kernel |
| `lgcp_summary_checkins.csv` | Per day-type: median kernel params, % bins rejecting normality |
| `lgcp_gaussianity_checkins.csv` | Per (s,d,k): Shapiro–Wilk W and p-value |
| `lgcp_empirical_cov_*.png` | K×K empirical covariance heatmaps |
| `lgcp_kernel_fit_*.png` | Empirical vs fitted kernel (diagonal + slice) |
| `lgcp_residual_qq_*.png` | Mahalanobis D² QQ-plots against χ²_K |

#### 4c. Full Bayesian LGCP (Laplace approximation)

```bash
# Posterior intensity with credible bands + predictive Fano factors
uv run python -m src.workflow.scripts.models.lgcp.step2_bayesian \
  --stations 03000 07112 06000 40000 02205 09121
```

> [!NOTE]
> Requires Phase 2 results (`lgcp_twostage`) to exist first.

| Output | What it shows |
|---|---|
| `lgcp_posterior_params_checkins.csv` | Per (s,d,k): posterior mean/std of z, posterior Λ with 95% band |
| `lgcp_predictive_fano_checkins.csv` | Observed vs model-implied Fano factor per bin |
| `lgcp_posterior_*.png` | Posterior mean Λ(t) with credible band + observed profiles |
| `lgcp_predictive_fano_checkins.png` | Fano factor comparison: model vs observed vs Poisson |

#### 4d. LGCP Goodness-of-Fit (time-rescaling)

```bash
# Compare LGCP vs NHPP via time-rescaling KS test
uv run python -m src.workflow.scripts.models.lgcp.step3_gof \
  --stations 03000 07112 06000 02205 09121
```

> [!NOTE]
> Requires Phase 3 results (`lgcp_bayesian`) and existing NHPP time-rescaling results.

| Output | What it shows |
|---|---|
| `lgcp_gof_ks_comparison_checkins.csv` | Per (s,d): NHPP vs LGCP KS statistics and reduction % |
| `lgcp_gof_qq_*.png` | Side-by-side QQ-plots (NHPP vs LGCP) |
| `lgcp_gof_ks_reduction_checkins.png` | Bar chart of KS improvement |

#### 4e. Continuous-Time Hawkes Process (Exact Timestamps)

```bash
# Fit continuous-time Hawkes process and generate branching ratio diagnostics
uv run python -m src.workflow.scripts.models.hawkes.step1_fit \
  --stations 03000 07112 06000 02205 09121 --date_percentage 0.1
uv run python -m src.workflow.scripts.models.hawkes.step2_diagnostics

# Simulate synthetic exact arrivals and aggregate 15-minute counts
uv run python -m src.workflow.scripts.models.hawkes.step3_simulate \
  --stations 03000 07112 06000 02205 09121 --n_days 5
```

> [!NOTE]
> Represents an alternative to LGCP, treating the variance as direct event-triggering (self-excitation) rather than a fluctuating conditionally independent environment. Works strictly on exact timestamp formats (check-ins).

| Output | What it shows |
|---|---|
| `hawkes_params_checkins.csv` | Branching ratios ($n$), excitation $\alpha$, decay $\beta$ and baseline volume $\kappa$ |
| `hawkes_branching_ratio_boxplots.png` | Distribution of $n$ factored by day type |
| `hawkes_ks_pval_hist.png` | Assessment of GOF uniformization strictness |
| `hawkes_simulated_events_checkins.csv` | Synthetic raw timestamps `[Fecha_Transaccion, Estacion_Parada]` |
| `hawkes_simulated_binned_checkins.csv` | Synthetic 15-minute counts generated from the raw timestamps |
| `hawkes_simulation_comparison_checkins.png` | Plot comparing mean synthetic counts vs observed empirical counts |

---

### Stage 5 — Network Analysis

Tests network topology and topological characteristics using graph theory.

```bash
# Calculate node centralities and generate graph layouts
uv run python -m src.workflow.scripts.network.analyze_network --layout kamada_kawai

# Use a different layout algorithm
uv run python -m src.workflow.scripts.network.analyze_network --layout spiral
```

| Output | What it shows |
|---|---|
| `network_graph.json` | The graph node/link data with centralities |
| `network_geo_*.png` | Geo-spatial network plots colored by centrality |
| `network_abstract_*.png` | Abstract force-directed graphs |

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
├── lgcp_bayesian/          # Bayesian LGCP posterior plots and Fano comparison
├── lgcp_twostage/          # LGCP kernel fit CSVs and diagnostic plots
├── negbin_fit/             # NegBin vs Poisson fit CSVs and plots
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
