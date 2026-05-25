# OSLTM — Stochastic Modelling for TransMilenio

Stochastic modelling of Bogotá's TransMilenio bus system using **point processes** and **queueing networks**. Analyzes passenger check-in/check-out patterns across stations to characterize arrival intensity, profile shapes, and day-type effects.

> *Modelación Estocástica con Procesos de Puntos y Redes de Colas para la Simulación del Sistema de TransMilenio*

## Overview

This project downloads daily passenger transaction data from TransMilenio's public datasets, computes 15-minute arrival counts per station, and runs statistical analyses to understand:

- **Profile shape** — How does the arrival curve vary across stations and day types (Weekday/Saturday/Sunday/Holiday)?
- **Day-type separation** — Do weekdays, weekends, and holidays produce meaningfully different patterns?
- **Station clustering** — Which stations behave similarly in terms of shape and/or volume?
- **Poisson hypothesis** — Is the arrival process well-described by a non-homogeneous Poisson process?

## Project Structure

```
osltm/
├── src/
│   ├── workflow/
│   │   ├── workflow.py              # Pipeline orchestrator (steps 1–4)
│   │   ├── params.json              # Configuration for all steps
│   │   ├── data_loader.py           # Load count data from DB → DataFrame
│   │   ├── data_reader.py           # Load raw transactions from CSV
│   │   ├── steps/                   # Pipeline steps
│   │   │   ├── step1_sample_dates.py
│   │   │   ├── step2_download_files.py
│   │   │   ├── step3_sample_stations.py
│   │   │   └── step4_populate_counts.py
│   │   ├── scripts/                 # Analysis scripts
│   │   │   ├── profiles/            # 9 profile analysis scripts
│   │   │   ├── intensity/           # 3 intensity analysis scripts
│   │   │   └── models/              # Fitted model pipelines
│   │   │       ├── lgcp/            # Log-Gaussian Cox Process (4 steps)
│   │   │       ├── hawkes/          # Hawkes self-exciting process (3 steps + core)
│   │   │       └── simulation_comparison.py
│   │   ├── utils/
│   │   │   └── drop_unused_columns.py
│   │   ├── data/                    # Persisted sampled dates & stations
│   │   └── docs/                    # Reference documentation
│   ├── db/                          # SQLAlchemy models
│   ├── repo/                        # Database repositories
│   ├── constants/                   # Project constants
│   └── utils/                       # Shared utilities (day_type, etc.)
├── data/                            # Downloaded CSV files (gitignored)
│   ├── check_ins/daily/
│   └── check_outs/daily/
├── osltm.db                         # SQLite database with 15-min counts
└── pyproject.toml
```

## Requirements

- **Python ≥ 3.13**
- **[uv](https://docs.astral.sh/uv/)** for dependency management

## Setup

```bash
# Install dependencies
uv sync
```

## Running the Web UI

The project includes a full-featured web interface to run experiments and view results interactively. It consists of a FastAPI backend and a Next.js frontend.

```bash
# Terminal 1: Start the backend API
uv run uvicorn src.ui.backend.main:app --reload

# Terminal 2: Start the frontend (from the frontend folder)
cd src/ui/frontend
npm run dev
```

Navigate to `http://localhost:3000` in your browser.

## Data Pipeline

The pipeline has 4 sequential steps, configured via `src/workflow/params.json`:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `step1_sample_dates` | Stratified sampling of dates (WD/SA/SU/HO) from a date range |
| 2 | `step2_download_files` | Downloads daily CSV files from TransMilenio's public URLs |
| 3 | `step3_sample_stations` | Discovers stations from files and samples a subset |
| 4 | `step4_populate_counts` | Computes 15-min bin counts and populates the SQLite database |

```bash
# Run the full pipeline
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps all

# Run specific steps
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 1 2
uv run python -m src.workflow.workflow --params src/workflow/params.json --steps 4
```

## Data Loading

```python
from src.workflow.data_loader import load_data

# Load all data from the database
data = load_data()
checkins_df = data["checkins"]    # DataFrame: year, month, day, date_type, station_code, t_400...t_2300
checkouts_df = data["checkouts"]

# Filter by station and data type
data = load_data(
    station_codes=["03000", "07112"],
    include_checkins=True,
    include_checkouts=False,
)
```

```bash
# Quick verification (prints shape of loaded data)
uv run python -m src.workflow.data_loader
```

## Analysis Scripts

### Profile Analysis (`scripts/profiles/`)

| Script | Purpose |
|--------|---------|
| `fpca_per_station` | FPCA on daily profiles per station, colored by day type |
| `fpca_across_stations` | FPCA on all (station, day) profiles together |
| `within_between_distances` | Day-type separation ratio (within/between group distances) |
| `mean_envelope_plots` | Mean ± envelope per station, overlaying day types |
| `mean_envelope_stations` | Mean ± envelope across stations, faceted by day type |
| `heatmap_station_profiles` | Heatmap (stations × time bins) for a fixed day type |
| `clustering_label_alignment` | KMeans clustering vs day-type labels (ARI, confusion matrix) |
| `cluster_stations_shape` | Shape-based station clustering (normalize → FPCA → Ward) |
| `cluster_stations_shape_scale` | Shape+scale clustering (log-transform → Ward) |

### Intensity Analysis (`scripts/intensity/`)

| Script | Purpose |
|--------|---------|
| `time_rescaling_qq_plots` | Time rescaling theorem QQ-plots (checkins only) |
| `fano_factor_analysis` | Fano factor (Var/Mean) across days per time bin |
| `fano_factor_within_bins` | Fano factor within bins using sub-minute resolution |

### Models (`scripts/models/`)

#### LGCP — Log-Gaussian Cox Process (`models/lgcp/`)

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `step1_twostage` | Fit GP kernels (SE + Matérn 3/2) to log-residual covariance |
| 2 | `step2_bayesian` | Full Bayesian LGCP posterior via Laplace approximation |
| 3 | `step3_gof` | Goodness-of-fit: LGCP vs NHPP via time-rescaling KS test |
| 4 | `step4_simulate` | Simulate synthetic arrivals and aggregate to 15-min counts |

#### Hawkes — Self-Exciting Point Process (`models/hawkes/`)

| Step | Script | Purpose |
|------|--------|---------|
| — | `core` | Math engine: fitting, compensator, branching simulation |
| 1 | `step1_fit` | Fit continuous-time Hawkes to exact timestamps |
| 2 | `step2_diagnostics` | Branching ratio boxplots and GOF assessment |
| 3 | `step3_simulate` | Simulate raw and binned synthetic traffic |

> **Note on checkouts:** Checkout data is 15-min aggregated. The Hawkes pipeline applies uniform jitter to enable fitting, but this is a pseudo-continuous approximation — LGCP is recommended for checkout analysis.

#### Cross-Model Comparison (`models/simulation_comparison.py`)

Computes MAE, RMSE, MAPE, Pearson r, total-count ratio, Wasserstein distance, and ±2σ coverage for each (model, count_type, station, day_type) combination.

### Running Analysis Scripts

```bash
# Profile analysis
uv run python -m src.workflow.scripts.profiles.fpca_per_station --stations 03000 --n_components 3
uv run python -m src.workflow.scripts.profiles.heatmap_station_profiles --day_type WD
uv run python -m src.workflow.scripts.profiles.mean_envelope_plots --stations 03000 05105 09116

# Intensity analysis
uv run python -m src.workflow.scripts.intensity.fano_factor_analysis --stations 03000
uv run python -m src.workflow.scripts.intensity.fano_factor_within_bins --stations 03000 --date_percentage 0.1
uv run python -m src.workflow.scripts.intensity.time_rescaling_qq_plots --stations 03000 --date_percentage 0.1

# LGCP pipeline
uv run python -m src.workflow.scripts.models.lgcp.step1_twostage --stations 03000
uv run python -m src.workflow.scripts.models.lgcp.step2_bayesian --stations 03000
uv run python -m src.workflow.scripts.models.lgcp.step3_gof --stations 03000
uv run python -m src.workflow.scripts.models.lgcp.step4_simulate --stations 03000

# Hawkes pipeline
uv run python -m src.workflow.scripts.models.hawkes.step1_fit --stations 03000 --date_percentage 0.1
uv run python -m src.workflow.scripts.models.hawkes.step2_diagnostics
uv run python -m src.workflow.scripts.models.hawkes.step3_simulate --stations 03000 --n_days 10

# Cross-model comparison
uv run python -m src.workflow.scripts.models.simulation_comparison
```

All scripts accept `--stations` to limit computation to specific stations and `--output_dir` to control where plots are saved. The `--date_percentage` flag (intensity scripts) samples a fraction of dates per day type for faster experimentation.

> For a complete parameter reference, see [WORKFLOW_REFERENCE.md](src/workflow/docs/WORKFLOW_REFERENCE.md).
> For pre-built experiment commands, see [EXPERIMENTS_PLAYBOOK.md](src/workflow/docs/EXPERIMENTS_PLAYBOOK.md).

## Utilities

### Drop Unused Columns

Strips downloaded CSVs to only the columns needed by step 4 (~85% reduction for check-ins, ~60% for check-outs).

```bash
# All files
uv run python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json

# Specific files
uv run python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json --files 20240625,20240628

# Date range
uv run python -m src.workflow.utils.drop_unused_columns --params src/workflow/params.json --date-start 20240625 --date-end 20240630
```

## Key Concepts

- **Day types:** WD (Weekday), SA (Saturday), SU (Sunday), HO (Holiday — uses the `holidays` library for Colombia)
- **Time bins:** 15-minute windows from 04:00 to 23:00 (columns `t_400` through `t_2300`)
- **Fano factor:** Var(N)/E(N). Equals 1 for Poisson processes; >1 indicates overdispersion
- **Time rescaling theorem:** If the estimated intensity is correct, rescaled inter-event times follow Uniform(0,1)
- **FPCA:** Functional Principal Component Analysis — PCA applied to daily count curves treated as functional data