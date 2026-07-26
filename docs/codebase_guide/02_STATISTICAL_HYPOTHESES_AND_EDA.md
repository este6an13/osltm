# Statistical Hypotheses & Exploratory Data Analysis (EDA)

> **Document Part 2 of 4 in OSLTM Codebase Review**  
> **Master Guide Index**: [00_MASTER_INDEX_AND_ROADMAP.md](file:///d:/dequi/repositories/osltm/docs/codebase_guide/00_MASTER_INDEX_AND_ROADMAP.md)

---

## 1. Core Statistical Questions & Verified Hypotheses

Before fitting complex stochastic point process models, you designed a structured suite of exploratory analysis scripts operating on daily 15-minute arrival profile curves \(x_{i,s,g}(t)\) (where \(i\) is the day, \(s\) is the station, and \(g \in \{\text{WD}, \text{SA}, \text{SU}, \text{HO}\}\) is the day type).

These scripts were built to systematically test **four fundamental statistical hypotheses**:

```mermaid
graph TD
    H1[Hypothesis 1: Day-Type Separation] -->|Verified by distance ratios R < 1 & FPCA| R1[Conclusion: Fit separate intensity functions for WD, SA, SU, HO]
    H2[Hypothesis 2: Station Heterogeneity] -->|Verified by cross-station FPCA & Ward Clustering| R2[Conclusion: Station-specific rate functions lambda_s(t) are mandatory]
    H3[Hypothesis 3: Seasonality Invariance] -->|Verified across months/years| R3[Conclusion: Omit month/year seasonality; treat day-type rate as stationary]
    H4[Hypothesis 4: Poisson Overdispersion] -->|Verified by Fano factor >> 1 & Time-Rescaling QQ| R4[Conclusion: Reject NHPP; require Cox, Hawkes, or Neyman-Scott models]
```

---

### Hypothesis 1: Day-Type Separation (WD vs SA vs SU vs HO)
- **Question**: Do passenger arrival profiles differ significantly across day types, and are weekdays homogenous enough to group together?
- **Mathematical Test**:
  - Distance Ratio \(R = \frac{\bar{d}_{\text{within}}}{\bar{d}_{\text{between}}}\), where \(\bar{d}_{\text{within}}\) is the average Euclidean (\(L_2\)) distance between days of the same day type, and \(\bar{d}_{\text{between}}\) is the average distance between different day types.
  - Adjusted Rand Index (ARI) comparing unsupervised KMeans clusters of daily curves against day-type labels.
- **Empirical Findings**:
  - Ratio \(R \ll 1\) across all stations, proving strong statistical separation between Weekdays, Saturdays, Sundays, and Holidays.
  - Weekdays (Monday through Friday) show nearly identical profile shapes with minimal intra-week variance.
  - Saturdays exhibit a broad midday peak, Sundays show lower overall volume with afternoon leisure peaks, and Holidays resemble Sundays with distinct morning delays.
- **Modeling Rationale**: **Fully justifies defining a separate intensity function \(\lambda_{s,g}(t)\) per (station, day-type) pair**, while collapsing Monday–Friday into a single generic Weekday (WD) class.

---

### Hypothesis 2: Station Profile Heterogeneity & Functional Clustering
- **Question**: Do all stations share a single master arrival shape rescaled by total daily volume, or do stations exhibit distinct functional shape profiles?
- **Mathematical Test**:
  - Functional Principal Component Analysis (FPCA) across stations:
    \[ x_{i}(t) = \mu(t) + \sum_{k=1}^K \xi_{i,k} \phi_k(t) + \epsilon_i(t) \]
  - Hierarchical Ward clustering on normalized functional shapes (\(L_2\)-normalized curves) versus unnormalized shapes (shape + scale).
- **Empirical Findings**:
  - Stations display strong structural heterogeneity:
    1. **Commuter Origin Stations**: Sharp, tall morning rush-hour peak (06:00–08:30) with flat afternoon demand.
    2. **Destination / Downtown Hubs**: Low morning demand, tall evening departure peak (17:00–19:30).
    3. **Transfer & Mixed Terminals**: Bimodal symmetric distribution (dual morning and evening peaks).
- **Modeling Rationale**: **Justifies estimating station-specific rate functions \(\lambda_s(t)\)** rather than scaling a global system-wide profile curve.

---

### Hypothesis 3: Inter-Month / Inter-Year Seasonality Invariance
- **Question**: Should the point process model include seasonal components (e.g., month-of-year or year-over-year trends)?
- **Empirical Findings**:
  - Comparing daily arrival curves across different months (e.g., March vs June vs October) for a fixed day type (WD) showed that month-to-month variation is negligible relative to intraday variance.
  - Annual school/university holiday periods (mid-June to July, December) cause minor volume drops but preserve profile shape.
- **Modeling Rationale**: **Justifies omitting month/year seasonal parameters**, allowing the intensity model to remain stationary across the year for a given day type.

---

### Hypothesis 4: Failure of Poisson Process (Overdispersion & Burstiness)
- **Question**: Are passenger arrivals well-described by a standard Non-Homogeneous Poisson Process (NHPP)?
- **Mathematical Test**:
  - **Fano Factor Analysis**: Across replicate days for a fixed time bin \(k\):
    \[ \text{Fano}_{s,g,k} = \frac{\text{Var}(N_{s,g,k})}{\mathbb{E}[N_{s,g,k}]} \]
    For a Poisson process, \(\text{Fano} = 1\). \(\text{Fano} > 1\) indicates overdispersion.
  - **Time-Rescaling Theorem Kolmogorov-Smirnov (KS) Test**: Under an NHPP with rate \(\hat{\lambda}(t)\), the transformed inter-event times \(\Delta \Lambda_i = \int_{t_{i-1}}^{t_i} \hat{\lambda}(u) du\) must follow an i.i.d. Exponential(1) distribution, and \(U_i = 1 - \exp(-\Delta \Lambda_i)\) must be Uniform(0,1).
- **Empirical Findings**:
  - Empirical Fano factors ranged from **5.0 to 50.0+** across 15-minute bins, strongly rejecting the Poisson hypothesis (\(\text{Fano} = 1\)).
  - Time-rescaling QQ plots showed severe systematic deviations from the 45-degree diagonal line with KS test \(p\)-values \(< 10^{-10}\).
  - Overdispersion is caused by bus batch arrivals (queues clearing simultaneously when a bus arrives at a station platform) and latent environmental stochasticity (weather, traffic delays).
- **Modeling Rationale**: **Conclusively proves that standard NHPP is statistically invalid for TransMilenio arrivals**, justifying the transition to **Log-Gaussian Cox Processes (LGCP)**, **Hawkes Self-Exciting Processes**, and **Neyman-Scott Cluster Processes**.

---

## 2. Complete Catalog of Profile Analysis Scripts (`scripts/profiles/`)

The 9 profile analysis scripts located in [src/workflow/scripts/profiles/](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles) operate on 15-minute aggregated count matrices.

| Script Name | Mathematical Method | Input Parameters | Key Outputs & Plots | Statistical Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| [fpca_per_station.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/fpca_per_station.py) | Functional Principal Component Analysis (FPCA) per station | `--stations`, `--n_components` | `fpca_scores_{station}.png`, `fpca_components_{station}.png`, `fpca_results.csv` | Visualizes day-type clusters in the 2D/3D score space of FPCA eigen-functions \(\phi_1(t), \phi_2(t)\). |
| [fpca_across_stations.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/fpca_across_stations.py) | Cross-station joint FPCA | `--stations`, `--day_types` | `fpca_across_stations_scores.png`, `fpca_across_results.csv` | Evaluates inter-station variance vs intra-station variance under fixed day types. |
| [within_between_distances.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/within_between_distances.py) | Pairwise \(L_2\) distance matrix analysis | `--stations` | `within_between_ratios.png`, `distance_ratios.csv` | Calculates ratio \(R = \bar{d}_{\text{within}} / \bar{d}_{\text{between}}\). Ratios \(< 0.5\) confirm strong day-type separation. |
| [mean_envelope_plots.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/mean_envelope_plots.py) | Mean profile \(\mu(t)\) \(\pm\) 1st/3rd quartile envelopes | `--stations` | `mean_envelope_{station}.png`, `envelope_summary.csv` | Overlays mean curves for WD, SA, SU, HO with shaded variability bands per station. |
| [mean_envelope_stations.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/mean_envelope_stations.py) | Cross-station mean envelopes faceted by day type | `--stations`, `--day_types` | `mean_envelope_stations_{day_type}.png` | Displays station-to-station volume and shape differences side-by-side. |
| [heatmap_station_profiles.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/heatmap_station_profiles.py) | Normalized matrix heatmap (Stations \(\times\) Bins) | `--day_type` | `heatmap_stations_{day_type}.png` | High-density grid visualizer highlighting peak arrival windows across all stations simultaneously. |
| [clustering_label_alignment.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/clustering_label_alignment.py) | Unsupervised KMeans vs Day-Type labels | `--stations`, `--n-clusters` | `cluster_confusion_matrix.png`, `ari_scores.csv` | Calculates Adjusted Rand Index (ARI) and confusion matrices to test if unsupervised curve clustering discovers day-types naturally. |
| [cluster_stations_shape.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/cluster_stations_shape.py) | Pure shape station clustering (\(L_2\)-normalized curves + Ward) | `--day_type`, `--n_clusters` | `station_shape_clusters.png`, `station_clusters.csv` | Groups stations into functional types purely based on profile shape, ignoring total daily volume. |
| [cluster_stations_shape_scale.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/profiles/cluster_stations_shape_scale.py) | Joint shape + scale station clustering (Log-transform + Ward) | `--day_type`, `--n_clusters` | `station_shape_scale_clusters.png` | Groups stations taking into account both arrival profile shape and absolute magnitude. |

---

## 3. Complete Catalog of Intensity Analysis Scripts (`scripts/intensity/`)

The 3 intensity diagnostic scripts in [src/workflow/scripts/intensity/](file:///d:/dequi/repositories/osltm/src/workflow/scripts/intensity) evaluate overdispersion and goodness-of-fit against Poisson assumptions.

| Script Name | Mathematical Method | Input Parameters | Key Outputs & Plots | Statistical Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| [fano_factor_analysis.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/intensity/fano_factor_analysis.py) | Inter-day Fano factor across replicate dates per bin | `--stations` | `fano_factor_{station}.png`, `fano_summary.csv` | Computes \(\text{Fano}_k = \text{Var}(N_k) / \mathbb{E}[N_k]\) for each 15-min bin. Values \(> 5\) indicate massive overdispersion. |
| [fano_factor_within_bins.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/intensity/fano_factor_within_bins.py) | Intra-bin sub-minute resolution Fano factor | `--stations`, `--date_percentage` | `fano_within_bins_{station}.png` | Analyzes overdispersion inside sub-minute intervals within each 15-min bin using raw transaction timestamps. |
| [time_rescaling_qq_plots.py](file:///d:/dequi/repositories/osltm/src/workflow/scripts/intensity/time_rescaling_qq_plots.py) | Time-Rescaling Theorem Kolmogorov-Smirnov QQ plots | `--stations`, `--date_percentage` | `qq_plot_{station}_{day_type}.png`, `time_rescaling_ks_stats.csv` | Transforms event timestamps using estimated kernel intensity \(\hat{\lambda}(t)\) and plots empirical quantiles vs Uniform(0,1). |

---

## 4. Document Navigation Links

- Return to **Master Guide Index**: [00_MASTER_INDEX_AND_ROADMAP.md](file:///d:/dequi/repositories/osltm/docs/codebase_guide/00_MASTER_INDEX_AND_ROADMAP.md)
- Return to **Data Pipeline Details**: [01_DATA_PIPELINE_AND_PERSISTENCE.md](file:///d:/dequi/repositories/osltm/docs/codebase_guide/01_DATA_PIPELINE_AND_PERSISTENCE.md)
- Proceed to **Stochastic Point Process Models**: [03_STOCHASTIC_POINT_PROCESS_MODELS.md](file:///d:/dequi/repositories/osltm/docs/codebase_guide/03_STOCHASTIC_POINT_PROCESS_MODELS.md)
