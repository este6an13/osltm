from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Literal

class ScriptParam(BaseModel):
    name: str
    type: Literal["int", "float", "str", "bool", "station_list", "choice", "multi_choice", "path", "flag", "date_percentage"]
    default: Any = None
    choices: Optional[List[Any]] = None
    required: bool = False
    description: str = ""

class ScriptDef(BaseModel):
    module: str
    name: str
    description: str
    category: str
    output_dir: str = ""
    depends_on: str = ""    # output_dir of the upstream step this script reads from
    input_arg: str = ""     # CLI flag used to pass the upstream path (e.g. --phase2_dir)
    params: List[ScriptParam]

# Based on EXPERIMENTS_PLAYBOOK.md and WORKFLOW_REFERENCE.md

SCRIPTS = {
    # Profiles
    "profiles/fpca_per_station": ScriptDef(
        module="src.workflow.scripts.profiles.fpca_per_station",
        name="FPCA Per Station",
        description="FPCA on daily profiles for each station individually. Points colored by date type reveal whether day types separate in principal component space.",
        category="profiles",
        output_dir="fpca_results",
        params=[
            ScriptParam(name="stations", type="station_list", description="Subset of station codes to analyze"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins", description="Checkins or checkouts"),
            ScriptParam(name="n_components", type="int", default=3, description="Number of PCs to extract"),
            ScriptParam(name="no_standardize", type="flag", default=False, description="Skip standardization before PCA")
        ]
    ),
    "profiles/fpca_across_stations": ScriptDef(
        module="src.workflow.scripts.profiles.fpca_across_stations",
        name="FPCA Across Stations",
        description="FPCA on all (station, day) profiles together. Shows inter-station variability.",
        category="profiles",
        output_dir="fpca_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="day_types", type="multi_choice", choices=["WD", "SA", "SU", "HO"], description="Filter to specific day types.", default=["WD"]),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="n_components", type="int", default=3),
            ScriptParam(name="no_standardize", type="flag", default=False)
        ]
    ),
    "profiles/within_between_distances": ScriptDef(
        module="src.workflow.scripts.profiles.within_between_distances",
        name="Within vs Between Distances",
        description="Compute ratio R = mean(within-group distance) / mean(between-group distance). R < 1 implies good day-type separation.",
        category="profiles",
        output_dir="distance_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="no_plot", type="flag", default=False, description="Skip plotting")
        ]
    ),
    "profiles/mean_envelope_plots": ScriptDef(
        module="src.workflow.scripts.profiles.mean_envelope_plots",
        name="Mean Envelope Plots",
        description="For each station, overlay mean +/- envelope for each day type.",
        category="profiles",
        output_dir="envelope_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="envelope_type", type="choice", choices=["std", "quantile"], default="std"),
            ScriptParam(name="quantile_low", type="float", default=0.1),
            ScriptParam(name="quantile_high", type="float", default=0.9),
        ]
    ),
    "profiles/mean_envelope_stations": ScriptDef(
        module="src.workflow.scripts.profiles.mean_envelope_stations",
        name="Mean Envelope Cross-Stations",
        description="Multiple stations overlaid, faceted by day type.",
        category="profiles",
        output_dir="envelope_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="day_types", type="multi_choice", choices=["WD", "SA", "SU", "HO"], description="Day types to include", default=["WD", "SA", "SU", "HO"]),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="envelope_type", type="choice", choices=["std", "quantile"], default="std"),
        ]
    ),
    "profiles/heatmap_station_profiles": ScriptDef(
        module="src.workflow.scripts.profiles.heatmap_station_profiles",
        name="Heatmap of Station Profiles",
        description="Heatmap of mean profiles for a fixed day type.",
        category="profiles",
        output_dir="heatmap_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="day_type", type="choice", choices=["WD", "SA", "SU", "HO"], default="WD"),
            ScriptParam(name="cmap", type="str", default="YlOrRd"),
        ]
    ),
    "profiles/clustering_label_alignment": ScriptDef(
        module="src.workflow.scripts.profiles.clustering_label_alignment",
        name="Clustering Label Alignment",
        description="K-Means clustering vs day-type labels (ARI score, confusion matrix).",
        category="profiles",
        output_dir="clustering_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["in", "out"], default="in", description="Note: Uses 'in'/'out' instead of checkins/checkouts"),
            ScriptParam(name="n_clusters", type="int", default=3),
            ScriptParam(name="no_normalize", type="flag", default=False),
            ScriptParam(name="seed", type="int", default=42),
        ]
    ),
    "profiles/cluster_stations_shape": ScriptDef(
        module="src.workflow.scripts.profiles.cluster_stations_shape",
        name="Cluster Stations by Shape",
        description="Shape-based clustering (normalized).",
        category="profiles",
        output_dir="clustering_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="day_type", type="choice", choices=["WD", "SA", "SU", "HO"], default="WD"),
            ScriptParam(name="n_clusters", type="int", default=4),
        ]
    ),
    "profiles/cluster_stations_shape_scale": ScriptDef(
        module="src.workflow.scripts.profiles.cluster_stations_shape_scale",
        name="Cluster Stations by Shape & Scale",
        description="Shape + scale clustering (log-transformed).",
        category="profiles",
        output_dir="clustering_results",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="day_type", type="choice", choices=["WD", "SA", "SU", "HO"], default="WD"),
            ScriptParam(name="n_clusters", type="int", default=4),
        ]
    ),

    # Intensity
    "intensity/time_rescaling_qq_plots": ScriptDef(
        module="src.workflow.scripts.intensity.time_rescaling_qq_plots",
        name="Time Rescaling QQ Plots",
        description="Apply time rescaling theorem. (Checkins only)",
        category="intensity",
        output_dir="time_rescaling",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="time_window_minutes", type="float", default=15.0),
            ScriptParam(name="date_percentage", type="date_percentage", default=0.1, description="Fraction of dates to sample (speedup)"),
            ScriptParam(name="date_type", type="multi_choice", choices=["WD", "SA", "SU", "HO"], description="Day types to include (empty = all)", default=["WD", "SA", "SU", "HO"]),
        ]
    ),
    "intensity/fano_factor_analysis": ScriptDef(
        module="src.workflow.scripts.intensity.fano_factor_analysis",
        name="Fano Factor Across Days",
        description="Fano factor = Var(N)/E(N) across days. Poisson → Fano ≈ 1.",
        category="intensity",
        output_dir="fano_factor",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
        ]
    ),
    "intensity/fano_factor_within_bins": ScriptDef(
        module="src.workflow.scripts.intensity.fano_factor_within_bins",
        name="Fano Factor Within Bins",
        description="Fano factor within each 15-min bin using delta-minute sub-bins.",
        category="intensity",
        output_dir="fano_factor_within_bins",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="date_percentage", type="date_percentage", default=0.1),
            ScriptParam(name="time_step", type="str", default="15", description="Outer bin size"),
            ScriptParam(name="delta_minutes", type="str", default="1", description="Sub-bin size"),
            ScriptParam(name="date_type", type="multi_choice", choices=["WD", "SA", "SU", "HO"], description="Day types to include (empty = all)", default=["WD", "SA", "SU", "HO"]),
        ]
    ),
    "intensity/negbin_fit": ScriptDef(
        module="src.workflow.scripts.intensity.negbin_fit",
        name="Negative Binomial Fit",
        description="Fit Poisson vs NegBin per time bin and day type.",
        category="intensity",
        output_dir="negbin_fit",
        params=[
            ScriptParam(name="stations", type="station_list"),
        ]
    ),
    
    # Models - LGCP
    "models/lgcp/step1_twostage": ScriptDef(
        module="src.workflow.scripts.models.lgcp.step1_twostage",
        name="LGCP - Step 1: Two-Stage",
        description="Fit GP kernels to log-residual covariance.",
        category="models/lgcp",
        output_dir="lgcp_twostage",
        params=[
            ScriptParam(name="stations", type="station_list"),
        ]
    ),
    "models/lgcp/step2_bayesian": ScriptDef(
        module="src.workflow.scripts.models.lgcp.step2_bayesian",
        name="LGCP - Step 2: Bayesian",
        description="Full Bayesian LGCP via Laplace approximation. Requires Step 1 kernel params.",
        category="models/lgcp",
        output_dir="lgcp_bayesian",
        depends_on="lgcp_twostage",
        input_arg="--phase2_dir",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="min_days", type="int", default=10, description="Min replicate days per (station, day_type)"),
        ]
    ),
    "models/lgcp/step3_gof": ScriptDef(
        module="src.workflow.scripts.models.lgcp.step3_gof",
        name="LGCP - Step 3: Goodness of Fit",
        description="PIT-based GoF comparison: Poisson (NHPP) vs LGCP. Requires Step 1 kernel params.",
        category="models/lgcp",
        output_dir="lgcp_gof",
        depends_on="lgcp_twostage",
        input_arg="--phase2_dir",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="min_days", type="int", default=5),
            ScriptParam(name="n_mc", type="int", default=500, description="Monte Carlo samples for PLN CDF"),
        ]
    ),
    
    # Models - Hawkes
    "models/hawkes/step1_fit": ScriptDef(
        module="src.workflow.scripts.models.hawkes.step1_fit",
        name="Hawkes - Step 1: Fit",
        description="Fit continuous-time Hawkes process.",
        category="models/hawkes",
        output_dir="hawkes_fit",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="date_percentage", type="date_percentage", default=0.1),
        ]
    ),
    "models/hawkes/step2_diagnostics": ScriptDef(
        module="src.workflow.scripts.models.hawkes.step2_diagnostics",
        name="Hawkes - Step 2: Diagnostics",
        description="Branching ratio diagnostics. Requires Step 1 fitted params CSV.",
        category="models/hawkes",
        output_dir="hawkes_fit",
        depends_on="hawkes_fit",
        input_arg="--input",
        params=[
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
        ]
    ),
    "models/hawkes/step3_simulate": ScriptDef(
        module="src.workflow.scripts.models.hawkes.step3_simulate",
        name="Hawkes - Step 3: Simulate",
        description="Simulate synthetic process and aggregate counts. Requires Step 1 fitted params.",
        category="models/hawkes",
        output_dir="hawkes_simulate",
        depends_on="hawkes_fit",
        input_arg="--fit_dir",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="n_days", type="int", default=5),
        ]
    ),
    "models/avg_profile/step1_fit": ScriptDef(
        module="src.workflow.scripts.models.avg_profile.step1_fit",
        name="Avg Profile - Step 1: Fit",
        description="Compute historical mean and standard deviation profiles per station and day type.",
        category="models/avg_profile",
        output_dir="avg_profile",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="day_types", type="multi_choice", choices=["WD", "SA", "SU", "HO"], default=["WD", "SA", "SU", "HO"]),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="cutoff_date", type="str", default="2025-11-30", description="YYYY-MM-DD cutoff for training data"),
        ]
    ),
    "models/avg_profile/step2_simulate": ScriptDef(
        module="src.workflow.scripts.models.avg_profile.step2_simulate",
        name="Avg Profile - Step 2: Simulate & Diagnostics",
        description="Simulate synthetic days and automatically generate diagnostic envelope plots against test data.",
        category="models/avg_profile",
        output_dir="avg_profile",
        depends_on="avg_profile",
        input_arg="--fit_dir",
        params=[
            ScriptParam(name="stations", type="station_list"),
            ScriptParam(name="day_types", type="multi_choice", choices=["WD", "SA", "SU", "HO"], default=["WD", "SA", "SU", "HO"]),
            ScriptParam(name="count_type", type="choice", choices=["checkins", "checkouts"], default="checkins"),
            ScriptParam(name="dist_type", type="choice", choices=["poisson", "neg_binomial"], default="poisson"),
            ScriptParam(name="n_days", type="int", default=30),
        ]
    ),
}
