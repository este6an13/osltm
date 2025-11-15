import warnings
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
from sqlalchemy.orm import Session

from src.db.session_v1 import SessionLocal
from src.repo.v1.estimates.models import Estimate


def _load_estimate_data(station_ids: list[int]) -> pd.DataFrame:
    """Load estimate records for selected stations from the database."""
    session: Session = SessionLocal()
    query = session.query(Estimate).filter(Estimate.station_id.in_(station_ids))
    df = pd.read_sql(query.statement, session.bind)
    session.close()
    return df


# ==========================================================
# 0. Test weekday consistency (Mon–Fri) within each station
# ==========================================================


def analyze_weekday_consistency(
    station_ids: list[int],
    lambda_type: str = "in",  # "in" or "out"
    plot: bool = False,
) -> pd.DataFrame:
    """
    Perform Kruskal–Wallis test across weekdays (Mon–Fri)
    for each selected station, testing if weekday λ patterns differ.

    Returns a DataFrame with station_id, H-statistic, and p-value.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"

    results = []
    for station_id, g in df.groupby("station_id"):
        # Filter only weekdays (WD type, Mon–Fri)
        g = g[(g["date_type"] == "WD") & (g["day_of_week"].between(0, 4))]

        if g.empty:
            continue

        samples = [group[value_col].dropna() for _, group in g.groupby("day_of_week")]
        if len(samples) > 1:
            stat, p = kruskal(*samples)
            results.append(
                {
                    "station_id": station_id,
                    "n_groups": len(samples),
                    "H_stat": stat,
                    "p_value": p,
                }
            )

        if plot:
            plt.figure(figsize=(8, 5))
            sns.boxplot(
                data=g,
                x="day_of_week",
                y=value_col,
                order=[0, 1, 2, 3, 4],
            )
            plt.title(f"λ_{lambda_type} Distribution by Weekday (Station {station_id})")
            plt.xlabel("Day of Week (0=Mon … 4=Fri)")
            plt.ylabel(value_col)
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


# ==========================================================
# 1. Test differences across date types (WD, SA, SU, HO)
# ==========================================================


def analyze_date_type_differences(
    station_ids: list[int],
    lambda_type: str = "in",  # "in" or "out"
    plot: bool = False,
) -> pd.DataFrame:
    """
    Perform Kruskal–Wallis test across date types (WD, SA, SU, HO)
    for each selected station.

    Returns a DataFrame with station_id, H-statistic, and p-value.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"

    results = []
    for station_id, g in df.groupby("station_id"):
        samples = [group[value_col].dropna() for _, group in g.groupby("date_type")]
        if len(samples) > 1:
            stat, p = kruskal(*samples)
            results.append(
                {
                    "station_id": station_id,
                    "n_groups": len(samples),
                    "H_stat": stat,
                    "p_value": p,
                }
            )

        if plot:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=g, x="date_type", y=value_col)
            plt.title(
                f"λ_{lambda_type} Distribution by Date Type (Station {station_id})"
            )
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


# ==========================================================
# 2. Compare SU vs HO (should be similar)
# ==========================================================


def compare_su_vs_ho(
    station_ids: list[int],
    lambda_type: str = "in",
    plot: bool = False,
) -> pd.DataFrame:
    """
    Perform Mann–Whitney U test comparing SU vs HO λ distributions
    for each selected station.

    Returns a DataFrame with station_id, statistic, and p-value.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"

    results = []
    for station_id, g in df.groupby("station_id"):
        su = g[g["date_type"] == "SU"][value_col].dropna()
        ho = g[g["date_type"] == "HO"][value_col].dropna()

        if len(su) > 0 and len(ho) > 0:
            stat, p = mannwhitneyu(su, ho, alternative="two-sided")
            results.append(
                {
                    "station_id": station_id,
                    "n_su": len(su),
                    "n_ho": len(ho),
                    "U_stat": stat,
                    "p_value": p,
                }
            )

            if plot:
                plt.figure(figsize=(8, 5))
                sns.kdeplot(su, fill=True, label="SU")
                sns.kdeplot(ho, fill=True, label="HO")
                plt.title(f"λ_{lambda_type} — SU vs HO (Station {station_id})")
                plt.legend()
                plt.grid(True, alpha=0.4)
                plt.tight_layout()
                plt.show()

    return pd.DataFrame(results)


# ==========================================================
# 3. Month-to-month stability
# ==========================================================


def analyze_monthly_stability(
    station_ids: list[int],
    lambda_type: str = "in",
    plot: bool = False,
) -> pd.DataFrame:
    """
    Perform Kruskal–Wallis test across months for each selected station.

    Returns a DataFrame with station_id, H-statistic, and p-value.
    """
    df = _load_estimate_data(station_ids)
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"

    results = []
    for station_id, g in df.groupby("station_id"):
        samples = [group[value_col].dropna() for _, group in g.groupby("month")]
        if len(samples) > 1:
            stat, p = kruskal(*samples)
            results.append(
                {
                    "station_id": station_id,
                    "n_months": len(samples),
                    "H_stat": stat,
                    "p_value": p,
                }
            )

        if plot:
            plt.figure(figsize=(8, 5))
            sns.boxplot(data=g, x="month", y=value_col)
            plt.title(f"λ_{lambda_type} Distribution by Month (Station {station_id})")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.show()

    return pd.DataFrame(results)


def _qq_plot(series: pd.Series, title: str = "QQ plot"):
    """Display QQ-plot for visual normality check."""
    plt.figure(figsize=(6, 6))
    stats.probplot(series.dropna(), dist="norm", plot=plt)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def _normality_tests(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Perform several normality tests on a 1D numeric series.

    Returns a dict with test statistics, p-values (where applicable), and boolean 'reject' flags.
    """
    arr = series.dropna().to_numpy()
    n = len(arr)
    result = {"n": n}

    if n < 8:
        # Too small for many tests; Shapiro requires n >= 3, but small samples are unreliable.
        result["note"] = "small_sample"
    # Shapiro-Wilk (sensitive for small/medium samples)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            sh_stat, sh_p = stats.shapiro(arr)
        result.update(
            {
                "shapiro_stat": float(sh_stat),
                "shapiro_p": float(sh_p),
                "shapiro_reject": sh_p < alpha,
            }
        )
    except Exception as e:
        result["shapiro_error"] = str(e)

    # D'Agostino K^2 (normaltest): best for moderate/large n
    try:
        k2_stat, k2_p = stats.normaltest(arr)
        result.update(
            {
                "dagostino_stat": float(k2_stat),
                "dagostino_p": float(k2_p),
                "dagostino_reject": k2_p < alpha,
            }
        )
    except Exception as e:
        result["dagostino_error"] = str(e)

    # Anderson-Darling (gives critical values)
    try:
        ad_res = stats.anderson(arr, dist="norm")
        # store statistic and the critical values / significance levels
        result.update(
            {
                "anderson_stat": float(ad_res.statistic),
                "anderson_critical_values": list(map(float, ad_res.critical_values)),
                "anderson_significance_levels": list(
                    map(float, ad_res.significance_level)
                ),
                # ad_reject boolean: statistic > critical_value at alpha closest to typical 5%
                "anderson_reject_at_5pct": float(ad_res.statistic)
                > float(
                    ad_res.critical_values[list(ad_res.significance_level).index(5.0)]
                )
                if 5.0 in ad_res.significance_level
                else None,
            }
        )
    except Exception as e:
        result["anderson_error"] = str(e)

    # Kolmogorov-Smirnov against fitted normal (use mean/std of sample)
    try:
        mu, sigma = np.mean(arr), np.std(arr, ddof=1)
        if sigma <= 0 or n < 2:
            result["ks_error"] = "sigma_zero_or_insufficient_data"
        else:
            ks_stat, ks_p = stats.kstest(arr, "norm", args=(mu, sigma))
            result.update(
                {
                    "ks_stat": float(ks_stat),
                    "ks_p": float(ks_p),
                    "ks_reject": ks_p < alpha,
                }
            )
    except Exception as e:
        result["ks_error"] = str(e)

    return result


def test_normality_by_station(
    station_ids: List[int],
    lambda_type: Literal["in", "out"] = "in",
    group_by: Optional[Literal["pooled", "date_type", "day_of_week"]] = "pooled",
    alpha: float = 0.05,
    plot: bool = False,
) -> pd.DataFrame:
    """
    Test normality of estimated lambda series for each station.

    Args:
        station_ids: list of station_id integers
        lambda_type: "in" or "out" (selects estimated_lambda_in/out)
        group_by:
            - "pooled": test the pooled λ series for the station (all rows)
            - "date_type": test per date_type group (WD/SA/SU/HO)
            - "day_of_week": test per day_of_week group (0..6)
        alpha: significance threshold for tests that provide p-values
        plot: show QQ-plots (one per tested group)

    Returns:
        DataFrame with one row per tested group and columns:
        [station_id, group_label, n, shapiro_stat, shapiro_p, shapiro_reject, dagostino_stat, dagostino_p, dagostino_reject, ad_stat, ks_stat, ks_p, ks_reject, ...]
    """
    value_col = "estimated_lambda_in" if lambda_type == "in" else "estimated_lambda_out"

    df = _load_estimate_data(station_ids)

    records = []
    for station_id, g in df.groupby("station_id"):
        if group_by == "pooled":
            groups_to_test = [("pooled", g[value_col].dropna())]
        elif group_by == "date_type":
            groups_to_test = [
                (dt, sub[value_col].dropna()) for dt, sub in g.groupby("date_type")
            ]
        elif group_by == "day_of_week":
            groups_to_test = [
                (int(dow), sub[value_col].dropna())
                for dow, sub in g.groupby("day_of_week")
            ]
        else:
            raise ValueError(
                "group_by must be one of 'pooled', 'date_type', 'day_of_week'"
            )

        for label, series in groups_to_test:
            if series.dropna().empty:
                # skip empty groups
                continue
            res = _normality_tests(series, alpha=alpha)
            # attach station/group metadata
            row = {"station_id": station_id, "group_label": label}
            row.update(res)
            records.append(row)

            if plot:
                _qq_plot(series, title=f"QQ plot station {station_id} — group {label}")

    return pd.DataFrame.from_records(records)
