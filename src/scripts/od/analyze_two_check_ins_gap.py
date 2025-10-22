from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def analyze_two_checkins(file_path: str):
    """
    Analyze users with exactly 2 check-ins:
    - Compute the time difference between the first and second check-in.
    - Plot the distribution of those time differences (in minutes),
      including mean and standard deviation in the plot.
    """
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path, parse_dates=["Fecha_Transaccion"])

    if "Numero_Tarjeta" not in df.columns or "Fecha_Transaccion" not in df.columns:
        print("❌ Missing required columns: 'Numero_Tarjeta' or 'Fecha_Transaccion'")
        return

    # Sort by card and transaction time
    df = df.sort_values(["Numero_Tarjeta", "Fecha_Transaccion"])

    # Count check-ins per card
    counts = df["Numero_Tarjeta"].value_counts()

    # Keep only cards with exactly two check-ins
    two_checkins_cards = counts[counts == 2].index
    df_two = df[df["Numero_Tarjeta"].isin(two_checkins_cards)]

    # Compute time difference between 1st and 2nd check-in for each card
    diffs = (
        df_two.groupby("Numero_Tarjeta")["Fecha_Transaccion"]
        .apply(lambda x: (x.iloc[1] - x.iloc[0]).total_seconds() / 60.0)  # minutes
        .reset_index(name="minutes_diff")
    )

    mean_diff = diffs["minutes_diff"].mean()
    std_diff = diffs["minutes_diff"].std()

    print(f"📊 Users with exactly 2 check-ins: {len(diffs)}")
    print(f"⏱️ Mean time difference: {mean_diff:.2f} minutes")
    print(f"📈 Standard deviation: {std_diff:.2f} minutes")

    # Plot the distribution
    plt.figure(figsize=(10, 5))
    plt.hist(diffs["minutes_diff"], bins=50, edgecolor="black", alpha=0.7)
    plt.title("Distribution of Time Differences Between Two Check-ins")
    plt.xlabel("Time Difference (minutes)")
    plt.ylabel("Number of Users")
    plt.grid(True, linestyle="--", alpha=0.6)

    # --- Add mean and std deviation text box ---
    text_str = f"Mean = {mean_diff:.2f} min\nStd Dev = {std_diff:.2f} min"
    plt.text(
        0.97,
        0.95,
        text_str,
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
    )

    plt.tight_layout()
    plt.show()

    return diffs


if __name__ == "__main__":
    file_path = "data/check_ins/daily/20251014.csv"
    analyze_two_checkins(file_path)
