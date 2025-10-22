from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def count_multiple_checkins(file_path: str):
    """
    Analyze and visualize the number of check-ins per user.

    - Counts how many unique Numero_Tarjeta values have multiple check-ins.
    - Plots the distribution of number of check-ins for those users, showing mean and std.
    """
    if not Path(file_path).exists():
        print(f"❌ File not found: {file_path}")
        return

    # Load data
    df = pd.read_csv(file_path)

    if "Numero_Tarjeta" not in df.columns:
        print("❌ 'Numero_Tarjeta' column not found in file.")
        return

    # Count occurrences per card
    counts = df["Numero_Tarjeta"].value_counts()

    # Filter cards with more than one check-in
    repeated_cards = counts[counts > 1]

    print(f"📅 File: {file_path}")
    print(f"💳 Total unique cards: {counts.shape[0]}")
    print(f"🔁 Cards with >1 check-in: {repeated_cards.shape[0]}")
    print(f"📈 Total repeated check-in records: {df.shape[0] - counts.shape[0]}")

    # Summary stats
    mean_val = repeated_cards.mean()
    std_val = repeated_cards.std()

    print("\n📊 Distribution summary:")
    print(repeated_cards.describe())

    # --- Plot distribution ---
    plt.figure(figsize=(10, 5))
    plt.hist(
        repeated_cards,
        bins=range(2, repeated_cards.max() + 2),
        edgecolor="black",
        alpha=0.7,
    )
    plt.title("Distribution of Number of Check-ins per Card (for users with >1)")
    plt.xlabel("Number of Check-ins")
    plt.ylabel("Number of Users")
    plt.grid(True, linestyle="--", alpha=0.6)

    # Annotate mean and std in the plot
    text_str = f"Mean = {mean_val:.2f}\nStd Dev = {std_val:.2f}"
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

    return repeated_cards


if __name__ == "__main__":
    file_path = "data/check_ins/daily/20251014.csv"
    repeated_cards = count_multiple_checkins(file_path)
