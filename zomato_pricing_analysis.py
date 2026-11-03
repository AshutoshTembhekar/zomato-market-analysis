import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean visual styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "font.sans-serif": "DejaVu Sans"})

# 1. Load Data
# Handle potential encoding variances in international character strings
try:
    df = pd.read_csv("Zomato Restaurant Dataset.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("Zomato Restaurant Dataset.csv", encoding="latin-1")

print(f"Loaded dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")

# 2. Standardize Column Names
df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)
print("Standardized columns:\n", df.columns.tolist())

# Identify core columns (adjusting for common column name variations)
cost_col = [c for c in df.columns if "cost" in c or "average_cost" in c][0]
rating_col = [c for c in df.columns if "rating" in c or "aggregate_rating" in c][0]
votes_col = [c for c in df.columns if "votes" in c][0]
name_col = [c for c in df.columns if "name" in c or "restaurant_name" in c][0]

# 3. Clean Numeric Features
# Clean Cost: Remove commas if stored as string, convert to float
if df[cost_col].dtype == object:
    df[cost_col] = (
        df[cost_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")
        .astype(float)
    )

# Clean Rating: Handle text placeholders like 'NEW', '-', etc.
if df[rating_col].dtype == object:
    df[rating_col] = pd.to_numeric(
        df[rating_col].astype(str).str.split("/").str[0].str.strip(),
        errors="coerce",
    )

df[votes_col] = pd.to_numeric(df[votes_col], errors="coerce").fillna(0)

# Filter out zero/unrated records for market positioning
clean_df = df.dropna(subset=[cost_col, rating_col]).copy()
clean_df = clean_df[(clean_df[cost_col] > 0) & (clean_df[rating_col] > 0)].copy()

print(f"Active rated restaurants for positioning: {clean_df.shape[0]:,}")

# 4. Segment into Strategic Spend Brackets
# These thresholds can be adjusted based on the catalog distribution
cost_percentiles = clean_df[cost_col].quantile([0.33, 0.66, 0.90]).to_dict()
print("Cost percentiles (33%, 66%, 90%):", cost_percentiles)

clean_df["price_tier"] = pd.qcut(
    clean_df[cost_col],
    q=[0, 0.33, 0.66, 0.90, 1.0],
    labels=["1. Budget", "2. Mid-Range", "3. Premium", "4. Luxury"],
)

# 5. Bayesian Weighted Rating Calibration
# Formula: Weighted Rating = (v / (v + m)) * R + (m / (v + m)) * C
C = clean_df[rating_col].mean()       # Overall market mean rating
m = clean_df[votes_col].quantile(0.60) # Minimum votes threshold (60th percentile)

v = clean_df[votes_col]
R = clean_df[rating_col]

clean_df["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C
print(f"Market Average Rating (C): {C:.2f} | 60th Percentile Votes (m): {m:.0f}")

# 6. Value-for-Money Matrix Categorization
median_cost = clean_df[cost_col].median()
median_weighted_rating = clean_df["weighted_rating"].median()

def assign_quadrant(row):
    cost = row[cost_col]
    rating = row["weighted_rating"]
    if cost <= median_cost and rating >= median_weighted_rating:
        return "Value Champion (Hidden Gem)"
    elif cost > median_cost and rating >= median_weighted_rating:
        return "Premium Leader"
    elif cost <= median_cost and rating < median_weighted_rating:
        return "Budget Staple"
    else:
        return "Overrated / High Churn Risk"

clean_df["market_quadrant"] = clean_df.apply(assign_quadrant, axis=1)

print("\n--- Market Quadrant Distribution ---")
print(clean_df["market_quadrant"].value_counts(normalize=True).mul(100).round(1).astype(str) + "%")

# 7. Generate Positioning Scatter Plot
plt.figure(figsize=(12, 7))

# Plot quadrant boundaries
plt.axvline(median_cost, color="gray", linestyle="--", alpha=0.7, label=f"Median Cost (₹{median_cost:,.0f})")
plt.axhline(median_weighted_rating, color="gray", linestyle="--", alpha=0.7, label=f"Median Weighted Rating ({median_weighted_rating:.2f})")

palette = {
    "Value Champion (Hidden Gem)": "#2ecc71",
    "Premium Leader": "#3498db",
    "Budget Staple": "#95a5a6",
    "Overrated / High Churn Risk": "#e74c3c",
}

scatter = sns.scatterplot(
    data=clean_df,
    x=cost_col,
    y="weighted_rating",
    hue="market_quadrant",
    palette=palette,
    alpha=0.65,
    s=50,
    edgecolor=None,
)

plt.xscale("log")  # Log scale handles extreme luxury cost outliers gracefully
plt.title("Zomato Market Positioning: Cost for Two vs. Bayesian Quality Rating", fontsize=14, weight="bold")
plt.xlabel("Average Cost for Two (Log Scale)", fontsize=12)
plt.ylabel("Bayesian Weighted Rating (1 - 5)", fontsize=12)
plt.legend(title="Market Quadrant", loc="lower right", frameon=True, framealpha=0.9)
plt.tight_layout()

plt.savefig("pricing_positioning_matrix.png", dpi=300)
plt.close()
print("Saved chart: pricing_positioning_matrix.png")

# Export clean summary
clean_df.to_csv("zomato_market_positioning.csv", index=False)
print("Saved dataset: zomato_market_positioning.csv")

# 8. Extract Top Outliers for Executive Summary
print("\n" + "=" * 55)
print("  TOP 5 VALUE CHAMPIONS (Highest Bayesian Quality)")
print("=" * 55)
top_value = clean_df[clean_df["market_quadrant"] == "Value Champion (Hidden Gem)"].sort_values(
    by=["weighted_rating", votes_col], ascending=False
)[[name_col, "locality", cost_col, rating_col, "weighted_rating", votes_col]].head(5)
print(top_value.to_string(index=False))

print("\n" + "=" * 55)
print("  TOP 5 OVERRATED / HIGH CHURN RISKS (Highest Votes)")
print("=" * 55)
top_churn = clean_df[clean_df["market_quadrant"] == "Overrated / High Churn Risk"].sort_values(
    by=[votes_col, cost_col], ascending=False
)[[name_col, "locality", cost_col, rating_col, "weighted_rating", votes_col]].head(5)
print(top_churn.to_string(index=False))