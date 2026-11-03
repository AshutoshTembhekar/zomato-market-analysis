import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 1. Ingest & Clean Data
try:
    df = pd.read_csv("Zomato Restaurant Dataset.csv", encoding="utf-8")
except UnicodeDecodeError:
    df = pd.read_csv("Zomato Restaurant Dataset.csv", encoding="latin-1")

df.columns = (
    df.columns.str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
)

cost_col = [c for c in df.columns if "cost" in c or "average_cost" in c][0]
rating_col = [c for c in df.columns if "rating" in c or "aggregate_rating" in c][0]
votes_col = [c for c in df.columns if "votes" in c][0]
delivery_col = [c for c in df.columns if "online_delivery" in c][0]
booking_col = [c for c in df.columns if "table_booking" in c][0]

# Numeric sanitization
if df[cost_col].dtype == object:
    df[cost_col] = (
        df[cost_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"(\d+)")
        .astype(float)
    )

if df[rating_col].dtype == object:
    df[rating_col] = pd.to_numeric(
        df[rating_col].astype(str).str.split("/").str[0].str.strip(),
        errors="coerce",
    )

df[votes_col] = pd.to_numeric(df[votes_col], errors="coerce").fillna(0)

# Filter valid NCR records
geo_df = df.dropna(subset=[cost_col, rating_col, "latitude", "longitude"]).copy()
ncr_df = geo_df[
    (geo_df["country_code"] == 1)
    & (geo_df[cost_col] > 0)
    & (geo_df[rating_col] > 0)
    & (geo_df["latitude"].between(28.2, 28.9))
    & (geo_df["longitude"].between(76.8, 77.5))
].copy()

# 2. Bayesian Quality Calibration
C = ncr_df[rating_col].mean()
m = ncr_df[votes_col].quantile(0.60)
ncr_df["weighted_rating"] = (
    (ncr_df[votes_col] / (ncr_df[votes_col] + m)) * ncr_df[rating_col]
    + (m / (ncr_df[votes_col] + m)) * C
)

# 3. Standardize Clustering
cluster_features = ["latitude", "longitude", cost_col, "weighted_rating"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ncr_df[cluster_features])
X_scaled[:, 0] *= 1.75
X_scaled[:, 1] *= 1.75

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
ncr_df["cluster_id"] = kmeans.fit_predict(X_scaled)

# Map human-readable cluster names
cluster_labels = {
    4: "Central Premium Core",
    0: "Southwest Corporate (Gurgaon)",
    3: "South Delhi & Student Hubs",
    1: "North & West Belt",
    2: "East NCR & Trans-Yamuna",
}
ncr_df["cluster_name"] = ncr_df["cluster_id"].map(cluster_labels)

# Clean boolean flags
ncr_df["online_delivery"] = ncr_df[delivery_col].astype(str).str.strip().str.capitalize()
ncr_df["table_booking"] = ncr_df[booking_col].astype(str).str.strip().str.capitalize()

# 4. Cluster-Level Operational Aggregation
cluster_ops = (
    ncr_df.groupby(["cluster_id", "cluster_name"])
    .agg(
        total_venues=("restaurant_name", "count"),
        delivery_pct=("online_delivery", lambda x: (x == "Yes").mean() * 100),
        booking_pct=("table_booking", lambda x: (x == "Yes").mean() * 100),
        mean_cost=(cost_col, "mean"),
        mean_rating=(rating_col, "mean"),
        weighted_rating=("weighted_rating", "mean"),
        mean_votes=(votes_col, "mean"),
    )
    .reset_index()
)

print("\n--- Operational Adoption by Spatial Cluster ---")
print(
    cluster_ops[
        [
            "cluster_name",
            "total_venues",
            "delivery_pct",
            "booking_pct",
            "mean_cost",
            "mean_rating",
            "mean_votes",
        ]
    ].to_string(index=False)
)

# 5. Channel Impact Breakdown (Yes vs No across Metrics)
delivery_impact = (
    ncr_df.groupby("online_delivery")
    .agg(
        venues=("restaurant_name", "count"),
        avg_cost=(cost_col, "mean"),
        avg_rating=(rating_col, "mean"),
        avg_votes=(votes_col, "mean"),
    )
    .reset_index()
)

booking_impact = (
    ncr_df.groupby("table_booking")
    .agg(
        venues=("restaurant_name", "count"),
        avg_cost=(cost_col, "mean"),
        avg_rating=(rating_col, "mean"),
        avg_votes=(votes_col, "mean"),
    )
    .reset_index()
)

print("\n--- Macro Impact of Online Delivery ---")
print(delivery_impact.to_string(index=False))

print("\n--- Macro Impact of Table Booking ---")
print(booking_impact.to_string(index=False))

# 6. Generate Comparative Visualization
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.set_theme(style="whitegrid")

# Plot 1: Online Delivery vs Table Booking Adoption Rate by Cluster
plot_df = cluster_ops.sort_values(by="mean_cost", ascending=False)
x = np.arange(len(plot_df))
width = 0.35

axes[0].bar(x - width / 2, plot_df["delivery_pct"], width, label="Online Delivery %", color="#2ecc71")
axes[0].bar(x + width / 2, plot_df["booking_pct"], width, label="Table Booking %", color="#9b59b6")
axes[0].set_xticks(x)
axes[0].set_xticklabels(plot_df["cluster_name"], rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("Adoption Rate (%)")
axes[0].set_title("Operational Adoption Rates Across Spatial Clusters", weight="bold")
axes[0].legend()

# Plot 2: Rating Premium by Operational Channel
channel_comparison = pd.DataFrame(
    {
        "Operational Feature": [
            "No Online Delivery",
            "With Online Delivery",
            "No Table Booking",
            "With Table Booking",
        ],
        "Mean Rating": [
            ncr_df[ncr_df["online_delivery"] == "No"][rating_col].mean(),
            ncr_df[ncr_df["online_delivery"] == "Yes"][rating_col].mean(),
            ncr_df[ncr_df["table_booking"] == "No"][rating_col].mean(),
            ncr_df[ncr_df["table_booking"] == "Yes"][rating_col].mean(),
        ],
        "Mean Votes": [
            ncr_df[ncr_df["online_delivery"] == "No"][votes_col].mean(),
            ncr_df[ncr_df["online_delivery"] == "Yes"][votes_col].mean(),
            ncr_df[ncr_df["table_booking"] == "No"][votes_col].mean(),
            ncr_df[ncr_df["table_booking"] == "Yes"][votes_col].mean(),
        ],
    }
)

# Clean barplot without deprecation warning
sns.barplot(
    data=channel_comparison,
    x="Operational Feature",
    y="Mean Rating",
    hue="Operational Feature",
    palette=["#e74c3c", "#2ecc71", "#e67e22", "#3498db"],
    legend=False,
    ax=axes[1],
)
axes[1].set_ylim(2.5, 4.0)
axes[1].set_title("Rating Premium by Channel Capability", weight="bold")
axes[1].set_ylabel("Average Rating (1 - 5)")
axes[1].tick_params(axis="x", rotation=15)

# Use standard text 'stars' instead of unicode glyphs to prevent Arial font warnings
for p in axes[1].patches:
    axes[1].annotate(
        f"{p.get_height():.2f}",
        (p.get_x() + p.get_width() / 2.0, p.get_height()),
        ha="center",
        va="center",
        xytext=(0, 6),
        textcoords="offset points",
        weight="bold",
    )

plt.tight_layout()
plt.savefig("operational_impact_clusters.png", dpi=300)
print("\nSaved chart: operational_impact_clusters.png")

# Export CSV artifact
cluster_ops.to_csv("cluster_operational_metrics.csv", index=False)
print("Saved summary: cluster_operational_metrics.csv")