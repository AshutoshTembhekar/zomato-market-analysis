import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# Set aesthetic styling
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.autolayout": True, "font.sans-serif": "DejaVu Sans"})

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

# Filter valid geospatial coordinates (remove 0.0 / null lat-longs)
geo_df = df.dropna(subset=[cost_col, rating_col, "latitude", "longitude"]).copy()
geo_df = geo_df[
    (geo_df[cost_col] > 0)
    & (geo_df[rating_col] > 0)
    & (geo_df["latitude"].abs() > 0.1)
    & (geo_df["longitude"].abs() > 0.1)
].copy()

# Focus on NCR (Delhi-NCR represents over 85% of active records in the dataset)
ncr_df = geo_df[
    (geo_df["country_code"] == 1)
    & (geo_df["latitude"].between(28.2, 28.9))
    & (geo_df["longitude"].between(76.8, 77.5))
].copy()

print(f"Filtered {len(ncr_df):,} active restaurants across the NCR metropolitan cluster.")

# 2. Bayesian Quality Calibration
C = ncr_df[rating_col].mean()
m = ncr_df[votes_col].quantile(0.60)
ncr_df["weighted_rating"] = (
    (ncr_df[votes_col] / (ncr_df[votes_col] + m)) * ncr_df[rating_col]
    + (m / (ncr_df[votes_col] + m)) * C
)

# 3. K-Means Spatial-Commercial Clustering
# Cluster on Spatial (lat, long) and Economic features (cost, quality)
cluster_features = ["latitude", "longitude", cost_col, "weighted_rating"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(ncr_df[cluster_features])

# Weight spatial coordinates higher so clusters remain geographically contiguous
X_scaled[:, 0] *= 1.75  # Latitude weight
X_scaled[:, 1] *= 1.75  # Longitude weight

k = 5
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
ncr_df["cluster_id"] = kmeans.fit_predict(X_scaled)

# 4. Cluster Profile Summary
cluster_profiles = (
    ncr_df.groupby("cluster_id")
    .agg(
        venue_count=("restaurant_name", "count"),
        mean_cost=(cost_col, "mean"),
        median_cost=(cost_col, "median"),
        mean_rating=(rating_col, "mean"),
        mean_weighted_rating=("weighted_rating", "mean"),
        top_localities=("locality", lambda x: ", ".join(x.value_counts().head(3).index)),
    )
    .reset_index()
)

print("\n--- Spatial-Commercial Cluster Profiles ---")
print(cluster_profiles.to_string(index=False))

# 5. High-Resolution Geospatial Mapping
fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot: Longitude vs Latitude
scatter = ax.scatter(
    x=ncr_df["longitude"],
    y=ncr_df["latitude"],
    c=ncr_df["cluster_id"],
    cmap="Spectral",
    s=ncr_df[cost_col] / 25,  # Size scaled by cost for two
    alpha=0.6,
    edgecolors="k",
    linewidths=0.3,
)

# Plot Cluster Centroids
centers = scaler.inverse_transform(kmeans.cluster_centers_)
for idx, center in enumerate(centers):
    lat_center, lon_center = center[0], center[1]
    cost_center = center[2]
    rating_center = center[3]
    ax.scatter(
        lon_center,
        lat_center,
        c="black",
        s=250,
        marker="X",
        edgecolors="white",
        linewidths=1.5,
        zorder=5,
    )
    ax.annotate(
        f"Cluster {idx}\nAvg: ₹{cost_center:,.0f} | {rating_center:.2f}★",
        (lon_center, lat_center),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=10,
        weight="bold",
        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.75),
        zorder=6,
    )

ax.set_title(
    "Geospatial Clustering: Restaurant Density, Spatial Spread & Spend Tiers",
    fontsize=14,
    weight="bold",
)
ax.set_xlabel("Longitude", fontsize=12)
ax.set_ylabel("Latitude", fontsize=12)
plt.tight_layout()

output_map = "geospatial_clusters_map_v2.png"
plt.savefig(output_map, dpi=300, bbox_inches="tight")
ncr_df.to_csv("zomato_geospatial_clustered.csv", index=False)
print("Saved clustered data: zomato_geospatial_clustered.csv")