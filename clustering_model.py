import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_clean_data(csv_path):
    df = pd.read_csv(csv_path, engine="python")
    df.columns = df.columns.str.strip()
    if "sale_price" in df.columns:
        df["sale_price"] = (
            df["sale_price"]
            .astype(str)
            .str.replace("\$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    numeric_cols = [
        "floor_area_sqft",
        "sale_price",
        "Individual",
        "Company",
        "Home",
        "Investment",
        "Country Encoding",
        "Region Encoding",
        "Sacled Satsifaction Score",
        "satisfaction",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[col for col in numeric_cols if col in df.columns])
    return df


def build_feature_matrix(df):
    features = [
        "floor_area_sqft",
        "sale_price",
        "Individual",
        "Company",
        "Home",
        "Investment",
        "Country Encoding",
        "Region Encoding",
        "Sacled Satsifaction Score",
        "satisfaction",
    ]
    features = [col for col in features if col in df.columns]
    X = df[features].copy()
    X = X.fillna(0)
    return X


def compute_optimal_k(X_scaled, k_min=2, k_max=7):
    scores = {}
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"K={k}: silhouette={score:.4f}")
    best_k = max(scores, key=scores.get)
    print(f"Best k by silhouette score: {best_k}")
    return best_k, scores


def plot_clusters(df, labels, output_path=None):
    df_plot = df.copy()
    df_plot["cluster"] = labels
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_plot,
        x="floor_area_sqft",
        y="sale_price",
        hue="cluster",
        palette="tab10",
        alpha=0.8,
        edgecolor="k",
        s=70,
    )
    plt.title("K-Means clusters: Floor Area vs Sale Price")
    plt.xlabel("Floor Area (sqft)")
    plt.ylabel("Sale Price")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()


def plot_dendrogram(X_scaled, output_path=None, truncate_mode="level", p=5):
    Z = linkage(X_scaled, method="ward")
    plt.figure(figsize=(12, 6))
    dendrogram(
        Z,
        truncate_mode=truncate_mode,
        p=p,
        show_leaf_counts=True,
        leaf_rotation=45,
        leaf_font_size=10,
    )
    plt.title("Hierarchical clustering dendrogram")
    plt.xlabel("Sample index or cluster size")
    plt.ylabel("Distance")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
    return Z


def label_encode_columns(df, columns):
    df_enc = df.copy()
    for col in columns:
        if col in df_enc.columns:
            encoder = LabelEncoder()
            df_enc[col] = encoder.fit_transform(df_enc[col].astype(str))
    return df_enc


def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_path, "Real_Estate_Project.csv")
    df = load_and_clean_data(csv_path)

    if df.empty:
        raise ValueError("No valid rows found after cleaning. Check the CSV and column names.")

    X = build_feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n--- Determining optimal K for K-Means ---")
    best_k, silhouette_scores = compute_optimal_k(X_scaled, k_min=2, k_max=7)

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    df["kmeans_cluster"] = kmeans_labels

    print("\n--- Fitting hierarchical clustering ---")
    hierarchical = AgglomerativeClustering(n_clusters=best_k, affinity="euclidean", linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    df["hierarchical_cluster"] = hierarchical_labels

    output_csv = os.path.join(base_path, "Real_Estate_Project_clustered.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved clustered dataset to: {output_csv}")

    plot_clusters(df, labels=kmeans_labels, output_path=os.path.join(base_path, "kmeans_clusters.png"))
    plot_dendrogram(X_scaled, output_path=os.path.join(base_path, "hierarchical_dendrogram.png"))

    print("\n--- Cluster counts ---")
    print(df["kmeans_cluster"].value_counts().sort_index())
    print(df["hierarchical_cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()
