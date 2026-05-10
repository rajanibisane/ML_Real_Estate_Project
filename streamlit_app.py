import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

NUMERIC_FEATURES = [
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


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, engine="python")
    df.columns = df.columns.str.strip()
    if "sale_price" in df.columns:
        df["sale_price"] = (
            df["sale_price"].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def build_feature_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = df[features].copy()
    X = X.dropna()
    scaler = StandardScaler()
    return X, scaler.fit_transform(X)


def plot_elbow(X_scaled: np.ndarray, max_k: int = 8) -> plt.Figure:
    inertias = []
    ks = list(range(2, max_k + 1))
    for k in ks:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        model.fit(X_scaled)
        inertias.append(model.inertia_)
    fig, ax = plt.subplots()
    ax.plot(ks, inertias, marker="o", linewidth=2)
    ax.set_title("K-Means Elbow Method")
    ax.set_xlabel("Number of clusters")
    ax.set_ylabel("Inertia")
    ax.set_xticks(ks)
    return fig


def plot_dendrogram(X_scaled: np.ndarray) -> plt.Figure:
    Z = linkage(X_scaled, method="ward")
    fig, ax = plt.subplots(figsize=(12, 5))
    dendrogram(Z, truncate_mode="level", p=5, ax=ax)
    ax.set_title("Hierarchical Clustering Dendrogram")
    ax.set_xlabel("Sample index or cluster size")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig


def plot_cluster_comparison(df: pd.DataFrame, feature_selection: list[str]) -> plt.Figure:
    cluster_means = df.groupby("cluster")[feature_selection].mean().reset_index()
    melted = cluster_means.melt(id_vars="cluster", var_name="feature", value_name="average")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=melted, x="feature", y="average", hue="cluster", ax=ax)
    ax.set_title("Average Selected Features by Cluster")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Average Value")
    ax.legend(title="Cluster")
    plt.xticks(rotation=30)
    fig.tight_layout()
    return fig


def plot_categorical_cluster_counts(df: pd.DataFrame, category: str, top_n: int = 10) -> plt.Figure:
    if category not in df.columns:
        raise ValueError(f"Category {category} does not exist in data.")
    top_categories = df[category].value_counts().nlargest(top_n).index
    plot_df = df[df[category].isin(top_categories)].copy()
    counts = plot_df.groupby([category, "cluster"]).size().reset_index(name="count")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=counts, x=category, y="count", hue="cluster", ax=ax)
    ax.set_title(f"Cluster counts by top {top_n} {category}")
    ax.set_xlabel(category.title())
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def plot_price_by_category(df: pd.DataFrame, category: str, top_n: int = 10) -> plt.Figure:
    if category not in df.columns or "sale_price" not in df.columns:
        raise ValueError("Required columns are missing for price-by-category plot.")
    top_categories = df[category].value_counts().nlargest(top_n).index
    plot_df = df[df[category].isin(top_categories)].copy()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=plot_df, x=category, y="sale_price", ax=ax)
    ax.set_title(f"Sale price distribution by top {top_n} {category}")
    ax.set_xlabel(category.title())
    ax.set_ylabel("Sale Price")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
    fig.tight_layout()
    return fig


def fit_kmeans(X_scaled: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return labels, score


def fit_hierarchical(X_scaled: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    return labels, score


def main():
    st.markdown(
        """
        <style>
        .dashboard-title {
            font-size: 3rem;
            font-weight: 700;
            color: #1f4f8b;
            margin: 0;
        }
        .dashboard-subtitle {
            font-size: 1.15rem;
            color: #4d5666;
            margin-top: 0.5rem;
            margin-bottom: 1.5rem;
            line-height: 1.5;
        }
        .header-box {
            background: linear-gradient(135deg, #f7fbff 0%, #e6eef9 100%);
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 16px 40px rgba(31, 79, 139, 0.12);
        }
        </style>
        <div class="header-box">
            <h1 class="dashboard-title">Real Estate Clustering Explorer</h1>
            <p class="dashboard-subtitle">Use interactive machine learning to uncover property clusters, sales trends, and customer segments with polished visual insights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    data_load_state = st.text("Loading data...")
    df = load_data("Real_Estate_Project.csv")
    data_load_state.text("Data loaded successfully.")

    st.sidebar.header("Model controls")
    algorithm = st.sidebar.selectbox("Clustering algorithm", ["K-Means", "Hierarchical"])
    k = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=4)
    feature_selection = st.sidebar.multiselect(
        "Features for clustering",
        options=[col for col in NUMERIC_FEATURES if col in df.columns],
        default=[col for col in NUMERIC_FEATURES if col in ["floor_area_sqft", "sale_price", "Sacled Satsifaction Score", "satisfaction"] and col in df.columns],
    )

    if len(feature_selection) < 2:
        st.warning("Select at least 2 features for clustering.")
        return

    st.subheader("Dataset preview")
    st.write(df.head(10))

    st.subheader("Feature summary")
    st.write(df[feature_selection].describe())

    X, X_scaled = build_feature_matrix(df, feature_selection)

    with st.expander("Cluster analysis plots"):
        if algorithm == "K-Means":
            st.pyplot(plot_elbow(X_scaled, max_k=8))
        st.pyplot(plot_dendrogram(X_scaled))

    if algorithm == "K-Means":
        labels, score = fit_kmeans(X_scaled, k)
    else:
        labels, score = fit_hierarchical(X_scaled, k)

    X["cluster"] = labels
    df["cluster"] = labels

    st.subheader("Cluster quality")
    st.write(f"Silhouette score: {score:.4f}")

    st.subheader("Cluster counts")
    st.write(pd.Series(labels).value_counts().sort_index())

    st.subheader("Cluster summary")
    st.write(df.groupby("cluster")[feature_selection].mean().round(3))

    st.subheader("Comparative charts")
    cluster_counts = pd.Series(labels).sort_index()
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Cluster size comparison**")
        st.bar_chart(cluster_counts)
    with col2:
        st.markdown("**Feature means by cluster**")
        st.pyplot(plot_cluster_comparison(df, feature_selection))

    with st.expander("Region and country comparative charts"):
        if "country" in df.columns:
            st.markdown("**Country-level cluster counts**")
            st.pyplot(plot_categorical_cluster_counts(df, "country", top_n=8))
            if "sale_price" in df.columns:
                st.markdown("**Sale price distribution by country**")
                st.pyplot(plot_price_by_category(df, "country", top_n=8))
        if "region" in df.columns:
            st.markdown("**Region-level cluster counts**")
            st.pyplot(plot_categorical_cluster_counts(df, "region", top_n=8))
            if "sale_price" in df.columns:
                st.markdown("**Sale price distribution by region**")
                st.pyplot(plot_price_by_category(df, "region", top_n=8))

    if "floor_area_sqft" in feature_selection and "sale_price" in feature_selection:
        st.subheader("Cluster scatter plot")
        fig, ax = plt.subplots()
        sns.scatterplot(
            data=df,
            x="floor_area_sqft",
            y="sale_price",
            hue="cluster",
            palette="tab10",
            alpha=0.8,
            ax=ax,
            edgecolor="k",
        )
        ax.set_title("Clusters by Floor Area and Sale Price")
        st.pyplot(fig)

    st.subheader("Cluster membership sample")
    st.write(df[[*feature_selection, "cluster"]].head(20))

    st.markdown(
        "---\nRun this app with `streamlit run streamlit_app.py` from the project folder."
    )


if __name__ == "__main__":
    main()
