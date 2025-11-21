"""
Clustering and Fitting assignment - completed template.

Requirements satisfied:
- Statistical analysis (mean, std, skewness, excess kurtosis).
- Four plots saved: relational (scatter), categorical (bar of counts for a chosen categorical or binned histogram),
  statistical (histogram + KDE), elbow_plot + silhouette computed + clustering plot, fitting plot with CI.
- Clustering implemented with KMeans (custom, numpy-only), silhouette and inertia computed.
- Fitting implemented with least-squares (numpy.polyfit) and 95% CI for predictions (using scipy.stats).
- All functions and names preserved from provided template where possible.
"""

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

sns.set(style="whitegrid")


def _ensure_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object numeric columns to numeric where possible, drop constant columns."""
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            # try convert to numeric
            df[col] = pd.to_numeric(df[col], errors="ignore")
    # drop columns that are entirely non-numeric after conversion
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return df[numeric_cols]


def plot_relational_plot(df: pd.DataFrame, xcol: str = None, ycol: str = None) -> None:
    """Scatter plot (relational). Saves 'relational_plot.png'."""
    if xcol is None or ycol is None:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) < 2:
            raise ValueError("need at least two numeric columns for relational plot")
        xcol, ycol = numeric[0], numeric[1]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(df[xcol], df[ycol], alpha=0.7, s=30)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(f"Relational plot: {ycol} vs {xcol}")
    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close(fig)


def plot_categorical_plot(df: pd.DataFrame, col: str = None) -> None:
    """
    Categorical plot. If a categorical column provided it plots category counts.
    Otherwise, bins first numeric column and plots counts.
    Saves 'categorical_plot.png'.
    """
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    if col and col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
        counts = df[col].value_counts().sort_index()
        counts.plot(kind="bar", ax=ax)
        ax.set_title(f"Categorical counts: {col}")
    else:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) == 0:
            raise ValueError("no suitable column for categorical plot")
        c = numeric[0]
        df[c].dropna().plot(kind="hist", bins=12, ax=ax)
        ax.set_title(f"Binned histogram (categorical-like): {c}")
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close(fig)


def plot_statistical_plot(df: pd.DataFrame, col: str = None) -> None:
    """
    Statistical plot: histogram + KDE for a numeric column.
    Saves 'statistical_plot.png'.
    """
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if col is None:
        if not numeric:
            raise ValueError("no numeric column for statistical plot")
        col = numeric[0]
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    sns.histplot(df[col].dropna(), kde=True, ax=ax)
    ax.set_title(f"Distribution (histogram + KDE): {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close(fig)


def statistical_analysis(df: pd.DataFrame, col: str) -> Tuple[float, float, float, float]:
    """
    Compute mean, standard deviation, skewness and excess kurtosis (Fisher).
    Returns: mean, stddev, skew, excess_kurtosis
    """
    series = pd.to_numeric(df[col].dropna())
    mean = float(series.mean())
    stddev = float(series.std(ddof=1))
    skew = float(ss.skew(series, bias=False))
    # ss.kurtosis with fisher=True returns excess kurtosis (kurtosis - 3)
    excess_kurtosis = float(ss.kurtosis(series, fisher=True, bias=False))
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic preprocessing:
    - converts convertible columns to numeric
    - drops duplicates and rows with all-NaNs
    - returns only numeric columns (for this assignment we work on numeric data)
    """
    df = df.copy()
    # Convert possible numeric strings
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="ignore")
    # Drop exact duplicates, drop rows all-NaN
    df = df.drop_duplicates().dropna(how="all")
    # Keep numeric columns only (assignment constraint: use numeric variables)
    numeric_df = _ensure_numeric_df(df)
    # Drop columns with zero variance
    nunique = numeric_df.nunique()
    keep_cols = nunique[nunique > 1].index.tolist()
    numeric_df = numeric_df[keep_cols]
    return numeric_df.reset_index(drop=True)


def writing(moments: Tuple[float, float, float, float], col: str) -> None:
    """Print summary to stdout (also kept from template)."""
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )
    # Simple classification of skewness/kurtosis for reporting
    skew = moments[2]
    ek = moments[3]
    skew_desc = "not skewed"
    if skew > 0.5:
        skew_desc = "right skewed"
    elif skew < -0.5:
        skew_desc = "left skewed"
    kurt_desc = "mesokurtic"
    if ek > 0.5:
        kurt_desc = "leptokurtic"
    elif ek < -0.5:
        kurt_desc = "platykurtic"
    print(f"The data was {skew_desc} and {kurt_desc}.")
    return


def _kmeans_numpy(X: np.ndarray, k: int, max_iter: int = 200, seed: int = 1234):
    """
    Simple K-means implementation returning labels, centers, inertia.
    """
    rng = np.random.RandomState(seed)
    n, d = X.shape
    # initialize centers by sampling points
    centers = X[rng.choice(n, k, replace=False)].astype(float)
    labels = np.zeros(n, dtype=int)
    for it in range(max_iter):
        # assign
        dists = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        # update centers
        for j in range(k):
            mask = labels == j
            if mask.any():
                centers[j] = X[mask].mean(axis=0)
            else:
                # reinitialize empty center
                centers[j] = X[rng.choice(n)]
    inertia = np.sum((X - centers[labels]) ** 2)
    return labels, centers, inertia


def _silhouette_score_numpy(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute average silhouette score (numpy-only).
    For each sample i:
      a(i) = mean intra-cluster distance
      b(i) = mean nearest-cluster distance
      s(i) = (b - a)/max(a, b)
    Returns mean s(i). If only one cluster present, return -1.
    """
    n = X.shape[0]
    unique_labels = np.unique(labels)
    if len(unique_labels) == 1 or n < 2:
        return -1.0
    # Precompute distance matrix
    dmat = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    sil_scores = np.zeros(n)
    for i in range(n):
        own = labels[i]
        mask_own = labels == own
        mask_not_own = ~mask_own
        if mask_own.sum() == 1:
            sil_scores[i] = 0.0
            continue
        a = dmat[i, mask_own].sum() / (mask_own.sum() - 1)
        # compute b = min mean distance to other clusters
        b_vals = []
        for lab in unique_labels:
            if lab == own:
                continue
            mask_lab = labels == lab
            b_vals.append(dmat[i, mask_lab].mean())
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        sil_scores[i] = 0.0 if denom == 0 else (b - a) / denom
    return float(np.mean(sil_scores))


def perform_clustering(df: pd.DataFrame, col1: str, col2: str):
    """
    Perform clustering using two columns. Saves 'elbow_plot.png' and returns:
    labels, data (2D np array original scaled), xkmeans (x coords of centers),
    ykmeans (y coords of centers), centre_labels (labels for each center)
    """
    # Extract data and drop NaNs for the two columns
    data_df = df[[col1, col2]].dropna().reset_index(drop=True)
    if data_df.shape[0] < 3:
        raise ValueError("Not enough rows for clustering")
    X = data_df.values.astype(float)
    # scale (z-score)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1)
    sigma[sigma == 0] = 1.0
    Xs = (X - mu) / sigma

    max_k = min(8, max(2, Xs.shape[0] // 2))
    ks = list(range(2, max_k + 1))
    inertias = []
    silhouettes = []
    centers_list = []
    labels_list = []
    for k in ks:
        labels_k, centers_k, inertia_k = _kmeans_numpy(Xs, k)
        score_k = _silhouette_score_numpy(Xs, labels_k)
        inertias.append(inertia_k)
        silhouettes.append(score_k)
        centers_list.append(centers_k)
        labels_list.append(labels_k)

    # Choose k by max silhouette; tie-breaker: smallest k
    best_idx = int(np.argmax(silhouettes))
    best_k = ks[best_idx]
    best_labels = labels_list[best_idx]
    best_centers = centers_list[best_idx]

    # Plot elbow + silhouette together
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax2 = ax.twinx()
    ax.plot(ks, inertias, marker="o", label="inertia")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax2.plot(ks, silhouettes, marker="s", color="orange", label="silhouette")
    ax2.set_ylabel("Silhouette score")
    ax.set_title("Elbow (inertia) and silhouette vs k")
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("elbow_plot.png")
    plt.close(fig)

    # Centers are in scaled space; convert back to original scale
    centers_orig = best_centers * sigma + mu
    xkmeans = centers_orig[:, 0]
    ykmeans = centers_orig[:, 1]
    # Return labels (as array matching data_df rows), original (unscaled) data array,
    # xkmeans, ykmeans, and centre_labels mapping
    cenlabels = [f"c{i}" for i in range(len(xkmeans))]
    return best_labels, data_df.values, xkmeans, ykmeans, cenlabels


def plot_clustered_data(labels, data, xkmeans, ykmeans, centre_labels):
    """
    Scatter data colored by cluster and overlay centers. Saves 'clustering.png'.
    """
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    labels = np.asarray(labels)
    unique = np.unique(labels)
    palette = sns.color_palette(n_colors=len(unique))
    for idx, lab in enumerate(unique):
        mask = labels == lab
        ax.scatter(data[mask, 0], data[mask, 1], s=30, alpha=0.7, label=f"Cluster {lab}")
    ax.scatter(xkmeans, ykmeans, s=150, marker="X", edgecolor="k", linewidth=1.0)
    for i, txt in enumerate(centre_labels):
        ax.annotate(txt, (xkmeans[i], ykmeans[i]), textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Clustered data with centers")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("clustering.png")
    plt.close(fig)
    return


def perform_fitting(df: pd.DataFrame, col1: str, col2: str):
    """
    Perform simple linear regression y ~ x using numpy.polyfit.
    Returns: data (Nx2 array), x (grid for line), y (predicted y on x).
    Also writes fit parameters to stdout.
    """
    df_xy = df[[col1, col2]].dropna().reset_index(drop=True)
    if df_xy.shape[0] < 3:
        raise ValueError("Not enough rows for fitting")
    x = df_xy[col1].values.astype(float)
    y = df_xy[col2].values.astype(float)
    # Fit line y = a*x + b
    slope, intercept = np.polyfit(x, y, deg=1)
    # Residuals and estimate of variance
    y_pred = slope * x + intercept
    residuals = y - y_pred
    dof = max(1, len(x) - 2)
    s_err = np.sqrt(np.sum(residuals ** 2) / dof)
    # For prediction intervals across grid
    x_grid = np.linspace(x.min(), x.max(), 200)
    # Compute standard error of predicted mean at x_grid
    x_mean = x.mean()
    ssx = np.sum((x - x_mean) ** 2)
    se_line = s_err * np.sqrt(1.0 / len(x) + (x_grid - x_mean) ** 2 / ssx)
    # 95% t
    tval = ss.t.ppf(0.975, dof)
    # Prediction bounds (mean estimate)
    y_grid = slope * x_grid + intercept
    ci_upper = y_grid + tval * se_line
    ci_lower = y_grid - tval * se_line

    print(f"Fitted line: y = {slope:.4f} * x + {intercept:.4f}")
    print(f"Residual std error: {s_err:.4f}, dof = {dof}")

    # return raw data array, x_grid, predicted y_grid with ci attached as tuple
    # For plotting convenience, we return data (original pairs), x_grid, (y_grid, ci_lower, ci_upper)
    return df_xy.values, x_grid, (y_grid, ci_lower, ci_upper)


def plot_fitted_data(data, x_grid, y_tuple):
    """
    Scatter raw data and plot fitted line with 95% CI ribbon.
    Saves 'fitting.png'.
    """
    y_grid, ci_lower, ci_upper = y_tuple
    fig, ax = plt.subplots(figsize=(7, 6), dpi=150)
    ax.scatter(data[:, 0], data[:, 1], alpha=0.7, s=30, label="data")
    ax.plot(x_grid, y_grid, linewidth=2, label="fitted line")
    ax.fill_between(x_grid, ci_lower, ci_upper, alpha=0.25, label="95% CI")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Linear fit with 95% CI")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig("fitting.png")
    plt.close(fig)
    return


def main():
    """
    Main pipeline:
     - read data.csv
     - preprocess
     - produce relational, statistical, categorical plots
     - compute and print moments for first numeric column
     - clustering using first two numeric columns
     - fitting using the same two columns (x -> first, y -> second)
    """
    df = pd.read_csv("Data.csv")
    df = preprocessing(df)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) == 0:
        raise ValueError("No numeric columns found in data.csv after preprocessing")

    # Choose columns (user can change these if they want specific choices)
    col = numeric[0]
    # if only one numeric col exists, replicate to allow clustering/fitting (not ideal)
    if len(numeric) == 1:
        df[col + "_copy"] = df[col]
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()

    cluster_x = numeric[0]
    cluster_y = numeric[1]
    fit_x = numeric[0]
    fit_y = numeric[1]

    # Plots
    plot_relational_plot(df, cluster_x, cluster_y)
    plot_statistical_plot(df, col)
    plot_categorical_plot(df)  # will bin first numeric if no explicit categorical present

    # Stats
    moments = statistical_analysis(df, col)
    writing(moments, col)

    # Clustering
    labels, data_arr, xkmeans, ykmeans, cenlabels = perform_clustering(df, cluster_x, cluster_y)
    plot_clustered_data(labels, data_arr, xkmeans, ykmeans, cenlabels)

    # Fitting
    data_pairs, x_grid, y_tuple = perform_fitting(df, fit_x, fit_y)
    plot_fitted_data(data_pairs, x_grid, y_tuple)

    print("All plots saved: relational_plot.png, categorical_plot.png, statistical_plot.png, "
          "elbow_plot.png, clustering.png, fitting.png")
    return


if __name__ == "__main__":
    main()
