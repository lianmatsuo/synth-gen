import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for subprocess execution
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings("ignore")

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)

# Ensure output is flushed immediately
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)

# Parse command line arguments
parser = argparse.ArgumentParser(
    description="Perform anomaly detection on MIMIC-III data"
)
parser.add_argument("--data", type=str, help="Path to data file (CSV or Parquet)")
args = parser.parse_args()

# Create output directory
output_dir = Path("eda/anomaly_stats")
output_dir.mkdir(parents=True, exist_ok=True)

# Load data
if args.data:
    data_path = Path(args.data)
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        # Try Parquet first, fallback to CSV
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path, low_memory=False)
else:
    # Default: use combined file if available, otherwise use ADMISSIONS
    mimic_combined = Path(
        "data/mimic-iii-clinical-database-1.4/combined/mimic_iii_combined.parquet"
    )
    if mimic_combined.exists():
        df = pd.read_parquet(mimic_combined)
    else:
        # Fallback to ADMISSIONS table (try Parquet first)
        mimic_dir = Path("data/mimic-iii-clinical-database-1.4")
        parquet_path = mimic_dir / "parquet" / "ADMISSIONS.parquet"
        csv_path = mimic_dir / "ADMISSIONS.csv"
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            raise FileNotFoundError(
                "No data file found. Please provide --data argument or ensure ADMISSIONS table exists."
            )

# Separate numeric and categorical columns
num_df = df.select_dtypes(include=["number"])
cat_cols_all = df.select_dtypes(include=["object"]).columns

# Filter categorical columns: skip those with too many unique values (high cardinality)
# This prevents memory issues with one-hot encoding
MAX_CATEGORIES_PER_COL = 50  # Maximum number of unique values per categorical column
cat_cols_to_encode = []
cat_cols_skipped = []

for col in cat_cols_all:
    unique_count = df[col].nunique()
    if unique_count <= MAX_CATEGORIES_PER_COL:
        cat_cols_to_encode.append(col)
    else:
        cat_cols_skipped.append((col, unique_count))

cat_cols_to_encode = pd.Index(cat_cols_to_encode)

print(f"Dataset shape: {df.shape}")
print(f"Total columns: {len(df.columns)}")
print(f"Numeric columns: {len(num_df.columns)}")
print(f"Categorical columns: {len(cat_cols_all)}")
print(f"  - Will one-hot encode: {len(cat_cols_to_encode)} columns")
if cat_cols_skipped:
    print(
        f"  - Skipping {len(cat_cols_skipped)} high-cardinality columns (> {MAX_CATEGORIES_PER_COL} unique values)"
    )
    for col, count in cat_cols_skipped[:5]:  # Show first 5
        print(f"    - {col}: {count} unique values")
    if len(cat_cols_skipped) > 5:
        print(f"    ... and {len(cat_cols_skipped) - 5} more")
print(f"\nNumeric columns being processed: {list(num_df.columns)}")
if len(cat_cols_to_encode) > 0:
    print(f"Categorical columns to one-hot encode: {list(cat_cols_to_encode)}")
sys.stdout.flush()

# One-hot encode categorical columns
cat_encoded_df = None
if len(cat_cols_to_encode) > 0:
    print("\n" + "=" * 60)
    print("One-hot encoding categorical columns...")
    print("=" * 60)
    try:
        # Convert to string and handle missing values
        cat_data = df[cat_cols_to_encode].copy()
        for col in cat_data.columns:
            cat_data[col] = cat_data[col].astype(str)
            cat_data[col] = cat_data[col].replace("nan", "missing")
            cat_data[col] = cat_data[col].fillna("missing")

        # One-hot encode
        encoder = OneHotEncoder(
            sparse_output=False, handle_unknown="ignore", drop="if_binary"
        )
        cat_encoded = encoder.fit_transform(cat_data)

        # Create DataFrame with meaningful column names
        feature_names = []
        for i, col in enumerate(cat_cols_to_encode):
            unique_vals = cat_data[col].unique()
            if len(unique_vals) == 2:
                # Binary: only one column needed (drop='if_binary' removes one)
                feature_names.append(f"{col}_{unique_vals[1]}")
            else:
                for val in unique_vals:
                    feature_names.append(f"{col}_{val}")

        # Adjust feature names if encoder dropped binary columns
        if len(feature_names) != cat_encoded.shape[1]:
            feature_names = encoder.get_feature_names_out(cat_cols_to_encode)

        cat_encoded_df = pd.DataFrame(
            cat_encoded, index=df.index, columns=feature_names
        )
        print(
            f"One-hot encoded {len(cat_cols_to_encode)} categorical columns into {cat_encoded_df.shape[1]} binary columns"
        )
        sys.stdout.flush()
    except Exception as e:
        print(f"Warning: Could not one-hot encode categorical columns: {e}")
        print("Continuing with numeric columns only...")
        sys.stdout.flush()
        cat_encoded_df = None

# Combine numeric and encoded categorical columns
if cat_encoded_df is not None:
    all_features_df = pd.concat([num_df, cat_encoded_df], axis=1)
    print(
        f"Combined dataset: {len(num_df.columns)} numeric + {cat_encoded_df.shape[1]} encoded categorical = {all_features_df.shape[1]} total features"
    )
else:
    all_features_df = num_df
    print(f"Using {len(num_df.columns)} numeric features only")
sys.stdout.flush()

results = {}

# 1. Z-scores (standardized scores)
print("\n" + "=" * 60)
print("1. Calculating Z-scores...")
print("=" * 60)
z_scores = pd.DataFrame(index=df.index)
processed_cols = []
skipped_cols = []
for col in num_df.columns:
    try:
        if num_df[col].std() != 0:  # Skip constant columns
            z_scores[f"{col}_zscore"] = (num_df[col] - num_df[col].mean()) / num_df[
                col
            ].std()
            processed_cols.append(col)
        else:
            skipped_cols.append(col)
    except Exception:
        skipped_cols.append(col)

if skipped_cols:
    print(
        f"  Skipped {len(skipped_cols)} columns (constant or error): {skipped_cols[:10]}..."
    )
    sys.stdout.flush()

results["z_scores"] = z_scores
print(f"Z-scores calculated for {len(processed_cols)} numeric columns")
sys.stdout.flush()

# Visualize Z-score distributions
if len(num_df.columns) > 0:
    print("Creating visualization: Z-score Distributions...")
    sys.stdout.flush()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sample_cols = num_df.columns[:4] if len(num_df.columns) >= 4 else num_df.columns

    print(f"  → Plotting distributions for {len(sample_cols)} columns...")
    sys.stdout.flush()
    for idx, col in enumerate(sample_cols):
        ax = axes[idx // 2, idx % 2]
        z_col = z_scores[f"{col}_zscore"]
        ax.hist(z_col.dropna(), bins=50, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="r", linestyle="--", label="Mean")
        ax.axvline(x=2, color="orange", linestyle="--", label="±2σ")
        ax.axvline(x=-2, color="orange", linestyle="--")
        ax.set_title(f"Z-scores: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Z-score")
        ax.set_ylabel("Frequency")
        ax.legend()
        ax.grid(True, alpha=0.3)

    print("  → Saving visualization to file...")
    sys.stdout.flush()
    plt.tight_layout()
    plt.savefig(output_dir / "z_scores_distributions.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved visualization: z_scores_distributions.png")
    sys.stdout.flush()

# 2. IQR (Interquartile Range)
print("\n" + "=" * 60)
print("2. Calculating IQR...")
print("=" * 60)
iqr_stats = pd.DataFrame(index=num_df.columns)
iqr_stats["Q1"] = num_df.quantile(0.25)
iqr_stats["Q3"] = num_df.quantile(0.75)
iqr_stats["IQR"] = iqr_stats["Q3"] - iqr_stats["Q1"]
iqr_stats["Lower_Bound"] = iqr_stats["Q1"] - 1.5 * iqr_stats["IQR"]
iqr_stats["Upper_Bound"] = iqr_stats["Q3"] + 1.5 * iqr_stats["IQR"]
results["iqr_stats"] = iqr_stats

# Count outliers using IQR method
outlier_counts = {}
for col in num_df.columns:
    lower = iqr_stats.loc[col, "Lower_Bound"]
    upper = iqr_stats.loc[col, "Upper_Bound"]
    outliers = ((num_df[col] < lower) | (num_df[col] > upper)).sum()
    outlier_counts[col] = outliers

iqr_stats["Outlier_Count"] = pd.Series(outlier_counts)
print(f"IQR calculated for {len(num_df.columns)} numeric columns")
print(f"Total outliers detected: {sum(outlier_counts.values())}")

# Visualize IQR outliers
if len(num_df.columns) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sample_cols = num_df.columns[:4] if len(num_df.columns) >= 4 else num_df.columns

    for idx, col in enumerate(sample_cols):
        ax = axes[idx // 2, idx % 2]
        data = num_df[col].dropna()
        lower = iqr_stats.loc[col, "Lower_Bound"]
        upper = iqr_stats.loc[col, "Upper_Bound"]

        ax.boxplot(data, vert=True)
        ax.axhline(y=lower, color="r", linestyle="--", alpha=0.7, label="Lower Bound")
        ax.axhline(y=upper, color="r", linestyle="--", alpha=0.7, label="Upper Bound")
        ax.set_title(f"IQR Outliers: {col}", fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "iqr_outliers.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved IQR outlier visualizations to {output_dir / 'iqr_outliers.png'}")

# 3. Quantile Ranks
print("\n" + "=" * 60)
print("3. Calculating Quantile Ranks...")
print("=" * 60)
quantile_ranks = pd.DataFrame(index=df.index)
for col in num_df.columns:
    quantile_ranks[f"{col}_quantile"] = num_df[col].rank(pct=True) * 100
results["quantile_ranks"] = quantile_ranks
print(f"Quantile ranks calculated for {len(num_df.columns)} numeric columns")

# Visualize quantile rank distributions
if len(num_df.columns) > 0:
    print("Creating visualization: Quantile Rank Distributions...")
    sys.stdout.flush()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    sample_cols = num_df.columns[:4] if len(num_df.columns) >= 4 else num_df.columns

    print(f"  → Plotting quantile ranks for {len(sample_cols)} columns...")
    sys.stdout.flush()
    for idx, col in enumerate(sample_cols):
        ax = axes[idx // 2, idx % 2]
        q_col = quantile_ranks[f"{col}_quantile"]
        ax.hist(q_col.dropna(), bins=20, edgecolor="black", alpha=0.7)
        ax.set_title(f"Quantile Ranks: {col}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Percentile Rank")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    print("  → Saving visualization to file...")
    sys.stdout.flush()
    plt.tight_layout()
    plt.savefig(
        output_dir / "quantile_ranks_distributions.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
    print("✓ Saved visualization: quantile_ranks_distributions.png")
    sys.stdout.flush()

# 4. Mahalanobis Distance
print("\n" + "=" * 60)
print("4. Calculating Mahalanobis Distance...")
print("=" * 60)
if len(all_features_df.columns) > 1:
    # Remove columns with constant values or too many NaNs
    valid_cols = []
    skipped_cols = []
    for col in all_features_df.columns:
        if (
            all_features_df[col].nunique() > 1
            and all_features_df[col].notna().sum() > len(df) * 0.5
        ):
            valid_cols.append(col)
        else:
            skipped_cols.append(col)

    if skipped_cols:
        print(
            f"  Skipping {len(skipped_cols)} columns for Mahalanobis (constant or >50% NaNs): {skipped_cols[:10]}..."
        )
        sys.stdout.flush()

    print(
        f"  Processing Mahalanobis distance for {len(valid_cols)} valid features (out of {len(all_features_df.columns)} total)..."
    )
    sys.stdout.flush()

    if len(valid_cols) > 1:
        num_clean = all_features_df[valid_cols].dropna()

        if len(num_clean) > 0:
            # Standardize data
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_clean)

            # Calculate covariance matrix
            cov_matrix = np.cov(num_scaled.T)

            # Calculate mean
            mean_vector = np.mean(num_scaled, axis=0)

            # Calculate Mahalanobis distance
            try:
                inv_cov_matrix = np.linalg.pinv(
                    cov_matrix
                )  # Use pseudo-inverse for stability
                mahal_distances = []

                for i in range(len(num_scaled)):
                    diff = num_scaled[i] - mean_vector
                    mahal_dist = np.sqrt(diff.T @ inv_cov_matrix @ diff)
                    mahal_distances.append(mahal_dist)

                mahal_distances = pd.Series(mahal_distances, index=num_clean.index)
                results["mahalanobis_distance"] = mahal_distances
                print(f"Mahalanobis distance calculated for {len(num_clean)} samples")

                # Visualize Mahalanobis distances
                plt.figure(figsize=(10, 6))
                plt.hist(mahal_distances, bins=50, edgecolor="black", alpha=0.7)
                plt.axvline(
                    x=mahal_distances.quantile(0.95),
                    color="r",
                    linestyle="--",
                    label="95th percentile",
                )
                plt.title(
                    "Mahalanobis Distance Distribution", fontsize=14, fontweight="bold"
                )
                plt.xlabel("Mahalanobis Distance")
                plt.ylabel("Frequency")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(
                    output_dir / "mahalanobis_distance.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
                print(
                    f"Saved Mahalanobis distance visualization to {output_dir / 'mahalanobis_distance.png'}"
                )
            except Exception as e:
                print(f"Error calculating Mahalanobis distance: {e}")
        else:
            print("Not enough valid data for Mahalanobis distance calculation")
    else:
        print("Not enough valid numeric columns for Mahalanobis distance")

# 5. kNN Distance
print("\n" + "=" * 60)
print("5. Calculating kNN Distance...")
print("=" * 60)
if len(all_features_df.columns) > 0:
    # Use same cleaned data as Mahalanobis
    if "valid_cols" in locals() and len(valid_cols) > 0:
        num_clean = all_features_df[valid_cols].dropna()
    else:
        # Recalculate valid columns
        valid_cols = [
            col for col in all_features_df.columns if all_features_df[col].nunique() > 1
        ]
        if len(valid_cols) > 0:
            num_clean = all_features_df[valid_cols].dropna()
        else:
            num_clean = pd.DataFrame()  # Empty DataFrame

    if len(num_clean) > 0 and len(num_clean) > 100:  # Need sufficient samples
        # Standardize
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_clean)

        # Use PCA to reduce dimensionality only if extremely high dimensional
        # Increased threshold to preserve more information
        if num_scaled.shape[1] > 200:
            print(
                f"  Reducing dimensions from {num_scaled.shape[1]} to 200 using PCA..."
            )
            sys.stdout.flush()
            pca = PCA(n_components=200)
            num_scaled = pca.fit_transform(num_scaled)
        else:
            print(
                f"  Using all {num_scaled.shape[1]} dimensions for kNN distance (no PCA reduction)"
            )
            sys.stdout.flush()

        # Calculate kNN distances (k=5)
        k = min(5, len(num_clean) - 1)
        nbrs = NearestNeighbors(n_neighbors=k + 1, metric="euclidean")
        nbrs.fit(num_scaled)
        distances, indices = nbrs.kneighbors(num_scaled)

        # Average distance to k nearest neighbors (excluding self)
        knn_distances = pd.Series(distances[:, 1:].mean(axis=1), index=num_clean.index)
        results["knn_distance"] = knn_distances
        print(f"kNN distance (k={k}) calculated for {len(num_clean)} samples")

        # Visualize kNN distances
        print("Creating visualization: kNN Distance Distribution...")
        sys.stdout.flush()
        print("  → Drawing histogram...")
        sys.stdout.flush()
        plt.figure(figsize=(10, 6))
        plt.hist(knn_distances, bins=50, edgecolor="black", alpha=0.7)
        plt.axvline(
            x=knn_distances.quantile(0.95),
            color="r",
            linestyle="--",
            label="95th percentile",
        )
        plt.title(f"kNN Distance Distribution (k={k})", fontsize=14, fontweight="bold")
        plt.xlabel("Average kNN Distance")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        print("  → Saving visualization to file...")
        sys.stdout.flush()
        plt.savefig(output_dir / "knn_distance.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Saved visualization: knn_distance.png")
        sys.stdout.flush()
    else:
        print("Not enough samples for kNN distance calculation")
else:
    print("Not enough valid features for kNN distance")

# 6. Off-the-shelf Anomaly Detectors
print("\n" + "=" * 60)
print("6. Running Off-the-shelf Anomaly Detectors...")
print("=" * 60)

if len(all_features_df.columns) > 0 and len(df) > 100:
    # Prepare data
    if "valid_cols" in locals() and len(valid_cols) > 0:
        print(
            f"  Processing anomaly detectors for {len(valid_cols)} valid features (out of {len(all_features_df.columns)} total)..."
        )
        sys.stdout.flush()
    elif len(all_features_df.columns) > 0:
        # If valid_cols wasn't defined, recalculate
        valid_cols = [
            col for col in all_features_df.columns if all_features_df[col].nunique() > 1
        ]
        skipped_cols = [
            col
            for col in all_features_df.columns
            if all_features_df[col].nunique() <= 1
        ]
        if skipped_cols:
            print(
                f"  Skipping {len(skipped_cols)} constant columns for anomaly detectors: {skipped_cols[:10]}..."
            )
            sys.stdout.flush()
        print(
            f"  Processing anomaly detectors for {len(valid_cols)} valid features (out of {len(all_features_df.columns)} total)..."
        )
        sys.stdout.flush()

    if "valid_cols" in locals() and len(valid_cols) > 0:
        num_clean = all_features_df[valid_cols].dropna()

        if len(num_clean) > 100:
            # Standardize
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_clean)

            # Reduce dimensions if needed
            if num_scaled.shape[1] > 50:
                pca = PCA(n_components=50)
                num_scaled = pca.fit_transform(num_scaled)

            anomaly_scores = pd.DataFrame(index=num_clean.index)

            # 6a. Isolation Forest
            print("  - Running Isolation Forest...")
            iso_forest = IsolationForest(contamination=0.1, random_state=42, n_jobs=-1)
            iso_predictions = iso_forest.fit_predict(num_scaled)
            iso_scores = iso_forest.score_samples(num_scaled)
            anomaly_scores["isolation_forest"] = iso_scores
            anomaly_scores["isolation_forest_outlier"] = (iso_predictions == -1).astype(
                int
            )
            print(
                f"    Isolation Forest: {anomaly_scores['isolation_forest_outlier'].sum()} outliers detected"
            )

            # 6b. Local Outlier Factor
            print("  - Running Local Outlier Factor...")
            n_neighbors = min(20, len(num_clean) - 1)
            lof = LocalOutlierFactor(
                n_neighbors=n_neighbors, contamination=0.1, n_jobs=-1
            )
            lof_predictions = lof.fit_predict(num_scaled)
            lof_scores = lof.negative_outlier_factor_
            anomaly_scores[
                "local_outlier_factor"
            ] = -lof_scores  # Convert to positive scores
            anomaly_scores["local_outlier_factor_outlier"] = (
                lof_predictions == -1
            ).astype(int)
            print(
                f"    Local Outlier Factor: {anomaly_scores['local_outlier_factor_outlier'].sum()} outliers detected"
            )

            # 6c. One-Class SVM
            print("  - Running One-Class SVM...")
            try:
                ocsvm = OneClassSVM(nu=0.1, kernel="rbf", gamma="scale")
                ocsvm_predictions = ocsvm.fit_predict(num_scaled)
                ocsvm_scores = ocsvm.score_samples(num_scaled)
                anomaly_scores["one_class_svm"] = ocsvm_scores
                anomaly_scores["one_class_svm_outlier"] = (
                    ocsvm_predictions == -1
                ).astype(int)
                print(
                    f"    One-Class SVM: {anomaly_scores['one_class_svm_outlier'].sum()} outliers detected"
                )
            except Exception as e:
                print(f"    One-Class SVM failed: {e}")

            results["anomaly_scores"] = anomaly_scores

            # Visualize anomaly scores
            print(
                "Creating visualization: Anomaly Detectors (Isolation Forest, LOF, One-Class SVM)..."
            )
            sys.stdout.flush()
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            plot_idx = 0
            if "isolation_forest" in anomaly_scores.columns:
                print("  → Plotting Isolation Forest scores...")
                sys.stdout.flush()
                axes[0, 0].hist(
                    anomaly_scores["isolation_forest"],
                    bins=50,
                    edgecolor="black",
                    alpha=0.7,
                )
                axes[0, 0].set_title(
                    "Isolation Forest Scores", fontsize=12, fontweight="bold"
                )
                axes[0, 0].set_xlabel("Anomaly Score")
                axes[0, 0].set_ylabel("Frequency")
                axes[0, 0].grid(True, alpha=0.3)
                plot_idx += 1

            if "local_outlier_factor" in anomaly_scores.columns:
                print("  → Plotting Local Outlier Factor scores...")
                sys.stdout.flush()
                axes[0, 1].hist(
                    anomaly_scores["local_outlier_factor"],
                    bins=50,
                    edgecolor="black",
                    alpha=0.7,
                )
                axes[0, 1].set_title(
                    "Local Outlier Factor Scores", fontsize=12, fontweight="bold"
                )
                axes[0, 1].set_xlabel("Anomaly Score")
                axes[0, 1].set_ylabel("Frequency")
                axes[0, 1].grid(True, alpha=0.3)
                plot_idx += 1

            if "one_class_svm" in anomaly_scores.columns:
                print("  → Plotting One-Class SVM scores...")
                sys.stdout.flush()
                axes[1, 0].hist(
                    anomaly_scores["one_class_svm"],
                    bins=50,
                    edgecolor="black",
                    alpha=0.7,
                )
                axes[1, 0].set_title(
                    "One-Class SVM Scores", fontsize=12, fontweight="bold"
                )
                axes[1, 0].set_xlabel("Anomaly Score")
                axes[1, 0].set_ylabel("Frequency")
                axes[1, 0].grid(True, alpha=0.3)
                plot_idx += 1

            # Consensus outliers (detected by at least 2 methods)
            if len([c for c in anomaly_scores.columns if "outlier" in c]) >= 2:
                print("  → Plotting consensus outliers...")
                sys.stdout.flush()
                outlier_cols = [c for c in anomaly_scores.columns if "outlier" in c]
                consensus = anomaly_scores[outlier_cols].sum(axis=1) >= 2
                axes[1, 1].bar(
                    ["Normal", "Outlier"], [consensus.sum(), (~consensus).sum()]
                )
                axes[1, 1].set_title(
                    "Consensus Outliers (≥2 methods)", fontsize=12, fontweight="bold"
                )
                axes[1, 1].set_ylabel("Count")
                axes[1, 1].grid(True, alpha=0.3)

            print("  → Saving visualization to file...")
            sys.stdout.flush()
            plt.tight_layout()
            plt.savefig(
                output_dir / "anomaly_detectors.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
            print("✓ Saved visualization: anomaly_detectors.png")
            sys.stdout.flush()
        else:
            print("Not enough samples for anomaly detection")
    else:
        print("Not enough valid numeric columns for anomaly detection")

# 7. Hard Rule Violations using LLM
print("\n" + "=" * 60)
print("7. Hard Rule Violations (LLM-based)...")
print("=" * 60)


def check_rule_violations_llm(df, sample_size=100):
    """
    Check for hard rule violations using LLM.
    This is a placeholder that can be extended with actual LLM API calls.

    For now, implements basic rule checks without LLM:
    - Logical inconsistencies
    - Domain-specific violations
    - Cross-field violations
    """
    violations = []

    # Sample data for LLM analysis (to avoid token limits)
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)

    # Basic rule checks (can be extended with LLM)
    rule_violations = pd.DataFrame(index=sample_df.index)
    rule_violations["has_violation"] = 0

    # Check for negative values in columns that shouldn't be negative
    for col in num_df.columns:
        if "age" in col.lower() or "count" in col.lower() or "number" in col.lower():
            negative_count = (sample_df[col] < 0).sum()
            if negative_count > 0:
                violations.append(f"{col}: {negative_count} negative values")
                rule_violations.loc[
                    sample_df[sample_df[col] < 0].index, "has_violation"
                ] = 1

    # Check for extreme values (domain-specific)
    for col in num_df.columns:
        if "bmi" in col.lower():
            extreme = (sample_df[col] > 100).sum()
            if extreme > 0:
                violations.append(f"{col}: {extreme} values > 100")
        elif "age" in col.lower():
            extreme = (sample_df[col] > 150).sum()
            if extreme > 0:
                violations.append(f"{col}: {extreme} values > 150")

    # LLM-based checking would go here:
    # Example: Use OpenAI API to check for complex cross-field violations
    # prompt = f"Check this healthcare record for rule violations: {record_dict}"
    # response = openai.ChatCompletion.create(...)

    return rule_violations, violations


rule_violations_df, violation_list = check_rule_violations_llm(df, sample_size=1000)
results["rule_violations"] = rule_violations_df

if violation_list:
    print(f"Found {len(violation_list)} types of rule violations:")
    for v in violation_list:
        print(f"  - {v}")
else:
    print("No obvious rule violations detected in sample")

print(
    "Note: For complex rule checking, integrate with LLM API (OpenAI, Anthropic, etc.)"
)
print("      See function check_rule_violations_llm() for extension points.")

# 8. Joint Distribution Measurement
print("\n" + "=" * 60)
print(
    "8. Measuring Joint Distribution for ALL Columns (Numeric + One-Hot Encoded Categorical)..."
)
print("=" * 60)

# Use all_features_df which includes both numeric and one-hot encoded categorical columns
if len(all_features_df.columns) > 1:
    # Process ALL features (numeric + one-hot encoded categorical)
    cols_to_use = (
        all_features_df.columns.tolist()
    )  # All features including encoded categoricals
    total_pairs = len(cols_to_use) * (len(cols_to_use) - 1) // 2
    print(
        f"Computing joint distribution statistics for ALL {len(cols_to_use)} features..."
    )
    print(f"  - Numeric columns: {len(num_df.columns)}")
    if cat_encoded_df is not None:
        print(f"  - One-hot encoded categorical columns: {len(cat_encoded_df.columns)}")
    print(f"Total pairs to compute: {total_pairs}")
    print(
        f"Sample column names: {cols_to_use[:10]}"
    )  # Debug: show actual column names being used
    sys.stdout.flush()

    # Calculate correlation matrix for ALL pairs (numeric + encoded categorical)
    print("  → Calculating correlation matrix for all feature pairs...")
    sys.stdout.flush()
    try:
        corr_matrix = all_features_df.corr()
        results["joint_correlation"] = corr_matrix
        print(f"    ✓ Computed correlations for all {len(cols_to_use)} features")
        sys.stdout.flush()
    except Exception as e:
        print(f"    Warning: Could not calculate full correlation matrix: {e}")
        corr_matrix = None

    # Calculate mutual information for ALL pairs (measure of dependence)
    print(
        "  → Calculating mutual information for all feature pairs (including encoded categoricals)..."
    )
    sys.stdout.flush()
    from sklearn.feature_selection import mutual_info_regression

    mi_matrix = np.zeros((len(cols_to_use), len(cols_to_use)))
    pair_count = 0

    for i, col1 in enumerate(cols_to_use):
        if (i + 1) % 10 == 0:
            print(
                f"    Processed {i + 1}/{len(cols_to_use)} features ({pair_count}/{total_pairs} pairs)..."
            )
            sys.stdout.flush()
        for j, col2 in enumerate(cols_to_use):
            if i == j:
                # Set diagonal to high value (self-information) to ensure column is not dropped
                # Use variance as approximation for self-MI
                try:
                    col_data = all_features_df[col1].dropna()
                    if len(col_data) > 100:
                        # Approximate entropy (self-information) - use variance as proxy
                        mi_matrix[i, j] = col_data.var() if col_data.var() > 0 else 1.0
                    else:
                        mi_matrix[i, j] = 1.0  # Default value
                except Exception:
                    mi_matrix[i, j] = 1.0  # Default value
            else:
                try:
                    common_idx = all_features_df[[col1, col2]].dropna().index
                    if len(common_idx) > 100:
                        mi = mutual_info_regression(
                            all_features_df.loc[common_idx, col1].values.reshape(-1, 1),
                            all_features_df.loc[common_idx, col2].values,
                            random_state=42,
                        )[0]
                        mi_matrix[i, j] = mi
                    pair_count += 1
                except Exception:
                    mi_matrix[i, j] = 0
                    pair_count += 1

    # Ensure we're using actual column names, not table names
    mi_df = pd.DataFrame(mi_matrix, index=cols_to_use, columns=cols_to_use)
    # Verify the DataFrame has correct column names and ALL columns
    print(f"    ✓ Mutual information matrix shape: {mi_df.shape}")
    print(
        f"    ✓ Expected columns: {len(cols_to_use)}, Actual columns in matrix: {len(mi_df.columns)}"
    )
    if len(mi_df.columns) != len(cols_to_use):
        print("    ⚠ WARNING: Column count mismatch! Missing columns:")
        missing = set(cols_to_use) - set(mi_df.columns)
        print(f"      {missing}")
    print(f"    ✓ Matrix index (first 5): {list(mi_df.index[:5])}")
    print(f"    ✓ Matrix columns (first 5): {list(mi_df.columns[:5])}")
    print(f"    ✓ Matrix columns (last 5): {list(mi_df.columns[-5:])}")
    results["mutual_information"] = mi_df
    print(f"    ✓ Computed mutual information for all {total_pairs} pairs")
    sys.stdout.flush()

    # Calculate additional joint distribution statistics for ALL pairs
    print(
        "  → Calculating additional joint distribution statistics (Pearson, Spearman, Covariance)..."
    )
    sys.stdout.flush()
    joint_stats_list = []

    for i, col1 in enumerate(cols_to_use):
        if (i + 1) % 20 == 0:
            print(
                f"    Processed {i + 1}/{len(cols_to_use)} features for joint stats..."
            )
            sys.stdout.flush()
        for j, col2 in enumerate(cols_to_use):
            if i < j:  # Only compute upper triangle to avoid duplicates
                try:
                    common_idx = all_features_df[[col1, col2]].dropna().index
                    if len(common_idx) > 100:
                        data1 = all_features_df.loc[common_idx, col1]
                        data2 = all_features_df.loc[common_idx, col2]

                        # Pearson correlation
                        pearson_corr = (
                            data1.corr(data2)
                            if data1.std() > 0 and data2.std() > 0
                            else np.nan
                        )

                        # Spearman correlation
                        spearman_corr = (
                            data1.corr(data2, method="spearman")
                            if len(data1) > 0
                            else np.nan
                        )

                        # Covariance
                        cov = np.cov(data1, data2)[0, 1] if len(data1) > 1 else np.nan

                        joint_stats_list.append(
                            {
                                "column1": col1,
                                "column2": col2,
                                "pearson_correlation": pearson_corr,
                                "spearman_correlation": spearman_corr,
                                "covariance": cov,
                                "mutual_information": mi_matrix[i, j],
                                "n_samples": len(common_idx),
                            }
                        )
                except Exception:
                    pass

    joint_stats_df = pd.DataFrame(joint_stats_list)
    results["joint_distribution_stats"] = joint_stats_df
    print(f"    ✓ Computed joint statistics for {len(joint_stats_list)} column pairs")
    sys.stdout.flush()

    # Visualize top correlated pairs (for visualization only - all pairs are computed above)
    print("\n  Creating visualizations (showing top correlated pairs for display)...")
    sys.stdout.flush()
    sample_pairs = []

    if corr_matrix is not None:
        try:
            corr_abs = corr_matrix.abs()
            np.fill_diagonal(corr_abs.values, 0)

            # Get top 4 most correlated pairs for visualization
            if len(corr_abs) > 0 and not corr_abs.isna().all().all():
                flat_corr = corr_abs.stack()
                flat_corr = flat_corr[flat_corr > 0]
                if len(flat_corr) > 0:
                    top_pairs = flat_corr.nlargest(min(4, len(flat_corr)))
                    sample_pairs = [(i, j) for (i, j), _ in top_pairs.items()]
        except Exception as e:
            print(f"    Warning: Could not select top pairs: {e}")

    # If no correlated pairs found, use first few columns
    if len(sample_pairs) == 0 and len(cols_to_use) >= 2:
        sample_pairs = [
            (cols_to_use[0], cols_to_use[i]) for i in range(1, min(5, len(cols_to_use)))
        ]

    if len(sample_pairs) > 0:
        print(
            "Creating visualization: Joint Distributions (hexbin plots - sample pairs)..."
        )
        sys.stdout.flush()
        n_pairs = min(4, len(sample_pairs))
        n_rows = (n_pairs + 1) // 2
        n_cols = 2 if n_pairs > 1 else 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 8 * n_rows))
        if n_pairs == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        print(
            f"  → Plotting {n_pairs} sample joint distribution pairs (all pairs computed above)..."
        )
        sys.stdout.flush()
        for idx, (col1, col2) in enumerate(sample_pairs[:n_pairs]):
            ax = axes[idx]

            # Scatter plot with density (use all_features_df to include encoded categoricals)
            data1 = all_features_df[col1].dropna()
            data2 = all_features_df[col2].dropna()
            common_idx = data1.index.intersection(data2.index)

            if len(common_idx) > 0:
                try:
                    ax.hexbin(
                        all_features_df.loc[common_idx, col1],
                        all_features_df.loc[common_idx, col2],
                        gridsize=30,
                        cmap="Blues",
                    )
                    ax.set_xlabel(col1, fontsize=10)
                    ax.set_ylabel(col2, fontsize=10)
                    ax.set_title(
                        f"Joint Distribution: {col1} vs {col2}",
                        fontsize=12,
                        fontweight="bold",
                    )
                    ax.grid(True, alpha=0.3)
                except Exception as e:
                    print(f"Warning: Could not plot {col1} vs {col2}: {e}")
                    ax.text(
                        0.5,
                        0.5,
                        f"Could not plot\n{col1} vs {col2}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"No overlapping data\n{col1} vs {col2}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis("off")

        print("  → Saving visualization to file...")
        sys.stdout.flush()
        plt.tight_layout()
        plt.savefig(
            output_dir / "joint_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Saved visualization: joint_distributions.png")
        sys.stdout.flush()
    else:
        print(
            "Warning: Could not find suitable column pairs for joint distribution visualization"
        )

    # Visualize mutual information
    if len(mi_df) > 0:
        print("Creating visualization: Mutual Information Matrix...")
        sys.stdout.flush()
        print("  → Drawing heatmap...")
        sys.stdout.flush()
        # Ensure we're using the correct column names for visualization
        print(f"  → Matrix has {len(mi_df)} rows and {len(mi_df.columns)} columns")
        print(f"  → Column names (all {len(mi_df.columns)}): {list(mi_df.columns)}")
        sys.stdout.flush()

        # Verify we have all expected columns
        if len(mi_df.columns) != len(cols_to_use):
            print(
                f"  ⚠ WARNING: Visualization will show {len(mi_df.columns)} columns, but expected {len(cols_to_use)}"
            )
            missing = set(cols_to_use) - set(mi_df.columns)
            if missing:
                print(f"  ⚠ Missing columns: {missing}")
        sys.stdout.flush()
        plt.figure(
            figsize=(
                max(12, len(mi_df.columns) * 0.3),
                max(10, len(mi_df.columns) * 0.3),
            )
        )
        # Ensure all columns are shown - use the actual DataFrame columns
        # Don't filter columns - show all of them
        sns.heatmap(
            mi_df,
            annot=False,
            cmap="viridis",
            square=True,
            cbar_kws={"label": "Mutual Information"},
            fmt=".3f",
            xticklabels=mi_df.columns.tolist() if len(mi_df.columns) <= 50 else False,
            yticklabels=mi_df.columns.tolist() if len(mi_df.columns) <= 50 else False,
        )
        plt.title(
            "Mutual Information Matrix (Joint Distribution Measure)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Column 1", fontsize=12)
        plt.ylabel("Column 2", fontsize=12)
        plt.tight_layout()
        print("  → Saving visualization to file...")
        sys.stdout.flush()
        plt.savefig(
            output_dir / "mutual_information_matrix.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        print("✓ Saved visualization: mutual_information_matrix.png")
        sys.stdout.flush()
    else:
        print("Warning: Could not compute mutual information matrix")

# Save all results
print("\n" + "=" * 60)
print("Saving Results...")
print("=" * 60)
results_dir = output_dir / "results"
results_dir.mkdir(exist_ok=True)

for key, value in results.items():
    if isinstance(value, pd.DataFrame):
        value.to_parquet(
            results_dir / f"{key}.parquet", compression="snappy", index=False
        )
        print(f"Saved {key} to {results_dir / f'{key}.parquet'}")
    elif isinstance(value, pd.Series):
        value.to_parquet(results_dir / f"{key}.parquet", compression="snappy")
        print(f"Saved {key} to {results_dir / f'{key}.parquet'}")

print(f"\nAll analysis complete! Results saved to {output_dir}")
