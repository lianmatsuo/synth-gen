import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for subprocess execution
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import sys
from pathlib import Path
from dython.nominal import associations, theils_u, cramers_v
from scipy.stats import f_oneway, kruskal

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
    description="Generate correlation statistics for MIMIC-III data"
)
parser.add_argument("--data", type=str, help="Path to data file (CSV or Parquet)")
args = parser.parse_args()

# Create output directory
output_dir = Path("eda/correlation_stats")
output_dir.mkdir(parents=True, exist_ok=True)

# Load MIMIC-III data
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

num_df = df.select_dtypes(include=["number"])
cat_cols_all = df.select_dtypes(include=["object"]).columns

# Filter categorical columns: skip those with too many unique values (high cardinality)
# This prevents memory issues with large cross-tabulation tables
MAX_CATEGORIES = 100  # Maximum number of unique values per categorical column
cat_cols = []
for col in cat_cols_all:
    unique_count = df[col].nunique()
    if unique_count <= MAX_CATEGORIES:
        cat_cols.append(col)
    else:
        print(
            f"Skipping {col} (too many unique values: {unique_count} > {MAX_CATEGORIES})"
        )
        sys.stdout.flush()

cat_cols = pd.Index(cat_cols)
print(
    f"Processing {len(cat_cols)} categorical columns (skipped {len(cat_cols_all) - len(cat_cols)} high-cardinality columns)"
)
sys.stdout.flush()

# Calculate correlations
print("Calculating numeric correlations (Pearson, Spearman, Kendall)...")
sys.stdout.flush()
pearson_corr = num_df.corr(method="pearson")
print("  ✓ Pearson correlation calculated")
sys.stdout.flush()
spearman_corr = num_df.corr(method="spearman")
print("  ✓ Spearman correlation calculated")
sys.stdout.flush()
kendall_corr = num_df.corr(method="kendall")
print("  ✓ Kendall correlation calculated")
sys.stdout.flush()

# Visualize numeric correlations
if len(num_df.columns) > 0:
    print(
        "Creating visualization: Numeric Correlations (Pearson, Spearman, Kendall)..."
    )
    sys.stdout.flush()
    fig, axes = plt.subplots(1, 3, figsize=(36, 10))

    print("  → Drawing Pearson correlation heatmap...")
    sys.stdout.flush()
    # Pearson correlation
    sns.heatmap(
        pearson_corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        ax=axes[0],
        cbar_kws={"label": "Correlation"},
    )
    axes[0].set_title("Pearson Correlation", fontsize=14, fontweight="bold")

    print("  → Drawing Spearman correlation heatmap...")
    sys.stdout.flush()
    # Spearman correlation
    sns.heatmap(
        spearman_corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        ax=axes[1],
        cbar_kws={"label": "Correlation"},
    )
    axes[1].set_title("Spearman Correlation", fontsize=14, fontweight="bold")

    print("  → Drawing Kendall correlation heatmap...")
    sys.stdout.flush()
    # Kendall correlation
    sns.heatmap(
        kendall_corr,
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
        ax=axes[2],
        cbar_kws={"label": "Correlation"},
    )
    axes[2].set_title("Kendall Correlation", fontsize=14, fontweight="bold")

    print("  → Saving visualization to file...")
    sys.stdout.flush()
    plt.tight_layout()
    plt.savefig(output_dir / "numeric_correlations.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved visualization: numeric_correlations.png")
    sys.stdout.flush()

# Calculate categorical associations
if len(cat_cols) > 0:
    print("Calculating Cramer's V matrix...")
    sys.stdout.flush()
    try:
        # Limit the number of categorical columns to process at once to avoid memory issues
        # Process in batches if there are too many columns
        MAX_COLS_PER_BATCH = 50
        if len(cat_cols) > MAX_COLS_PER_BATCH:
            print(
                f"  Processing {len(cat_cols)} categorical columns in batches of {MAX_COLS_PER_BATCH}..."
            )
            sys.stdout.flush()
            # For large datasets, calculate pairwise Cramer's V manually
            cat_list = list(cat_cols)
            cramers_v_matrix = np.eye(
                len(cat_list)
            )  # Identity matrix (1.0 on diagonal)

            for i in range(len(cat_list)):
                if i % 10 == 0:
                    print(
                        f"  Processing Cramer's V: {i + 1}/{len(cat_list)} columns..."
                    )
                    sys.stdout.flush()
                for j in range(i + 1, len(cat_list)):
                    try:
                        cv = cramers_v(df[cat_list[i]], df[cat_list[j]])
                        cramers_v_matrix[i, j] = cv
                        cramers_v_matrix[j, i] = cv  # Symmetric
                    except Exception:
                        # Skip pairs that cause memory issues
                        cramers_v_matrix[i, j] = np.nan
                        cramers_v_matrix[j, i] = np.nan

            cramers_v_matrix = pd.DataFrame(
                cramers_v_matrix, index=cat_list, columns=cat_list
            )
        else:
            assoc = associations(
                df[cat_cols], nom_nom_assoc="cramer", plot=False, compute_only=True
            )
            cramers_v_matrix = assoc["corr"]  # includes Cramer's V for cat↔cat
    except Exception as e:
        print(f"Error calculating Cramer's V: {e}")
        print("Skipping Cramer's V calculation due to memory constraints")
        sys.stdout.flush()
        cramers_v_matrix = None

    # Visualize Cramer's V
    if cramers_v_matrix is not None:
        print("Creating visualization: Cramer's V Matrix...")
        sys.stdout.flush()
        print("  → Drawing Cramer's V heatmap...")
        sys.stdout.flush()
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cramers_v_matrix,
            annot=False,
            cmap="viridis",
            square=True,
            fmt=".2f",
            cbar_kws={"label": "Cramer's V"},
        )
        plt.title(
            "Cramer's V Association Matrix (Categorical Variables)",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        print("  → Saving visualization to file...")
        sys.stdout.flush()
        plt.savefig(output_dir / "cramers_v_matrix.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("✓ Saved visualization: cramers_v_matrix.png")
        sys.stdout.flush()
    else:
        print("Skipping Cramer's V visualization (calculation failed)")
        sys.stdout.flush()

# Calculate Theil's U for all categorical pairs (asymmetric)
if len(cat_cols) > 1:
    cat_list = list(cat_cols)
    theils_u_matrix = np.zeros((len(cat_list), len(cat_list)))
    total_pairs = len(cat_list) * len(cat_list)
    current_pair = 0

    print("Calculating Theil's U matrix...")
    sys.stdout.flush()
    for i, col_x in enumerate(cat_list):
        print(
            f"Calculating Theil's U: {col_x} vs all categories ({i + 1}/{len(cat_list)})..."
        )
        sys.stdout.flush()
        for j, col_y in enumerate(cat_list):
            if i != j:
                try:
                    u_xy = theils_u(df[col_x], df[col_y])  # U(X|Y)
                    theils_u_matrix[i, j] = u_xy
                except Exception:
                    # Skip if calculation fails (e.g., too many unique values)
                    theils_u_matrix[i, j] = np.nan
            else:
                theils_u_matrix[i, j] = 1.0  # Perfect predictability with itself
            current_pair += 1

    theils_u_df = pd.DataFrame(theils_u_matrix, index=cat_list, columns=cat_list)

    # Visualize Theil's U
    print("Creating visualization: Theil's U Matrix...")
    sys.stdout.flush()
    print("  → Drawing Theil's U heatmap...")
    sys.stdout.flush()
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        theils_u_df,
        annot=False,
        cmap="YlOrRd",
        square=True,
        fmt=".2f",
        cbar_kws={"label": "Theil's U"},
    )
    plt.title(
        "Theil's U Matrix (Categorical → Categorical Predictability)",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("Y (conditioned on)", fontsize=12)
    plt.ylabel("X (predicting)", fontsize=12)
    plt.tight_layout()
    print("  → Saving visualization to file...")
    sys.stdout.flush()
    plt.savefig(output_dir / "theils_u_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved visualization: theils_u_matrix.png")
    sys.stdout.flush()

# Calculate ANOVA F-statistics and Kruskal-Wallis H-statistics (categorical → numerical)
if len(cat_cols) > 0 and len(num_df.columns) > 0:
    cat_list = list(cat_cols)
    num_list = list(num_df.columns)

    anova_f_matrix = np.zeros((len(cat_list), len(num_list)))
    anova_p_matrix = np.zeros((len(cat_list), len(num_list)))
    kruskal_h_matrix = np.zeros((len(cat_list), len(num_list)))
    kruskal_p_matrix = np.zeros((len(cat_list), len(num_list)))

    print("Calculating ANOVA and Kruskal-Wallis statistics...")
    sys.stdout.flush()
    total_combinations = len(cat_list) * len(num_list)
    current_combination = 0

    for i, cat_col in enumerate(cat_list):
        print(
            f"Calculating ANOVA/Kruskal-Wallis: {cat_col} vs all numerical ({i + 1}/{len(cat_list)})..."
        )
        sys.stdout.flush()
        for j, num_col in enumerate(num_list):
            try:
                # Group numerical values by categorical values
                groups = [
                    group[num_col].dropna().values
                    for name, group in df.groupby(cat_col)
                ]
                groups = [g for g in groups if len(g) > 0]  # Remove empty groups

                if len(groups) > 1:
                    # ANOVA F-test
                    F_stat, p_val = f_oneway(*groups)
                    anova_f_matrix[i, j] = F_stat
                    anova_p_matrix[i, j] = p_val

                    # Kruskal-Wallis H-test
                    H_stat, p_val = kruskal(*groups)
                    kruskal_h_matrix[i, j] = H_stat
                    kruskal_p_matrix[i, j] = p_val
                else:
                    anova_f_matrix[i, j] = np.nan
                    anova_p_matrix[i, j] = np.nan
                    kruskal_h_matrix[i, j] = np.nan
                    kruskal_p_matrix[i, j] = np.nan
            except Exception:
                anova_f_matrix[i, j] = np.nan
                anova_p_matrix[i, j] = np.nan
                kruskal_h_matrix[i, j] = np.nan
                kruskal_p_matrix[i, j] = np.nan
            current_combination += 1

    anova_f_df = pd.DataFrame(anova_f_matrix, index=cat_list, columns=num_list)
    anova_p_df = pd.DataFrame(anova_p_matrix, index=cat_list, columns=num_list)
    kruskal_h_df = pd.DataFrame(kruskal_h_matrix, index=cat_list, columns=num_list)
    kruskal_p_df = pd.DataFrame(kruskal_p_matrix, index=cat_list, columns=num_list)

    # Visualize ANOVA F-statistics
    print("Creating visualization: ANOVA/Kruskal-Wallis Matrix...")
    sys.stdout.flush()
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))

    print("  → Drawing ANOVA F-statistics heatmap...")
    sys.stdout.flush()
    # ANOVA F-statistics
    sns.heatmap(
        anova_f_df,
        annot=False,
        cmap="Reds",
        square=False,
        fmt=".2f",
        ax=axes[0, 0],
        cbar_kws={"label": "F-statistic"},
        mask=anova_f_df.isna(),
    )
    axes[0, 0].set_title(
        "ANOVA F-statistics (Categorical → Numerical)", fontsize=14, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Numerical Variables", fontsize=12)
    axes[0, 0].set_ylabel("Categorical Variables", fontsize=12)

    print("  → Drawing ANOVA p-values heatmap...")
    sys.stdout.flush()
    # ANOVA p-values
    sns.heatmap(
        anova_p_df,
        annot=False,
        cmap="RdYlGn_r",
        square=False,
        fmt=".3f",
        ax=axes[0, 1],
        cbar_kws={"label": "p-value"},
        vmin=0,
        vmax=0.05,
        mask=anova_p_df.isna(),
    )
    axes[0, 1].set_title(
        "ANOVA p-values (Categorical → Numerical)", fontsize=14, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Numerical Variables", fontsize=12)
    axes[0, 1].set_ylabel("Categorical Variables", fontsize=12)

    print("  → Drawing Kruskal-Wallis H-statistics heatmap...")
    sys.stdout.flush()
    # Kruskal-Wallis H-statistics
    sns.heatmap(
        kruskal_h_df,
        annot=False,
        cmap="Blues",
        square=False,
        fmt=".2f",
        ax=axes[1, 0],
        cbar_kws={"label": "H-statistic"},
        mask=kruskal_h_df.isna(),
    )
    axes[1, 0].set_title(
        "Kruskal-Wallis H-statistics (Categorical → Numerical)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 0].set_xlabel("Numerical Variables", fontsize=12)
    axes[1, 0].set_ylabel("Categorical Variables", fontsize=12)

    print("  → Drawing Kruskal-Wallis p-values heatmap...")
    sys.stdout.flush()
    # Kruskal-Wallis p-values
    sns.heatmap(
        kruskal_p_df,
        annot=False,
        cmap="RdYlGn_r",
        square=False,
        fmt=".3f",
        ax=axes[1, 1],
        cbar_kws={"label": "p-value"},
        vmin=0,
        vmax=0.05,
        mask=kruskal_p_df.isna(),
    )
    axes[1, 1].set_title(
        "Kruskal-Wallis p-values (Categorical → Numerical)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1, 1].set_xlabel("Numerical Variables", fontsize=12)
    axes[1, 1].set_ylabel("Categorical Variables", fontsize=12)

    print("  → Saving visualization to file...")
    sys.stdout.flush()
    plt.tight_layout()
    plt.savefig(output_dir / "anova_kruskal_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("✓ Saved visualization: anova_kruskal_matrix.png")
    sys.stdout.flush()

print(f"\nAll visualizations saved to {output_dir}")
