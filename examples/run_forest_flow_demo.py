#!/usr/bin/env python3
"""End-to-end demo of Forest-Flow on MIMIC-III data.

This script demonstrates:
1. Loading and preparing MIMIC-III combined data
2. Preprocessing with TabularPreprocessor
3. Training ForestFlow
4. Generating synthetic samples
5. Inverse transforming and sanity checks
"""

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, memory monitoring disabled")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from forest_flow import ForestFlow, TabularPreprocessor


# =============================================================================
# Memory Management Utilities
# =============================================================================


def get_memory_usage():
    """Get current memory usage in GB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    process = psutil.Process()
    return process.memory_info().rss / (1024**3)


def get_memory_percent():
    """Get current memory usage as percentage of available RAM."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    return psutil.virtual_memory().percent


def wait_if_memory_high(threshold_percent=85, wait_time=2.0):
    """Wait if memory usage is above threshold to prevent crashes."""
    if not PSUTIL_AVAILABLE:
        # Still do garbage collection even without monitoring
        gc.collect()
        return False

    mem_percent = get_memory_percent()
    if mem_percent > threshold_percent:
        print(f"  ⚠ Memory usage high ({mem_percent:.1f}%), waiting {wait_time}s...")
        time.sleep(wait_time)
        gc.collect()
        return True
    return False


def safe_load_parquet(filepath, max_rows=None, chunk_size=10000):
    """Load parquet file with memory safety checks.

    Args:
        filepath: Path to parquet file
        max_rows: Maximum rows to load (None = all)
        chunk_size: Chunk size for reading if file is large
    """
    file_size_mb = filepath.stat().st_size / (1024**2)
    print(f"  File size: {file_size_mb:.1f} MB")

    # For small files, load directly
    if file_size_mb < 500:
        df = pd.read_parquet(filepath)
        if max_rows and len(df) > max_rows:
            print(f"  Sampling {max_rows} rows from {len(df)} total...")
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return df

    # For large files, read in chunks
    print("  Large file detected, reading in chunks...")
    try:
        # Try to read metadata first to get row count
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(filepath)
        num_rows = parquet_file.metadata.num_rows

        if max_rows and num_rows > max_rows:
            print(f"  Sampling {max_rows} rows from {num_rows} total...")
            # Read random sample by reading all and sampling
            # For very large files, this is still memory-intensive
            # But better than loading everything
            df = pd.read_parquet(filepath)
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        else:
            df = pd.read_parquet(filepath)

        return df
    except Exception as e:
        print(f"  Warning: Could not read metadata, loading directly: {e}")
        df = pd.read_parquet(filepath)
        if max_rows and len(df) > max_rows:
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
        return df


# =============================================================================
# Configuration
# =============================================================================

DATA_PATH = Path(
    "data/mimic-iii-clinical-database-1.4/combined/mimic_iii_combined.parquet"
)

# Demo configuration (adjust for faster/slower runs)
DEMO_CONFIG = {
    "max_rows": None,  # None = use all rows, or set to limit dataset size
    "nt": 5,  # Number of time steps (5=very fast, 20=normal, 50=production)
    "n_noise": 3,  # Noise samples per data point (3=very fast, 10=normal, 100=production)
    "n_synth": 100,  # Number of synthetic samples to generate
    "n_jobs": 2,  # Parallel jobs (-1 = all CPUs, lower = less memory)
}

# Columns to exclude (IDs, dates, free text)
EXCLUDE_COLS = [
    # IDs
    "ROW_ID",
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "ROW_ID_pat",
    "ROW_ID_icu",
    "ROW_ID_services",
    "ROW_ID_transfers",
    "ICUSTAY_ID_transfers",
    # Dates (could be processed separately if needed)
    "ADMITTIME",
    "DISCHTIME",
    "DEATHTIME",
    "DOB",
    "DOD",
    "DOD_HOSP",
    "DOD_SSN",
    "EDREGTIME",
    "EDOUTTIME",
    "INTIME",
    "OUTTIME",
    "TRANSFERTIME",
    "INTIME_transfers",
    "OUTTIME_transfers",
    # Free text (high cardinality, not suitable for dummy encoding)
    "DIAGNOSIS",
]

# Categorical columns
CATEGORICAL_COLS = [
    "ADMISSION_TYPE",
    "ADMISSION_LOCATION",
    "DISCHARGE_LOCATION",
    "INSURANCE",
    "LANGUAGE",
    "RELIGION",
    "MARITAL_STATUS",
    "ETHNICITY",
    "GENDER",
    "DBSOURCE",
    "FIRST_CAREUNIT",
    "LAST_CAREUNIT",
    "PREV_SERVICE",
    "CURR_SERVICE",
    "DBSOURCE_transfers",
    "EVENTTYPE",
    "PREV_CAREUNIT",
    "CURR_CAREUNIT",
]

# Integer columns (for rounding after inverse transform)
INT_COLS = [
    "HOSPITAL_EXPIRE_FLAG",
    "HAS_CHARTEVENTS_DATA",
    "EXPIRE_FLAG",
    "FIRST_WARDID",
    "LAST_WARDID",
    "PREV_WARDID",
    "CURR_WARDID",
    "DIAGNOSES_ICD_count",
    "PROCEDURES_ICD_count",
    "CPTEVENTS_count",
    "DRGCODES_count",
    "PRESCRIPTIONS_count",
    "OUTPUTEVENTS_count",
    "INPUTEVENTS_CV_count",
    "INPUTEVENTS_MV_count",
    "PROCEDUREEVENTS_MV_count",
    "DATETIMEEVENTS_count",
    "MICROBIOLOGYEVENTS_count",
]


def main():
    print("=" * 60)
    print("Forest-Flow Demo on MIMIC-III Data")
    print("=" * 60)

    # =========================================================================
    # Step 1: Load data (with memory safety)
    # =========================================================================
    print("\n[1/6] Loading MIMIC-III combined data...")
    print(f"  Memory before loading: {get_memory_usage():.2f} GB")

    df = safe_load_parquet(DATA_PATH, max_rows=DEMO_CONFIG["max_rows"])
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Memory after loading: {get_memory_usage():.2f} GB")

    wait_if_memory_high()

    # =========================================================================
    # Step 2: Select and prepare columns
    # =========================================================================
    print("\n[2/6] Preparing columns...")

    # Remove excluded columns
    available_cols = [c for c in df.columns if c not in EXCLUDE_COLS]

    # Identify categorical columns (present in data)
    categorical_cols = [c for c in CATEGORICAL_COLS if c in available_cols]

    # Identify numeric columns (everything else)
    numeric_cols = [c for c in available_cols if c not in categorical_cols]

    # Filter to valid int columns
    int_cols = [c for c in INT_COLS if c in numeric_cols]

    print(f"  Numeric columns: {len(numeric_cols)}")
    print(f"  Categorical columns: {len(categorical_cols)}")

    # Select only the columns we need
    df = df[numeric_cols + categorical_cols].copy()
    wait_if_memory_high()

    # =========================================================================
    # Step 3: Train/Val/Test split
    # =========================================================================
    print("\n[3/6] Splitting data (60/20/20)...")
    df_train, df_temp = train_test_split(df, test_size=0.4, random_state=42)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    # Free original and intermediate dataframes after split
    del df, df_temp
    gc.collect()

    print(f"  Train: {len(df_train)} rows")
    print(f"  Val:   {len(df_val)} rows")
    print(f"  Test:  {len(df_test)} rows")
    print(f"  Memory: {get_memory_usage():.2f} GB")

    wait_if_memory_high()

    # =========================================================================
    # Step 4: Fit preprocessor and transform
    # =========================================================================
    print("\n[4/6] Fitting preprocessor on training data...")
    preprocessor = TabularPreprocessor(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        int_cols=int_cols,
    )
    preprocessor.fit(df_train)

    print(f"  Total features after encoding: {preprocessor.n_features}")

    X_train = preprocessor.transform(df_train)
    print(f"  X_train shape: {X_train.shape}")
    print(f"  Memory: {get_memory_usage():.2f} GB")
    wait_if_memory_high()

    X_val = preprocessor.transform(df_val)
    print(f"  X_val shape:   {X_val.shape}")
    wait_if_memory_high()

    X_test = preprocessor.transform(df_test)
    print(f"  X_test shape:  {X_test.shape}")

    # Free val/test dataframes after transformation (keep df_train for sanity checks)
    del df_val, df_test
    gc.collect()
    print(f"  Memory after cleanup: {get_memory_usage():.2f} GB")
    wait_if_memory_high()

    # =========================================================================
    # Step 5: Fit ForestFlow (with memory monitoring)
    # =========================================================================
    print("\n[5/6] Training ForestFlow model...")

    # Estimate memory requirements
    n_train = len(X_train)
    n_noise = DEMO_CONFIG["n_noise"]
    n_features = X_train.shape[1]
    estimated_memory_gb = (n_train * n_noise * n_features * 4) / (1024**3)
    print(f"  Training on {n_train} samples")
    print(f"  Estimated peak memory: ~{estimated_memory_gb:.2f} GB")
    print(f"  Current memory: {get_memory_usage():.2f} GB")
    print("  (This may take a while...)")

    # Check memory before training
    if get_memory_percent() > 80:
        print(f"  ⚠ Memory already at {get_memory_percent():.1f}%, clearing cache...")
        gc.collect()
        time.sleep(2)
        wait_if_memory_high(threshold_percent=75, wait_time=3.0)

    # Use config parameters
    flow = ForestFlow(
        nt=DEMO_CONFIG["nt"],
        n_noise=DEMO_CONFIG["n_noise"],
        n_jobs=DEMO_CONFIG["n_jobs"],
        random_state=42,
    )

    # Monitor memory during training
    mem_before = get_memory_usage()
    flow.fit(X_train)
    mem_after = get_memory_usage()

    print("  Training complete!")
    print(f"  Memory used during training: {mem_after - mem_before:.2f} GB")
    print(f"  Current memory: {get_memory_usage():.2f} GB")

    # Clean up intermediate arrays if possible
    gc.collect()
    wait_if_memory_high()

    # =========================================================================
    # Step 6: Generate synthetic samples
    # =========================================================================
    print("\n[6/6] Generating synthetic samples...")
    n_synth = min(DEMO_CONFIG["n_synth"], len(df_train))

    X_synth = flow.sample(n_synth, random_state=123)

    # Clip to [-1, 1] for stability
    X_synth = np.clip(X_synth, -1, 1)

    print(f"  Generated {n_synth} synthetic samples")

    # Inverse transform
    print(f"  Memory before inverse transform: {get_memory_usage():.2f} GB")
    df_synth = preprocessor.inverse_transform(X_synth)
    wait_if_memory_high()

    # =========================================================================
    # Sanity checks
    # =========================================================================
    print("\n" + "=" * 60)
    print("Sanity Checks")
    print("=" * 60)

    print("\n--- Synthetic Data Sample (first 5 rows) ---")
    print(df_synth.head().to_string())

    print("\n--- Numeric Column Statistics Comparison ---")
    # Compare a few key numeric columns
    sample_numeric = ["LOS", "DIAGNOSES_ICD_count", "HOSPITAL_EXPIRE_FLAG"]
    sample_numeric = [c for c in sample_numeric if c in numeric_cols]

    if sample_numeric:
        print("\nReal data (train):")
        print(df_train[sample_numeric].describe().round(2).to_string())

        print("\nSynthetic data:")
        print(df_synth[sample_numeric].describe().round(2).to_string())

    print("\n--- Categorical Column Value Counts ---")
    sample_categorical = ["GENDER", "ADMISSION_TYPE", "INSURANCE"]
    sample_categorical = [c for c in sample_categorical if c in categorical_cols]

    for col in sample_categorical[:2]:
        print(f"\n{col} - Real (train) vs Synthetic:")
        real_counts = df_train[col].value_counts(normalize=True).head(5)
        synth_counts = df_synth[col].value_counts(normalize=True).head(5)
        comparison = (
            pd.DataFrame({"Real": real_counts, "Synthetic": synth_counts})
            .fillna(0)
            .round(3)
        )
        print(comparison.to_string())

    # Free df_train after sanity checks
    del df_train
    gc.collect()

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


# =============================================================================
# Label-conditional generation example (skeleton)
# =============================================================================
def label_conditional_example():
    """Example of label-conditional generation (e.g., mortality prediction).

    This is a skeleton showing how to extend to conditional generation.
    """

    # Assuming X_train and y_train are prepared (y = mortality label)
    # y_train = df_train["HOSPITAL_EXPIRE_FLAG"].values

    # Fit separate models per label
    # conditional_result = fit_label_conditional(X_train, y_train, nt=20, n_noise=10)

    # Sample from conditional models
    # X_synth, y_synth = sample_label_conditional(conditional_result, n_samples=1000)

    # Inverse transform and assign labels
    # df_synth = preprocessor.inverse_transform(X_synth)
    # df_synth["HOSPITAL_EXPIRE_FLAG"] = y_synth

    pass


if __name__ == "__main__":
    main()
