#!/usr/bin/env python3
"""
Combine all MIMIC-III tables into a single dataset.
Handles large files with chunking and efficient joins.

Note: Very large tables (CHARTEVENTS ~33GB, NOTEEVENTS ~3.7GB) are skipped
by default to prevent memory issues. Event tables are aggregated by HADM_ID
to reduce memory usage.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings
import gc
from collections import Counter

warnings.filterwarnings("ignore")

mimic_dir = Path("data/mimic-iii-clinical-database-1.4")
output_dir = Path("data/mimic-iii-clinical-database-1.4/combined")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("MIMIC-III Data Combination")
print("=" * 60)

# Define table relationships and join keys
# All tables now use Parquet format (with CSV fallback)
# Core tables (patient/admission/ICU stay info)
core_tables = {
    "PATIENTS": {"file": "PATIENTS.parquet", "key": "SUBJECT_ID"},
    "ADMISSIONS": {"file": "ADMISSIONS.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
    "ICUSTAYS": {
        "file": "ICUSTAYS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
    },
    "SERVICES": {"file": "SERVICES.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
    "TRANSFERS": {"file": "TRANSFERS.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
}

# Event tables (link to admissions/ICU stays)
event_tables = {
    "LABEVENTS": {
        "file": "LABEVENTS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID"],
        "itemid": "ITEMID",
    },
    "CHARTEVENTS": {
        "file": "CHARTEVENTS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "OUTPUTEVENTS": {
        "file": "OUTPUTEVENTS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "INPUTEVENTS_CV": {
        "file": "INPUTEVENTS_CV.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "INPUTEVENTS_MV": {
        "file": "INPUTEVENTS_MV.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "PROCEDUREEVENTS_MV": {
        "file": "PROCEDUREEVENTS_MV.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "DATETIMEEVENTS": {
        "file": "DATETIMEEVENTS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
        "itemid": "ITEMID",
    },
    "MICROBIOLOGYEVENTS": {
        "file": "MICROBIOLOGYEVENTS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID"],
    },
    "NOTEEVENTS": {"file": "NOTEEVENTS.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
}

# Diagnosis/procedure tables
diagnosis_tables = {
    "DIAGNOSES_ICD": {
        "file": "DIAGNOSES_ICD.parquet",
        "key": ["SUBJECT_ID", "HADM_ID"],
    },
    "PROCEDURES_ICD": {
        "file": "PROCEDURES_ICD.parquet",
        "key": ["SUBJECT_ID", "HADM_ID"],
    },
    "CPTEVENTS": {"file": "CPTEVENTS.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
    "DRGCODES": {"file": "DRGCODES.parquet", "key": ["SUBJECT_ID", "HADM_ID"]},
    "PRESCRIPTIONS": {
        "file": "PRESCRIPTIONS.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
    },
    "CALLOUT": {
        "file": "CALLOUT.parquet",
        "key": ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID"],
    },
}

# Dictionary tables (for lookups, will be joined separately)
dict_tables = {
    "D_ITEMS": {"file": "D_ITEMS.parquet", "key": "ITEMID"},
    "D_LABITEMS": {"file": "D_LABITEMS.parquet", "key": "ITEMID"},
    "D_ICD_DIAGNOSES": {"file": "D_ICD_DIAGNOSES.parquet", "key": "ICD9_CODE"},
    "D_ICD_PROCEDURES": {"file": "D_ICD_PROCEDURES.parquet", "key": "ICD9_CODE"},
    "D_CPT": {"file": "D_CPT.parquet", "key": "CPT_CD"},
    "CAREGIVERS": {"file": "CAREGIVERS.parquet", "key": "CGID"},
}


def load_table(filepath, chunk_size=None):
    """Load a table, preferring Parquet format for speed."""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    file_size = filepath.stat().st_size / (1024**3)

    if filepath.suffix == ".parquet":
        print(f"  Loading from Parquet ({file_size:.2f}GB)...")
        return pd.read_parquet(filepath)
    else:
        # CSV file
        print(
            f"  Loading from CSV ({file_size:.2f}GB)... [Consider converting to Parquet for speed]"
        )
        if file_size > 1.0:
            print("  Large file, using chunking...")
            chunks = []
            for chunk in pd.read_csv(
                filepath, chunksize=chunk_size or 100000, low_memory=False
            ):
                chunks.append(chunk)
            return pd.concat(chunks, ignore_index=True)
        else:
            return pd.read_csv(filepath, low_memory=False)


# Step 1: Load core tables
print("\n" + "=" * 60)
print("Step 1: Loading Core Tables")
print("=" * 60)

# Try Parquet files first, fallback to CSV
parquet_dir = mimic_dir / "parquet"


# Helper function to get file path (try Parquet first, then CSV)
def get_table_path(table_name):
    parquet_path = parquet_dir / f"{table_name}.parquet"
    if parquet_path.exists():
        return parquet_path
    csv_path = mimic_dir / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path
    raise FileNotFoundError(f"Neither Parquet nor CSV file found for {table_name}")


df_patients = load_table(get_table_path("PATIENTS"))
print(f"Loaded PATIENTS: {len(df_patients)} rows")

df_admissions = load_table(get_table_path("ADMISSIONS"))
print(f"Loaded ADMISSIONS: {len(df_admissions)} rows")

df_icustays = load_table(get_table_path("ICUSTAYS"))
print(f"Loaded ICUSTAYS: {len(df_icustays)} rows")

# Step 2: Join core tables
print("\n" + "=" * 60)
print("Step 2: Joining Core Tables")
print("=" * 60)

# Start with admissions (smaller than patients for joining)
# Then join patients to admissions
combined = df_admissions.copy()
print(f"Starting with ADMISSIONS: {len(combined)} rows")

# Join patients
print("Joining PATIENTS...")
combined = combined.merge(
    df_patients, on="SUBJECT_ID", how="left", suffixes=("", "_pat")
)
print(f"✓ After joining PATIENTS: {len(combined)} rows")
del df_patients
gc.collect()

# Join ICU stays
print("Joining ICUSTAYS...")
combined = combined.merge(
    df_icustays, on=["SUBJECT_ID", "HADM_ID"], how="left", suffixes=("", "_icu")
)
print(f"✓ After joining ICUSTAYS: {len(combined)} rows")
del df_icustays
gc.collect()

# Join other core tables
core_tables_to_process = {
    k: v
    for k, v in core_tables.items()
    if k not in ["PATIENTS", "ADMISSIONS", "ICUSTAYS"]
}

core_progress = tqdm(
    core_tables_to_process.items(), desc="Joining core tables", unit="table"
)

for table_name, table_info in core_progress:
    core_progress.set_description(f"Joining {table_name}")
    print(f"\nJoining {table_name}...")
    try:
        # Try Parquet first, fallback to CSV
        table_file = table_info["file"].replace(".parquet", "").replace(".csv", "")
        parquet_path = mimic_dir / "parquet" / f"{table_file}.parquet"
        csv_path = mimic_dir / f"{table_file}.csv"

        if parquet_path.exists():
            filepath = parquet_path
        elif csv_path.exists():
            filepath = csv_path
        else:
            print(
                f"  ⚠ Warning: Neither Parquet nor CSV found for {table_file}, skipping..."
            )
            continue

        df_table = load_table(filepath)
        keys = (
            table_info["key"]
            if isinstance(table_info["key"], list)
            else [table_info["key"]]
        )

        # Check which keys exist in combined
        available_keys = [k for k in keys if k in combined.columns]
        if available_keys:
            combined = combined.merge(
                df_table,
                on=available_keys,
                how="left",
                suffixes=("", f"_{table_name.lower()}"),
            )
            print(f"  ✓ Joined {table_name}: {len(combined)} rows")
            del df_table
            gc.collect()
        else:
            print(f"  ⚠ Skipping {table_name}: no matching keys")
    except Exception as e:
        print(f"  ✗ Error joining {table_name}: {e}")

    core_progress.update(1)

# Step 3: Load dictionary tables for later use
print("\n" + "=" * 60)
print("Step 3: Loading Dictionary Tables")
print("=" * 60)

dict_data = {}
for table_name, table_info in dict_tables.items():
    try:
        # Try Parquet first, fallback to CSV
        table_file = table_info["file"].replace(".parquet", "").replace(".csv", "")
        parquet_path = mimic_dir / "parquet" / f"{table_file}.parquet"
        csv_path = mimic_dir / f"{table_file}.csv"

        if parquet_path.exists():
            filepath = parquet_path
        elif csv_path.exists():
            filepath = csv_path
        else:
            print(
                f"  ⚠ Warning: Neither Parquet nor CSV found for {table_file}, skipping..."
            )
            continue

        dict_data[table_name] = load_table(filepath)
        print(f"Loaded {table_name}: {len(dict_data[table_name])} rows")
    except Exception as e:
        print(f"Error loading {table_name}: {e}")

# Step 4: Join diagnosis/procedure tables
print("\n" + "=" * 60)
print("Step 4: Joining Diagnosis/Procedure Tables")
print("=" * 60)

# Progress bar for overall step
diagnosis_progress = tqdm(
    diagnosis_tables.items(), desc="Processing diagnosis tables", unit="table"
)

for table_name, table_info in diagnosis_progress:
    diagnosis_progress.set_description(f"Processing {table_name}")
    print(f"\nJoining {table_name}...")
    try:
        # Try Parquet first, fallback to CSV
        table_file = table_info["file"].replace(".parquet", "").replace(".csv", "")
        parquet_path = mimic_dir / "parquet" / f"{table_file}.parquet"
        csv_path = mimic_dir / f"{table_file}.csv"

        if parquet_path.exists():
            filepath = parquet_path
        elif csv_path.exists():
            filepath = csv_path
        else:
            print(
                f"  ⚠ Warning: Neither Parquet nor CSV found for {table_file}, skipping..."
            )
            continue

        file_size = filepath.stat().st_size / (1024**3)

        # Always aggregate diagnosis/procedure tables to avoid row explosion
        # These tables can have multiple rows per admission, causing memory issues
        keys = (
            table_info["key"]
            if isinstance(table_info["key"], list)
            else [table_info["key"]]
        )
        available_keys = [k for k in keys if k in combined.columns]

        if not available_keys:
            print(f"  Skipping {table_name}: no matching keys")
            continue

        print(f"  File size: {file_size:.2f}GB, aggregating by {available_keys}...")

        # Aggregate by reading in chunks
        chunk_stats = []
        chunk_size = 50000  # Smaller chunks for memory efficiency

        # Count total rows for progress bar
        total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract header
        num_chunks = (total_rows // chunk_size) + 1

        chunk_progress = tqdm(
            pd.read_csv(filepath, chunksize=chunk_size, low_memory=False),
            total=num_chunks,
            desc=f"  Reading {table_name}",
            unit="chunk",
            leave=False,
        )

        for chunk in chunk_progress:
            # Filter available_keys to only those that exist in the chunk
            chunk_available_keys = [k for k in available_keys if k in chunk.columns]
            if not chunk_available_keys:
                continue

            # Count rows per key combination (aggregate to avoid row explosion)
            stats = (
                chunk.groupby(chunk_available_keys)
                .size()
                .reset_index(name=f"{table_name}_count")
            )
            chunk_stats.append(stats)

            # Process in batches to avoid memory buildup
            if len(chunk_stats) >= 20:
                batch_df = pd.concat(chunk_stats, ignore_index=True)
                batch_df = (
                    batch_df.groupby(available_keys)[f"{table_name}_count"]
                    .sum()
                    .reset_index()
                )

                if len(chunk_stats) == 20:  # First batch
                    final_stats = batch_df
                else:
                    final_stats = final_stats.merge(
                        batch_df, on=available_keys, how="outer", suffixes=("", "_new")
                    )
                    final_stats[f"{table_name}_count"] = final_stats[
                        f"{table_name}_count"
                    ].fillna(0) + final_stats[f"{table_name}_count_new"].fillna(0)
                    final_stats = final_stats.drop(columns=[f"{table_name}_count_new"])

                chunk_stats = []
                gc.collect()

        # Process remaining chunks
        if chunk_stats:
            batch_df = pd.concat(chunk_stats, ignore_index=True)
            batch_df = (
                batch_df.groupby(available_keys)[f"{table_name}_count"]
                .sum()
                .reset_index()
            )

            if "final_stats" in locals():
                final_stats = final_stats.merge(
                    batch_df, on=available_keys, how="outer", suffixes=("", "_new")
                )
                final_stats[f"{table_name}_count"] = final_stats[
                    f"{table_name}_count"
                ].fillna(0) + final_stats[f"{table_name}_count_new"].fillna(0)
                final_stats = final_stats.drop(columns=[f"{table_name}_count_new"])
            else:
                final_stats = batch_df

        # Join aggregated stats
        if "final_stats" in locals() and len(final_stats) > 0:
            print("  Merging aggregated stats...")
            combined = combined.merge(final_stats, on=available_keys, how="left")
            print(f"  ✓ Joined aggregated {table_name}: {len(combined)} rows")
            del final_stats, chunk_stats
            gc.collect()
        else:
            print(f"  ⚠ No data to join from {table_name}")

    except Exception as e:
        print(f"  ✗ Error joining {table_name}: {e}")
        import traceback

        traceback.print_exc()

    diagnosis_progress.update(1)

# Step 5: Handle event tables (these are very large, so we'll aggregate or sample)
print("\n" + "=" * 60)
print("Step 5: Processing Event Tables")
print("=" * 60)
print(
    "Note: Event tables are very large. We'll aggregate key statistics at HADM_ID level."
)

# Get unique HADM_IDs from combined for efficient filtering
hadm_ids = set(combined["HADM_ID"].dropna().unique())
print(f"Processing events for {len(hadm_ids)} unique admissions")

# For each event table, aggregate statistics
event_stats = {}

# Skip very large tables that cause memory issues
skip_tables = ["CHARTEVENTS", "NOTEEVENTS"]  # CHARTEVENTS is 33GB, NOTEEVENTS is 3.7GB

event_tables_to_process = {
    k: v for k, v in event_tables.items() if k not in skip_tables
}
event_progress = tqdm(
    event_tables_to_process.items(), desc="Processing event tables", unit="table"
)

for table_name, table_info in event_progress:
    event_progress.set_description(f"Processing {table_name}")
    print(f"\nProcessing {table_name}...")
    try:
        # Try Parquet first, fallback to CSV
        table_file = table_info["file"].replace(".parquet", "").replace(".csv", "")
        parquet_path = mimic_dir / "parquet" / f"{table_file}.parquet"
        csv_path = mimic_dir / f"{table_file}.csv"

        if parquet_path.exists():
            filepath = parquet_path
        elif csv_path.exists():
            filepath = csv_path
        else:
            print(
                f"  ⚠ Warning: Neither Parquet nor CSV found for {table_file}, skipping..."
            )
            continue

        file_size = filepath.stat().st_size / (1024**3)

        if file_size > 2.0:  # Very large files (>2GB)
            print(
                f"  File is very large ({file_size:.2f}GB). Aggregating by admission..."
            )

            # For very large files, we'll aggregate by key columns
            keys = (
                table_info["key"]
                if isinstance(table_info["key"], list)
                else [table_info["key"]]
            )

            # Use only HADM_ID for aggregation to reduce memory
            agg_keys = ["HADM_ID"] if "HADM_ID" in keys else keys[:1]

            # First pass: identify top ITEMIDs across all chunks (if applicable)
            top_itemids = None
            if "ITEMID" in pd.read_csv(filepath, nrows=100, low_memory=False).columns:
                print("  First pass: Identifying top ITEMIDs...")
                itemid_counts = Counter()
                for chunk in pd.read_csv(
                    filepath, chunksize=chunk_size, low_memory=False
                ):
                    if "ITEMID" in chunk.columns:
                        itemid_counts.update(chunk["ITEMID"].value_counts().to_dict())
                top_itemids = [itemid for itemid, _ in itemid_counts.most_common(50)]
                print(
                    f"  Using top {len(top_itemids)} ITEMIDs (out of {len(itemid_counts)} total)"
                )

            # Second pass: process chunks and pivot by ITEMID to get actual values
            chunk_stats = []
            chunk_size = 50000  # Smaller chunks

            print("  Processing chunks and pivoting by ITEMID to get actual values...")
            for chunk in tqdm(
                pd.read_csv(filepath, chunksize=chunk_size, low_memory=False),
                desc=f"  Processing {table_name}",
            ):
                # If we have ITEMID and VALUENUM, pivot to get actual values per ITEMID
                if (
                    "ITEMID" in chunk.columns
                    and "VALUENUM" in chunk.columns
                    and top_itemids
                ):
                    chunk_filtered = chunk[chunk["ITEMID"].isin(top_itemids)].copy()
                    if len(chunk_filtered) > 0:
                        # Pivot: mean value per ITEMID per admission
                        pivoted = (
                            chunk_filtered.groupby(agg_keys + ["ITEMID"])["VALUENUM"]
                            .mean()
                            .reset_index()
                        )
                        pivoted = pivoted.pivot_table(
                            index=agg_keys,
                            columns="ITEMID",
                            values="VALUENUM",
                            aggfunc="mean",
                        )
                        pivoted.columns = [
                            f"{table_name}_ITEMID_{itemid}"
                            for itemid in pivoted.columns
                        ]
                        pivoted = pivoted.reset_index()
                        chunk_stats.append(pivoted)
                elif "VALUENUM" in chunk.columns:
                    # Aggregate statistics (fallback if no ITEMID)
                    stats = (
                        chunk.groupby(agg_keys)["VALUENUM"]
                        .agg(["count", "mean", "std", "min", "max"])
                        .reset_index()
                    )
                    stats.columns = agg_keys + [
                        f"{table_name}_count",
                        f"{table_name}_mean",
                        f"{table_name}_std",
                        f"{table_name}_min",
                        f"{table_name}_max",
                    ]
                    chunk_stats.append(stats)
                else:
                    # Count events per key
                    stats = (
                        chunk.groupby(agg_keys)
                        .size()
                        .reset_index(name=f"{table_name}_count")
                    )
                    chunk_stats.append(stats)

            if chunk_stats:
                # Process chunks in batches to avoid memory issues
                batch_size = 10
                final_stats = None

                for i in range(0, len(chunk_stats), batch_size):
                    batch = chunk_stats[i : i + batch_size]
                    df_batch = pd.concat(batch, ignore_index=True)

                    # Re-aggregate (mean for pivoted values, sum for counts)
                    numeric_cols = [c for c in df_batch.columns if c not in agg_keys]
                    if numeric_cols:
                        agg_dict = {}
                        for col in numeric_cols:
                            if "ITEMID_" in col:
                                # Pivoted values: use mean
                                agg_dict[col] = "mean"
                            elif "count" in col:
                                # Count columns: use sum
                                agg_dict[col] = "sum"
                            else:
                                # Other stats: use mean
                                agg_dict[col] = "mean"
                        df_batch = (
                            df_batch.groupby(agg_keys)[numeric_cols]
                            .agg(agg_dict)
                            .reset_index()
                        )

                    if final_stats is None:
                        final_stats = df_batch
                    else:
                        # Merge batches - combine columns intelligently
                        numeric_cols = [
                            c for c in final_stats.columns if c not in agg_keys
                        ]
                        final_stats = (
                            final_stats.groupby(agg_keys)[numeric_cols]
                            .agg(
                                {
                                    col: "mean"
                                    if "ITEMID_" in col or col not in ["count"]
                                    else "sum"
                                    for col in numeric_cols
                                }
                            )
                            .reset_index()
                        )
                        final_stats = final_stats.merge(
                            df_batch, on=agg_keys, how="outer", suffixes=("", "_new")
                        )
                        # Combine columns
                        for col in numeric_cols:
                            if f"{col}_new" in final_stats.columns:
                                if "ITEMID_" in col:
                                    # For pivoted values, take mean of both
                                    final_stats[col] = final_stats[
                                        [col, f"{col}_new"]
                                    ].mean(axis=1)
                                elif "count" in col:
                                    # For counts, sum
                                    final_stats[col] = final_stats[col].fillna(
                                        0
                                    ) + final_stats[f"{col}_new"].fillna(0)
                                else:
                                    # For other stats, take mean
                                    final_stats[col] = final_stats[
                                        [col, f"{col}_new"]
                                    ].mean(axis=1)
                                final_stats.drop(columns=[f"{col}_new"], inplace=True)

                    del df_batch, batch
                    gc.collect()

                # Join to combined
                if "HADM_ID" in combined.columns and final_stats is not None:
                    combined = combined.merge(final_stats, on="HADM_ID", how="left")
                    print(f"  Joined aggregated stats from {table_name}")
                    del final_stats
                    gc.collect()
                else:
                    print(
                        "  Skipping join: HADM_ID not in combined or no stats generated"
                    )

                del chunk_stats
                gc.collect()
        else:
            # Smaller files - pivot by ITEMID to get actual values instead of aggregating
            df_table = load_table(filepath)
            keys = (
                table_info["key"]
                if isinstance(table_info["key"], list)
                else [table_info["key"]]
            )
            available_keys = [k for k in keys if k in combined.columns]

            if available_keys:
                # Check if this table has ITEMID and VALUENUM (e.g., LABEVENTS, OUTPUTEVENTS)
                if "ITEMID" in df_table.columns and "VALUENUM" in df_table.columns:
                    print(f"  Pivoting {table_name} by ITEMID to get actual values...")

                    # Get most common ITEMIDs (limit to top N to avoid too many columns)
                    top_itemids = (
                        df_table["ITEMID"].value_counts().head(50).index.tolist()
                    )
                    print(
                        f"  Using top {len(top_itemids)} most common ITEMIDs (out of {df_table['ITEMID'].nunique()} total)"
                    )

                    # Filter to top ITEMIDs and pivot
                    df_filtered = df_table[df_table["ITEMID"].isin(top_itemids)].copy()

                    # For each ITEMID, get the mean value per admission (or latest value)
                    # Using mean to handle multiple measurements per admission
                    pivoted = (
                        df_filtered.groupby(available_keys + ["ITEMID"])["VALUENUM"]
                        .mean()
                        .reset_index()
                    )
                    pivoted = pivoted.pivot_table(
                        index=available_keys,
                        columns="ITEMID",
                        values="VALUENUM",
                        aggfunc="mean",
                    )

                    # Rename columns to include table name
                    pivoted.columns = [
                        f"{table_name}_ITEMID_{itemid}" for itemid in pivoted.columns
                    ]
                    pivoted = pivoted.reset_index()

                    # Join to combined
                    combined = combined.merge(pivoted, on=available_keys, how="left")
                    print(
                        f"  Joined {len(pivoted.columns) - len(available_keys)} actual value columns from {table_name}"
                    )

                    del df_table, df_filtered, pivoted
                    gc.collect()
                elif "VALUENUM" in df_table.columns:
                    # Has VALUENUM but no ITEMID - aggregate statistics
                    stats = (
                        df_table.groupby(available_keys)["VALUENUM"]
                        .agg(["count", "mean", "std", "min", "max"])
                        .reset_index()
                    )
                    stats.columns = available_keys + [
                        f"{table_name}_count",
                        f"{table_name}_mean",
                        f"{table_name}_std",
                        f"{table_name}_min",
                        f"{table_name}_max",
                    ]
                    combined = combined.merge(stats, on=available_keys, how="left")
                    print(f"  Joined aggregated stats from {table_name}")
                    del df_table, stats
                    gc.collect()
                else:
                    # No VALUENUM - just count events
                    stats = (
                        df_table.groupby(available_keys)
                        .size()
                        .reset_index(name=f"{table_name}_count")
                    )
                    combined = combined.merge(stats, on=available_keys, how="left")
                    print(f"  Joined count from {table_name}")
                    del df_table, stats
                    gc.collect()

    except Exception as e:
        print(f"  Error processing {table_name}: {e}")
        import traceback

        traceback.print_exc()

# Step 6: Save combined dataset
print("\n" + "=" * 60)
print("Step 6: Saving Combined Dataset")
print("=" * 60)

print(f"\nFinal dataset shape: {combined.shape}")
print(f"Columns: {len(combined.columns)}")

# Free up memory before saving
gc.collect()

# Save to Parquet (more efficient)
output_parquet = output_dir / "mimic_iii_combined.parquet"
print(f"\nSaving to Parquet: {output_parquet}")
print("This may take a while for large datasets...")

# Save in chunks if dataset is very large
if len(combined) > 1000000:
    print("Large dataset detected, saving in chunks...")
    combined.to_parquet(
        output_parquet, index=False, compression="snappy", engine="pyarrow"
    )
else:
    combined.to_parquet(output_parquet, index=False, compression="snappy")
print("✓ Saved Parquet file")

# Save sample Parquet for quick inspection
output_sample = output_dir / "mimic_iii_combined_sample.parquet"
print(f"\nSaving sample Parquet (first 1000 rows): {output_sample}")
combined.head(1000).to_parquet(output_sample, compression="snappy", index=False)
print("✓ Saved sample Parquet")

# CSV output removed - use Parquet instead
# Uncomment below if CSV is needed for compatibility
# output_csv = output_dir / 'mimic_iii_combined.csv'
# print(f"\nSaving full CSV: {output_csv}")
# print("This may take a while for large datasets...")
# combined.to_csv(output_csv, index=False)
# print("✓ Saved full CSV file")

# Save column info
output_cols = output_dir / "mimic_iii_combined_columns.txt"
with open(output_cols, "w") as f:
    f.write(f"Total columns: {len(combined.columns)}\n\n")
    for col in combined.columns:
        f.write(f"{col}\n")
print(f"✓ Saved column list: {output_cols}")

print("\n" + "=" * 60)
print("Combination Complete!")
print("=" * 60)
print("Output files:")
print(f"  - {output_parquet}")
print(f"  - {output_sample}")
print(f"  - {output_cols}")
