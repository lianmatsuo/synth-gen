#!/usr/bin/env python3
"""
Combine all NHANES 2017-2020 data files by joining on SEQN (sequence number).
SEQN is the unique identifier for each participant in NHANES.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent.parent / "data" / "nhanes" / "2017-2020"
OUTPUT_DIR = BASE_DIR / "combined"
OUTPUT_FILE = OUTPUT_DIR / "nhanes_2017_2020_combined.parquet"
OUTPUT_CSV = OUTPUT_DIR / "nhanes_2017_2020_combined.csv"


def read_xpt_file(file_path):
    """Read an XPT file and return DataFrame."""
    try:
        df = pd.read_sas(file_path, format='xport', encoding='utf-8')
        return df
    except Exception as e:
        print(f"  ⚠ Error reading {file_path.name}: {e}")
        return None


def get_all_xpt_files():
    """Get all XPT files organized by component."""
    components = {
        "demographics": [],
        "laboratory": [],
        "questionnaire": [],
        "examination": []
    }
    
    for component in components.keys():
        component_dir = BASE_DIR / component
        if component_dir.exists():
            files = list(component_dir.glob("*.xpt")) + list(component_dir.glob("*.XPT"))
            components[component] = sorted(files)
    
    return components


def combine_nhanes_data():
    """Combine all NHANES data files by joining on SEQN."""
    print("="*60)
    print("NHANES 2017-2020 Data Combination")
    print("="*60)
    
    # Get all files
    all_files = get_all_xpt_files()
    
    total_files = sum(len(files) for files in all_files.values())
    print(f"\nFound {total_files} data files to combine:")
    for component, files in all_files.items():
        print(f"  {component.capitalize()}: {len(files)} files")
    
    # Start with demographics (should have SEQN for all participants)
    print("\n" + "-"*60)
    print("Step 1: Loading Demographics (base dataset)...")
    print("-"*60)
    
    demo_files = all_files["demographics"]
    if not demo_files:
        print("❌ No demographics files found!")
        return None
    
    # Load demographics - this will be our base
    combined_df = None
    for file_path in tqdm(demo_files, desc="Demographics"):
        df = read_xpt_file(file_path)
        if df is not None and 'SEQN' in df.columns:
            if combined_df is None:
                combined_df = df.copy()
                print(f"  ✓ Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
            else:
                # If multiple demo files, merge them
                combined_df = pd.merge(combined_df, df, on='SEQN', how='outer', suffixes=('', '_dup'))
                print(f"  ✓ Merged {file_path.name}: {len(combined_df)} rows, {len(combined_df.columns)} columns")
    
    if combined_df is None:
        print("❌ Failed to load demographics data!")
        return None
    
    print(f"\nBase dataset: {len(combined_df)} participants, {len(combined_df.columns)} columns")
    
    # Remove duplicate columns (keep original, drop _dup)
    dup_cols = [col for col in combined_df.columns if col.endswith('_dup')]
    if dup_cols:
        combined_df = combined_df.drop(columns=dup_cols)
    
    # Now join all other components
    components_order = ["laboratory", "questionnaire", "examination"]
    
    for component in components_order:
        files = all_files[component]
        if not files:
            continue
        
        print("\n" + "-"*60)
        print(f"Step 2: Loading {component.capitalize()} data...")
        print("-"*60)
        
        component_dfs = []
        for file_path in tqdm(files, desc=component.capitalize()):
            df = read_xpt_file(file_path)
            if df is not None and 'SEQN' in df.columns:
                component_dfs.append(df)
                print(f"  ✓ Loaded {file_path.name}: {len(df)} rows, {len(df.columns)} columns")
        
        if component_dfs:
            # Combine all files from this component first
            component_combined = component_dfs[0]
            for df in component_dfs[1:]:
                # Merge files within component, handling duplicate column names
                component_combined = pd.merge(
                    component_combined, 
                    df, 
                    on='SEQN', 
                    how='outer',
                    suffixes=('', '_dup')
                )
                # Remove duplicate columns
                dup_cols = [col for col in component_combined.columns if col.endswith('_dup')]
                if dup_cols:
                    component_combined = component_combined.drop(columns=dup_cols)
            
            # Now merge with main combined dataset
            print(f"\n  Merging {component} data with main dataset...")
            before_cols = len(combined_df.columns)
            combined_df = pd.merge(
                combined_df,
                component_combined,
                on='SEQN',
                how='outer',
                suffixes=('', f'_{component}')
            )
            after_cols = len(combined_df.columns)
            print(f"  ✓ Merged: {len(combined_df)} participants, {after_cols - before_cols} new columns added")
    
    # Final dataset info
    print("\n" + "="*60)
    print("Combined Dataset Summary")
    print("="*60)
    print(f"  Total participants (rows): {len(combined_df):,}")
    print(f"  Total variables (columns): {len(combined_df.columns):,}")
    print(f"  Memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Check for missing SEQN
    if combined_df['SEQN'].isna().any():
        print(f"  ⚠ Warning: {combined_df['SEQN'].isna().sum()} rows with missing SEQN")
    
    # Save the combined dataset
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "-"*60)
    print("Saving combined dataset...")
    print("-"*60)
    
    # Save as Parquet (efficient, preserves data types)
    print(f"  Saving to {OUTPUT_FILE}...")
    combined_df.to_parquet(OUTPUT_FILE, index=False, engine='pyarrow')
    print(f"  ✓ Saved as Parquet ({OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB)")
    
    # Also save as CSV (for compatibility)
    print(f"  Saving to {OUTPUT_CSV}...")
    combined_df.to_csv(OUTPUT_CSV, index=False)
    print(f"  ✓ Saved as CSV ({OUTPUT_CSV.stat().st_size / 1024**2:.2f} MB)")
    
    print("\n" + "="*60)
    print("✓ Data combination complete!")
    print("="*60)
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"  - Parquet: {OUTPUT_FILE.name}")
    print(f"  - CSV: {OUTPUT_CSV.name}")
    
    return combined_df


if __name__ == "__main__":
    df = combine_nhanes_data()
    if df is not None:
        print("\nFirst few rows:")
        print(df.head())
        print("\nColumn names (first 20):")
        print(df.columns[:20].tolist())
