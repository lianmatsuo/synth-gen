#!/usr/bin/env python3
"""
Convert all MIMIC-III CSV files to Parquet format for faster access.
Parquet is typically 5-10x faster to read and uses less disk space.
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

mimic_dir = Path("data/mimic-iii-clinical-database-1.4")
parquet_dir = mimic_dir / "parquet"
parquet_dir.mkdir(exist_ok=True)

print("=" * 60)
print("Converting MIMIC-III CSVs to Parquet")
print("=" * 60)
print(f"Source: {mimic_dir}")
print(f"Destination: {parquet_dir}")
print()

# Find all CSV files (excluding combined directory)
csv_files = list(mimic_dir.glob("*.csv"))
csv_files = [f for f in csv_files if "combined" not in str(f)]

print(f"Found {len(csv_files)} CSV files to convert\n")

converted = 0
skipped = 0
errors = []

for csv_file in tqdm(csv_files, desc="Converting", unit="file"):
    parquet_file = parquet_dir / f"{csv_file.stem}.parquet"

    # Skip if already converted and newer
    if (
        parquet_file.exists()
        and parquet_file.stat().st_mtime > csv_file.stat().st_mtime
    ):
        skipped += 1
        continue

    try:
        # Get file size for progress info
        file_size_gb = csv_file.stat().st_size / (1024**3)

        if file_size_gb > 1.0:
            # Large files: use chunked reading
            print(f"\n  Converting {csv_file.name} ({file_size_gb:.2f}GB)...")
            chunks = []
            chunk_size = 100000

            # Count rows for progress bar
            total_rows = sum(1 for _ in open(csv_file)) - 1
            num_chunks = (total_rows // chunk_size) + 1

            chunk_progress = tqdm(
                pd.read_csv(csv_file, chunksize=chunk_size, low_memory=False),
                total=num_chunks,
                desc=f"    Reading {csv_file.name}",
                unit="chunk",
                leave=False,
            )

            for chunk in chunk_progress:
                chunks.append(chunk)

            df = pd.concat(chunks, ignore_index=True)

            # Save as Parquet
            print("    Saving to Parquet...")
            df.to_parquet(parquet_file, compression="snappy", index=False)

            # Show compression ratio
            parquet_size_gb = parquet_file.stat().st_size / (1024**3)
            compression_ratio = (1 - parquet_size_gb / file_size_gb) * 100
            print(
                f"    ✓ Converted: {file_size_gb:.2f}GB → {parquet_size_gb:.2f}GB ({compression_ratio:.1f}% smaller)"
            )

            del df, chunks
        else:
            # Small files: read directly
            df = pd.read_csv(csv_file, low_memory=False)
            df.to_parquet(parquet_file, compression="snappy", index=False)

            csv_size_mb = csv_file.stat().st_size / (1024**2)
            parquet_size_mb = parquet_file.stat().st_size / (1024**2)
            compression_ratio = (1 - parquet_size_mb / csv_size_mb) * 100
            print(
                f"  ✓ {csv_file.name}: {csv_size_mb:.1f}MB → {parquet_size_mb:.1f}MB ({compression_ratio:.1f}% smaller)"
            )

        converted += 1

    except Exception as e:
        errors.append((csv_file.name, str(e)))
        print(f"  ✗ Error converting {csv_file.name}: {e}")

print("\n" + "=" * 60)
print("Conversion Summary")
print("=" * 60)
print(f"Converted: {converted}")
print(f"Skipped (already up-to-date): {skipped}")
if errors:
    print(f"Errors: {len(errors)}")
    for filename, error in errors:
        print(f"  - {filename}: {error}")
print(f"\nParquet files saved to: {parquet_dir}")
print("\nNext steps:")
print("1. Update scripts to use Parquet files instead of CSV")
print("2. Parquet files are typically 5-10x faster to read")
