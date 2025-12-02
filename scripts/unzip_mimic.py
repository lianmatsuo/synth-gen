#!/usr/bin/env python3
"""
Unzip all .gz files in the MIMIC-III database folder.
"""

import gzip
import shutil
from pathlib import Path
from tqdm import tqdm

mimic_dir = Path("data/mimic-iii-clinical-database-1.4")

if not mimic_dir.exists():
    print(f"Error: Directory {mimic_dir} does not exist")
    exit(1)

# Find all .gz files
gz_files = list(mimic_dir.glob("*.gz"))
print(f"Found {len(gz_files)} .gz files to decompress")

if len(gz_files) == 0:
    print("No .gz files found")
    exit(0)

# Decompress each file
for gz_file in tqdm(gz_files, desc="Decompressing"):
    output_file = gz_file.with_suffix("")  # Remove .gz extension

    # Skip if already decompressed
    if output_file.exists():
        print(f"Skipping {gz_file.name} (already decompressed)")
        continue

    try:
        with gzip.open(gz_file, "rb") as f_in:
            with open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        print(f"Decompressed: {gz_file.name} -> {output_file.name}")
    except Exception as e:
        print(f"Error decompressing {gz_file.name}: {e}")

print("\nDecompression complete!")
