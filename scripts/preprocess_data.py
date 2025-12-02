#!/usr/bin/env python3
"""
Generic data preprocessing pipeline.
Handles train/val/test splits, preprocessing, and versioning.
Works with any dataset - automatically detects column types.
"""

import pandas as pd
import numpy as np
import yaml
import hashlib
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import warnings

warnings.filterwarnings("ignore")


class DataPreprocessor:
    """Data preprocessing pipeline for synthetic data generation."""

    def __init__(self, random_seed=50, missing_threshold=0.95):
        self.random_seed = random_seed
        self.missing_threshold = missing_threshold
        self.version_hash = None
        self.config_snapshot = {}
        self.transformers = {}

        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)

    def _compute_dataset_hash(self, df):
        """Compute hash of dataset for versioning."""
        # Hash based on shape, column names, and sample of data
        hash_data = {
            "shape": df.shape,
            "columns": sorted(df.columns.tolist()),
            "sample": df.head(100).to_dict(),
        }
        hash_str = json.dumps(hash_data, sort_keys=True, default=str)
        return hashlib.sha256(hash_str.encode()).hexdigest()[:16]

    def _create_config_snapshot(self):
        """Create configuration snapshot for this run."""
        return {
            "timestamp": datetime.now().isoformat(),
            "random_seed": self.random_seed,
            "dataset_version_hash": self.version_hash,
            "preprocessing_config": {
                "missing_threshold": self.missing_threshold,
                "scaling": "standard",
                "encoding": "onehot",
            },
            "split_config": {"train": 0.6, "val": 0.2, "test": 0.2},
        }

    def load_data(self, data_file):
        """Load raw data."""
        print("=" * 60)
        print("Loading Data")
        print("=" * 60)

        if data_file.suffix == ".parquet":
            df = pd.read_parquet(data_file)
        elif data_file.suffix == ".csv":
            # CSV support (but Parquet is preferred)
            df = pd.read_csv(data_file)
            print(
                "⚠ Warning: CSV format detected. Consider converting to Parquet for faster access."
            )
        else:
            raise ValueError(f"Unsupported file format: {data_file.suffix}")

        print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Compute dataset hash
        self.version_hash = self._compute_dataset_hash(df)
        print(f"Dataset version hash: {self.version_hash}")

        return df

    def handle_missing_values(self, df):
        """Handle missing values."""
        print("\n" + "-" * 60)
        print("Handling Missing Values")
        print("-" * 60)

        # Drop columns with >threshold missing values
        missing_ratios = df.isnull().sum() / len(df)
        cols_to_drop = missing_ratios[
            missing_ratios > self.missing_threshold
        ].index.tolist()

        if cols_to_drop:
            print(
                f"Dropping {len(cols_to_drop)} columns with >{self.missing_threshold * 100}% missing"
            )
            df = df.drop(columns=cols_to_drop)

        return df

    def identify_column_types(self, df):
        """Identify numeric and categorical columns."""
        print("\n" + "-" * 60)
        print("Identifying Column Types")
        print("-" * 60)

        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            if df[col].dtype in ["int64", "float64"]:
                # Check if it's categorical (low cardinality) or numeric
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 50:
                    categorical_cols.append(col)
                else:
                    numeric_cols.append(col)
            elif df[col].dtype == "object":
                categorical_cols.append(col)

        print(f"Numeric columns: {len(numeric_cols)}")
        print(f"Categorical columns: {len(categorical_cols)}")

        return {"numeric": numeric_cols, "categorical": categorical_cols}

    def split_data(self, df):
        """Split data into train/val/test (60/20/20)."""
        print("\n" + "-" * 60)
        print("Splitting Data (60/20/20)")
        print("-" * 60)

        # First split: train vs temp (val+test)
        train_df, temp_df = train_test_split(
            df, test_size=0.4, random_state=self.random_seed
        )

        # Second split: val vs test
        val_df, test_df = train_test_split(
            temp_df, test_size=0.5, random_state=self.random_seed
        )

        print(f"Train: {len(train_df)} rows ({len(train_df) / len(df) * 100:.1f}%)")
        print(f"Val: {len(val_df)} rows ({len(val_df) / len(df) * 100:.1f}%)")
        print(f"Test: {len(test_df)} rows ({len(test_df) / len(df) * 100:.1f}%)")

        return train_df, val_df, test_df

    def fit_preprocessing(self, train_df, column_types):
        """Fit preprocessing transformers on training data only."""
        print("\n" + "-" * 60)
        print("Fitting Preprocessing Transformers (Train Only)")
        print("-" * 60)

        # Numeric preprocessing
        if column_types["numeric"]:
            print(
                f"Fitting StandardScaler for {len(column_types['numeric'])} numeric columns"
            )
            numeric_imputer = SimpleImputer(strategy="median")
            numeric_scaler = StandardScaler()

            numeric_data = train_df[column_types["numeric"]].copy()
            numeric_data_imputed = numeric_imputer.fit_transform(numeric_data)
            numeric_scaler.fit(numeric_data_imputed)

            self.transformers["numeric_imputer"] = numeric_imputer
            self.transformers["numeric_scaler"] = numeric_scaler

        # Categorical preprocessing
        if column_types["categorical"]:
            print(
                f"Fitting OneHotEncoder for {len(column_types['categorical'])} categorical columns"
            )
            categorical_imputer = SimpleImputer(
                strategy="constant", fill_value="missing"
            )
            categorical_encoder = OneHotEncoder(
                sparse_output=False, handle_unknown="ignore"
            )

            categorical_data = train_df[column_types["categorical"]].copy()
            # Convert all categorical columns to string to handle mixed types
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].astype(str)

            categorical_data_imputed = categorical_imputer.fit_transform(
                categorical_data
            )
            categorical_encoder.fit(categorical_data_imputed)

            self.transformers["categorical_imputer"] = categorical_imputer
            self.transformers["categorical_encoder"] = categorical_encoder

        # Store column types for transform
        self.transformers["column_types"] = column_types

        print("✓ All transformers fitted")

    def transform_data(self, df):
        """Transform data using fitted transformers."""
        column_types = self.transformers["column_types"]
        result_df = df.copy()

        # Transform numeric columns
        if "numeric_scaler" in self.transformers and column_types["numeric"]:
            numeric_data = df[column_types["numeric"]].copy()
            numeric_data_imputed = self.transformers["numeric_imputer"].transform(
                numeric_data
            )
            numeric_data_scaled = self.transformers["numeric_scaler"].transform(
                numeric_data_imputed
            )

            numeric_df = pd.DataFrame(
                numeric_data_scaled,
                columns=[f"{col}_scaled" for col in column_types["numeric"]],
                index=df.index,
            )
            result_df = pd.concat(
                [result_df.drop(columns=column_types["numeric"]), numeric_df], axis=1
            )

        # Transform categorical columns
        if "categorical_encoder" in self.transformers and column_types["categorical"]:
            categorical_data = df[column_types["categorical"]].copy()
            # Convert all categorical columns to string to handle mixed types
            for col in categorical_data.columns:
                categorical_data[col] = categorical_data[col].astype(str)

            categorical_data_imputed = self.transformers[
                "categorical_imputer"
            ].transform(categorical_data)
            categorical_data_encoded = self.transformers[
                "categorical_encoder"
            ].transform(categorical_data_imputed)

            cat_feature_names = self.transformers[
                "categorical_encoder"
            ].get_feature_names_out(column_types["categorical"])
            categorical_df = pd.DataFrame(
                categorical_data_encoded, columns=cat_feature_names, index=df.index
            )
            result_df = pd.concat(
                [result_df.drop(columns=column_types["categorical"]), categorical_df],
                axis=1,
            )

        return result_df

    def save_transformers(self, run_id, output_dir, transformers_dir):
        """Save fitted transformers."""
        transformers_file = transformers_dir / f"transformers_{run_id}.pkl"
        joblib.dump(self.transformers, transformers_file)
        print(f"✓ Saved transformers to {transformers_file}")

    def save_config(self, run_id, configs_dir):
        """Save configuration snapshot."""
        self.config_snapshot = self._create_config_snapshot()

        config_file = configs_dir / f"config_{run_id}.yaml"
        with open(config_file, "w") as f:
            yaml.dump(self.config_snapshot, f, default_flow_style=False)
        print(f"✓ Saved config to {config_file}")

    def save_processed_data(self, train_df, val_df, test_df, run_id, output_dir):
        """Save processed datasets."""
        print("\n" + "-" * 60)
        print("Saving Processed Data")
        print("-" * 60)

        for split_name, split_df in [
            ("train", train_df),
            ("val", val_df),
            ("test", test_df),
        ]:
            parquet_file = output_dir / f"{split_name}_{run_id}.parquet"
            split_df.to_parquet(parquet_file, compression="snappy", index=False)
            print(f"  Saved {split_name} split to {parquet_file}")

            # CSV output removed - use Parquet instead
            # Uncomment below if CSV is needed for compatibility
            # csv_file = output_dir / f"{split_name}_{run_id}.csv"
            # split_df.to_csv(csv_file, index=False)

            print(
                f"✓ Saved {split_name}: {parquet_file.name} ({len(split_df)} rows, {len(split_df.columns)} cols)"
            )

    def process(self, data_file, output_dir, transformers_dir, configs_dir):
        """Run complete preprocessing pipeline."""
        # Load data
        df = self.load_data(data_file)

        # Handle missing values
        df = self.handle_missing_values(df)

        # Identify column types
        column_types = self.identify_column_types(df)

        # Split data (60/20/20)
        train_df, val_df, test_df = self.split_data(df)

        # Fit preprocessing on train only
        self.fit_preprocessing(train_df, column_types)

        # Transform all splits
        print("\n" + "-" * 60)
        print("Transforming Data")
        print("-" * 60)
        train_df_transformed = self.transform_data(train_df)
        val_df_transformed = self.transform_data(val_df)
        test_df_transformed = self.transform_data(test_df)

        # Create run ID
        run_id = f"{self.version_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Save everything
        self.save_transformers(run_id, output_dir, transformers_dir)
        self.save_config(run_id, configs_dir)
        self.save_processed_data(
            train_df_transformed,
            val_df_transformed,
            test_df_transformed,
            run_id,
            output_dir,
        )

        print("\n" + "=" * 60)
        print("✓ Preprocessing Complete!")
        print("=" * 60)
        print(f"Run ID: {run_id}")
        print(f"Dataset version hash: {self.version_hash}")
        print(f"Output directory: {output_dir}")

        return {
            "run_id": run_id,
            "version_hash": self.version_hash,
            "train_shape": train_df_transformed.shape,
            "val_shape": val_df_transformed.shape,
            "test_shape": test_df_transformed.shape,
        }


def main():
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess dataset for synthetic data generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process MIMIC-III data (suffix auto-extracted from filename)
  python preprocess_data.py \\
    --data data/mimic-iii-clinical-database-1.4/combined/mimic_iii_combined.parquet
  # Output: data/processed/mimic_iii_combined/
    --seed 42
  # Output: data/processed/diabetic_data/
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to input data file (CSV or Parquet)",
    )

    parser.add_argument(
        "--missing-threshold",
        type=float,
        default=0.95,
        help="Threshold for dropping columns with missing values (default: 0.95)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Resolve paths
    BASE_DIR = Path(__file__).parent.parent
    data_file = Path(args.data)

    # Validate data file
    if not data_file.exists():
        print(f"❌ Data file not found: {data_file}")
        return 1

    # Extract suffix from filename (without extension)
    suffix = data_file.stem

    # Set output directory with suffix
    output_dir = BASE_DIR / "data" / "processed" / suffix
    transformers_dir = output_dir / "transformers"
    configs_dir = output_dir / "configs"

    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    transformers_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Run preprocessing
    preprocessor = DataPreprocessor(
        random_seed=args.seed, missing_threshold=args.missing_threshold
    )
    preprocessor.process(data_file, output_dir, transformers_dir, configs_dir)

    return 0


if __name__ == "__main__":
    exit(main())
