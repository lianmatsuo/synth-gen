"""Tabular preprocessing for Forest-Flow."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class TabularPreprocessor:
    """Preprocess mixed-type DataFrames for Forest-Flow.

    Converts numeric + categorical columns into a scaled [-1, 1] matrix.
    Supports inverse transformation back to original format.
    """

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        int_cols: list[str] | None = None,
        max_missing_ratio: float | None = 0.5,
    ):
        """Initialize preprocessor.

        Args:
            numeric_cols: List of numeric column names.
            categorical_cols: List of categorical column names.
            int_cols: Optional list of numeric columns to cast to int on inverse.
            max_missing_ratio: Maximum ratio of missing values per row (0.0-1.0).
                              Rows with more missing values are dropped.
                              Default 0.5 (50%). Set to None to disable filtering.
        """
        self.numeric_cols = list(numeric_cols)
        self.categorical_cols = list(categorical_cols)
        self.int_cols = list(int_cols) if int_cols else []
        self.max_missing_ratio = max_missing_ratio

        self.scaler: MinMaxScaler | None = None
        self.dummy_columns: list[str] | None = None
        self.cat_groups: dict[str, list[str]] | None = None
        self._is_fitted = False
        self._n_rows_filtered: int = 0

    def fit(self, df: pd.DataFrame) -> "TabularPreprocessor":
        """Fit the preprocessor on training data.

        Args:
            df: Training DataFrame with numeric and categorical columns.

        Returns:
            self
        """
        # Filter rows with majority missing values if enabled
        if self.max_missing_ratio is not None:
            df = self._filter_missing_rows(df, is_fit=True)

        # Build dummy-encoded DataFrame for categoricals
        df_cat = df[self.categorical_cols].astype("category")
        df_dummies = pd.get_dummies(df_cat, dummy_na=True)

        # Store dummy column names and mapping
        self.dummy_columns = list(df_dummies.columns)
        self.cat_groups = {}
        for col in self.categorical_cols:
            prefix = f"{col}_"
            self.cat_groups[col] = [
                c for c in self.dummy_columns if c.startswith(prefix)
            ]

        # Concatenate numeric and dummy columns
        df_numeric = df[self.numeric_cols].astype(float)
        df_all = pd.concat([df_numeric, df_dummies], axis=1)

        # Fit scaler to [-1, 1]
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.scaler.fit(df_all.values)

        self._is_fitted = True
        return self

    def _filter_missing_rows(
        self, df: pd.DataFrame, is_fit: bool = False
    ) -> pd.DataFrame:
        """Filter out rows with majority missing values.

        Args:
            df: DataFrame to filter.
            is_fit: If True, store the count of filtered rows.

        Returns:
            Filtered DataFrame.
        """
        # Count missing values per row across all relevant columns
        relevant_cols = self.numeric_cols + self.categorical_cols
        missing_per_row = df[relevant_cols].isnull().sum(axis=1)
        total_cols = len(relevant_cols)
        missing_ratio = missing_per_row / total_cols

        # Filter rows where missing ratio exceeds threshold
        valid_mask = missing_ratio <= self.max_missing_ratio
        n_filtered = (~valid_mask).sum()

        if is_fit and n_filtered > 0:
            self._n_rows_filtered = n_filtered
            print(
                f"  Filtered {n_filtered} rows ({n_filtered / len(df) * 100:.1f}%) "
                f"with >{self.max_missing_ratio * 100:.0f}% missing values"
            )

        return df[valid_mask].copy()

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform DataFrame to scaled numpy array.

        Args:
            df: DataFrame with same columns as training data.

        Returns:
            Scaled array of shape (n_samples, d) in range [-1, 1].
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before transform")

        # Filter rows with majority missing values if enabled
        if self.max_missing_ratio is not None:
            df = self._filter_missing_rows(df, is_fit=False)

        # Dummy encode categoricals
        df_cat = df[self.categorical_cols].astype("category")
        df_dummies = pd.get_dummies(df_cat, dummy_na=True)

        # Align dummy columns with fitted columns
        df_dummies = df_dummies.reindex(columns=self.dummy_columns, fill_value=0.0)

        # Concatenate numeric and dummy columns
        df_numeric = df[self.numeric_cols].astype(float)
        df_all = pd.concat([df_numeric, df_dummies], axis=1)

        # Scale
        return self.scaler.transform(df_all.values)

    def inverse_transform(self, X: np.ndarray) -> pd.DataFrame:
        """Inverse transform scaled array back to DataFrame.

        Args:
            X: Scaled array of shape (n_samples, d).

        Returns:
            DataFrame with original column structure.
        """
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit before inverse_transform")

        # Inverse scale
        X_inv = self.scaler.inverse_transform(X)

        # Split into numeric and dummy parts
        n_numeric = len(self.numeric_cols)
        X_numeric = X_inv[:, :n_numeric]
        X_dummies = X_inv[:, n_numeric:]

        # Reconstruct numeric columns
        result = pd.DataFrame(X_numeric, columns=self.numeric_cols)

        # Round and cast integer columns
        for col in self.int_cols:
            if col in result.columns:
                result[col] = result[col].round().astype("Int64")

        # Reconstruct categorical columns via argmax
        dummy_df = pd.DataFrame(X_dummies, columns=self.dummy_columns)
        for col in self.categorical_cols:
            group_cols = self.cat_groups[col]
            if not group_cols:
                result[col] = np.nan
                continue

            # Get argmax index for each row
            group_values = dummy_df[group_cols].values
            argmax_idx = np.argmax(group_values, axis=1)

            # Extract category from dummy column name
            prefix = f"{col}_"
            categories = [c[len(prefix) :] for c in group_cols]

            # Map argmax to category, handle NaN category
            result[col] = [categories[i] for i in argmax_idx]
            # Replace "nan" string with actual NaN
            result[col] = result[col].replace("nan", np.nan)

        return result

    @property
    def n_features(self) -> int:
        """Return total number of features after transformation."""
        if not self._is_fitted:
            raise RuntimeError("Preprocessor must be fit first")
        return len(self.numeric_cols) + len(self.dummy_columns)
