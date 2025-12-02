You are an assistant helping me implement a Forest-Flow style tabular generative model (from “Generating and Imputing Tabular Data via Diffusion and Flow-based Gradient-Boosted Trees”, 2024) in a Python project for synthetic data generation on healthcare tabular data (MIMIC-style).

IMPORTANT:
- Follow good coding practice at all times:
  - Clean, consistent formatting (PEP8-like where possible).
  - Remove redundant code and unused imports.
  - Avoid dead code and “TODO” placeholders for core logic.
  - Keep functions focused and small.
  - Use clear naming and docstrings for public classes and methods.
- Only implement the requested features. Do NOT add extra features unless they are strictly necessary.
- Do NOT skip or hand-wave any crucial steps. Every step described below must be implemented explicitly and correctly.

---

## Overall goals

Implement a clean, well-structured, and minimal-but-complete Python module that:

1. Preprocesses a tabular dataset (numeric + categorical) into a numeric matrix suitable for XGBoost.
2. Implements a Forest-Flow generative model using XGBoost + Flow Matching (Independent Conditional Flow Matching).
3. Exposes a simple API:
   - `TabularPreprocessor` for preprocessing / inverse preprocessing.
   - `ForestFlow` for fitting and sampling synthetic points in the preprocessed space.
4. Includes an end-to-end demo script:
   - Takes a pandas DataFrame.
   - Splits into train/val/test.
   - Fits the preprocessor and ForestFlow on the train set.
   - Generates synthetic samples.
   - Inverse-transforms them back to the original tabular format.
   - Prints basic sanity checks.

---

## Core algorithm details (Forest-Flow)

Forest-Flow is a flow-matching generative model for tabular data where the vector field is approximated with Gradient-Boosted Trees (XGBoost), following Independent Conditional Flow Matching (I-CFM):

- Training data: `X` of shape `(n, d)` in a continuous space (post-preprocessing).
- Hyperparameters:
  - `nt`: number of discrete time levels (e.g. 50).
  - `n_noise`: number of Gaussian noise samples per data point (e.g. 50–100).

### 1. Data duplication and Gaussian noise

This is crucial and must be implemented exactly (no shortcuts):

1. Given `X` of shape `(n, d)` in preprocessed space:
   - Duplicate each row of `X` exactly `n_noise` times to create `X_dup` of shape `(n * n_noise, d)`.
2. Sample Gaussian noise `Z` of the same shape as `X_dup`:
   - `Z ~ N(0, I)` elementwise.

This approximates the expectation over data–noise pairs, and is required because XGBoost does not use minibatches in the same way as neural networks.

### 2. Time discretization

Define `nt` time levels:

- `t_levels = [1/nt, 2/nt, ..., nt/nt = 1.0]` (uniform grid from 0 to 1, excluding 0, including 1).

Each time level `t` has its own model; do NOT merge all times into a single model.

### 3. Construct training pairs for each time level

For each time level `t` in `t_levels`:

- Compute the noisy inputs:
  - `X_t = (1 - t) * Z + t * X_dup`  (linear interpolation between noise and data).
- Compute targets (the conditional vector field for I-CFM):
  - `Y_t = X_dup - Z`.

We then train a separate regression model `f_t` such that:

- `f_t(X_t) ≈ Y_t`.

This approximates the vector field at time `t`.

### 4. Model architecture per time level

For each time level `t`, train a multi-output regression model:

- Use `MultiOutputRegressor(XGBRegressor(...))`:
  - Input: `X_t` of shape `(n * n_noise, d)`.
  - Output: `Y_t` of shape `(n * n_noise, d)`.

Important choices:

- Use `XGBRegressor` with:
  - `n_estimators=100` (as in the paper).
  - `reg_alpha=0.0` and `reg_lambda=0.0` (no L1/L2 regularization).
  - Reasonable defaults for other parameters.
  - `tree_method="hist"` or another CPU-friendly method.
  - `n_jobs=1` inside each individual XGBRegressor.
- Parallelization must happen at the time-level granularity, not inside each model:
  - Use `joblib.Parallel` or similar to train one `MultiOutputRegressor` per time level in parallel.
- Store models in a mapping: `self.models_[t_index] = model`, where `t_index` indexes into `t_levels`.

Do NOT partially implement this; the multi-output regression and per-time-level models must be fully coded.

### 5. Sampling (generation)

To generate `n_samples` synthetic points in the preprocessed space:

1. Let `h = 1.0 / nt` (time step size).
2. Initialize `X_t` at `t = 1`:
   - Sample `X_t ~ N(0, I)` of shape `(n_samples, d)` (Gaussian noise).
3. For `t_index` from `nt-1` down to `0` (i.e., stepping backwards in time):
   - `t = t_levels[t_index]`.
   - Get the corresponding model `f_t = self.models_[t_index]`.
   - Compute `Y_hat = f_t.predict(X_t)`.
   - Update:
     - `X_t = X_t + h * Y_hat`.
4. After this loop, `X_t` approximates samples from the data distribution in preprocessed space.
5. Optionally, clip the result to `[-1, 1]` for stability before inverse transforming.

This backward integration must be implemented exactly; do not skip or approximate away this logic.

### 6. Handling missing values

- XGBoost can handle missing values natively.
- Therefore:
  - Do NOT impute missing values before training Forest-Flow.
  - Allow NaNs in the preprocessed matrix `X` passed into `ForestFlow.fit`.
  - Still ensure the encoding step produces numeric arrays (float), but NaNs are allowed.

---

## Preprocessing and postprocessing (TabularPreprocessor)

Implement a `TabularPreprocessor` class that converts a mixed-type DataFrame into a continuous, scaled matrix, and can invert that mapping.

### Inputs

- A list of numeric columns.
- A list of categorical columns.

### Method: fit(df: pd.DataFrame)

1. `df` is a pandas DataFrame containing both numeric and categorical columns with possible NaNs.
2. For categorical columns:
   - Cast to pandas `category` dtype.
   - Apply `pd.get_dummies(..., dummy_na=True)` so that missing values in categoricals become a “NaN” category.
3. For numeric columns:
   - Keep them as they are, including NaNs.
4. Concatenate numeric and dummy-encoded categorical columns into a single DataFrame `df_all`.
5. Fit a `MinMaxScaler(feature_range=(-1, 1))` on `df_all.values` to scale all features (including dummy columns) into the range `[-1, 1]`.
6. Store:
   - `self.numeric_cols`
   - `self.categorical_cols`
   - The list of dummy columns in a fixed order (e.g. `self.dummy_columns`).
   - A mapping from each original categorical column name to its associated dummy columns (for inverse transform), e.g. `self.cat_groups[col] = list_of_dummy_cols`.
   - The fitted `MinMaxScaler` instance as `self.scaler`.

### Method: transform(df: pd.DataFrame) -> np.ndarray

1. Re-apply dummy encoding to `df[self.categorical_cols]` with `pd.get_dummies(..., dummy_na=True)`.
2. Align dummy columns with `self.dummy_columns` by reindexing and filling missing columns with 0.0.
3. Concatenate numeric columns and aligned categorical dummy columns in the same order as during `fit`.
4. Use `self.scaler.transform(...)` to scale to `[-1, 1]`.
5. Return a NumPy array of shape `(n_samples, d)`.

### Method: inverse_transform(X: np.ndarray) -> pd.DataFrame

1. Use `self.scaler.inverse_transform(X)` to move from `[-1, 1]` back to the original numeric scales.
2. Split the resulting array into:
   - Numeric columns (in order of `self.numeric_cols`).
   - Dummy categorical columns (in order of `self.dummy_columns`).
3. For each original categorical column:
   - Get its dummy column group from `self.cat_groups[col]`.
   - For each row, compute the `argmax` over that group to choose the most likely category.
   - Derive the category label from the dummy column name (e.g. if dummy is `"gender_F"`, original column is `"gender"`, category is `"F"`).
4. Construct and return a DataFrame with:
   - Numeric columns as float (or cast to original dtypes if desired).
   - Categorical columns reconstructed from dummy groups.
5. Provide a straightforward hook (e.g. an optional list of int-like columns or a simple method) to round certain numeric columns and cast them to integer types (e.g. `Int64`), but keep this minimal and focused.

The preprocessing and inverse preprocessing must be fully implemented and tested. Do NOT just stub these out.

---

## ForestFlow class API and implementation details

Implement a `ForestFlow` class with the following public API:

- Constructor:
  - `__init__(self, nt=50, n_noise=100, n_jobs=-1, xgb_params=None, random_state=42)`
- Methods:
  - `fit(self, X: np.ndarray) -> "ForestFlow"`
  - `sample(self, n_samples: int, random_state=None) -> np.ndarray`

### Constructor requirements

- Parameters:
  - `nt`: number of time steps (noise levels).
  - `n_noise`: number of noise samples per data point.
  - `n_jobs`: number of parallel jobs for training across time levels.
  - `xgb_params`: optional dict of parameters for XGBRegressor. If None, set reasonable defaults:
    - `n_estimators=100`
    - `max_depth=6` (or similar sensible default)
    - `learning_rate=0.1`
    - `subsample=1.0`
    - `colsample_bytree=1.0`
    - `reg_alpha=0.0`
    - `reg_lambda=0.0`
    - `tree_method="hist"`
    - `n_jobs=1` (to be overridden by outer parallelization)
  - `random_state`: seed for reproducibility.
- Store all constructor arguments as attributes.
- Initialize:
  - `self.models_ = {}` (mapping time index -> fitted model).
  - `self.t_levels_` later during fit.
  - `self.d_` (feature dimensionality) during fit.

### Internal helper methods

Implement internal helpers (private methods) to keep the code clean:

- `_duplicate_and_noise(self, X) -> (X_dup, Z)`:
  - Duplicates `X` by `self.n_noise` times along the row dimension.
  - Samples `Z ~ N(0, I)` of the same shape using a reproducible RNG.
- `_build_training_pairs_for_t(self, X_dup, Z, t) -> (X_t, Y_t)`:
  - Computes:
    - `X_t = (1 - t) * Z + t * X_dup`
    - `Y_t = X_dup - Z`
- `_train_level(self, X_dup, Z, t, t_index)`:
  - Builds `X_t, Y_t` using `_build_training_pairs_for_t`.
  - Instantiates a `MultiOutputRegressor(XGBRegressor(...))` with `xgb_params` and `random_state`.
  - Fits the model on `(X_t, Y_t)`.
  - Returns `(t_index, model)` for collection by parallel code.

Keep all these helpers focused and free of redundant work.

### Method: fit(self, X: np.ndarray)

1. Validate `X` is a 2D array; store `self.d_ = X.shape[1]`.
2. Call `_duplicate_and_noise(X)` to get `X_dup` and `Z`.
3. Construct `t_levels` as a 1D NumPy array of length `nt`:
   - `t_levels[i] = (i + 1) / nt` for `i = 0, ..., nt-1`.
4. Use `joblib.Parallel` (or similar) to train one model per time level:
   - For each `(t_index, t)`:
     - Call `_train_level(X_dup, Z, t, t_index)`.
5. Collect results into `self.models_`:
   - `self.models_[t_index] = model`.
6. Store `self.t_levels_ = t_levels`.
7. Return `self`.

Ensure that internal XGBoost models use `n_jobs=1` and that parallelism is controlled by the outer loop over time levels.

### Method: sample(self, n_samples: int, random_state=None) -> np.ndarray

1. Check that `self.models_` is non-empty; if not, raise a clear error (e.g. “ForestFlow must be fit before sampling”).
2. Set up RNG with `self.random_state` if `random_state` is None, otherwise use the provided seed.
3. Let `d = self.d_` and `h = 1.0 / self.nt`.
4. Initialize `X_t` as:
   - `X_t ~ N(0, I)` of shape `(n_samples, d)`.
5. Iterate over time indices in reverse order:
   - For `t_index` in `reversed(range(self.nt))`:
     - Get `model = self.models_[t_index]`.
     - Compute `Y_hat = model.predict(X_t)` (shape `(n_samples, d)`).
     - Update `X_t = X_t + h * Y_hat`.
6. Return `X_t` (in preprocessed space; caller can clip and inverse-transform).

Do NOT add noise or stochasticity here beyond the initial noise; this is a deterministic ODE-style flow.

---

## Module structure and code organization

Use a simple, clean package structure, for example:

- `forest_flow/__init__.py`
- `forest_flow/preprocessing.py` (contains `TabularPreprocessor`)
- `forest_flow/model.py` (contains `ForestFlow`)
- `examples/run_forest_flow_demo.py` (or similar demo script)

Requirements:

- Use type hints for function signatures where sensible.
- Provide concise docstrings for public classes and methods.
- Avoid repeating logic across modules; factor out shared behavior into helpers.
- Keep imports minimal and remove unused imports.

---

## Demo script (end-to-end example)

Create a demo script (e.g. `examples/run_forest_flow_demo.py`) that:

1. Imports pandas, numpy, `TabularPreprocessor`, and `ForestFlow`.
2. Loads a sample tabular dataset:
   - For now, this can be any CSV file or a toy DataFrame.
   - Explicitly mark in comments where to plug in the real MIMIC flat table.
3. Explicitly defines:
   - `numeric_cols` (e.g. selected numeric feature names).
   - `categorical_cols` (e.g. selected categorical feature names).
   - Excludes ID-like columns such as patient_id, etc.
4. Splits the DataFrame into train/val/test, e.g. 60/20/20 with a fixed random seed.
5. Instantiates and fits `TabularPreprocessor` on the training split only.
6. Transforms train, val, and test into numeric arrays via `pre.transform(...)`.
7. Instantiates `ForestFlow` (with reasonable nt and n_noise, e.g. 50 and 20 for the demo).
8. Calls `flow.fit(X_train)` to train the model.
9. Calls `flow.sample(n_synth)` to generate synthetic samples in preprocessed space.
10. Clips the synthetic samples to `[-1, 1]`.
11. Uses `pre.inverse_transform(...)` to get a synthetic DataFrame with original column names.
12. Prints:
    - `df_synth.head()`.
    - Basic descriptive stats (e.g. `.describe()`) or value counts for some categorical columns.
    - Any simple sanity checks to confirm shapes and types.

Make sure this script is fully runnable and does not leave any core steps unimplemented.

---

## Label-conditional extension (optional but requested as a skeleton)

Provide a small, clearly marked example (in comments or a separate function) showing how to extend to label-conditional generation:

- Assume there is a label column `y` in the original DataFrame.
- Show how to:
  - Split the training data by label value (e.g. mortality = 0 vs 1).
  - For each label value, fit a separate `ForestFlow` model on the subset of rows with that label.
  - To sample label-conditional synthetic data:
    - Sample labels according to the empirical training distribution.
    - For each label value:
      - Call the corresponding `ForestFlow.sample` for some number of rows.
      - Inverse-transform and assign the label column.
    - Concatenate synthetic data across labels.

This can be written as a helper function or an example block, but it must be conceptually correct and syntactically valid.

---

## Final tasks

1. Scaffold the `forest_flow` package structure in the current workspace.
2. Implement `TabularPreprocessor` in `preprocessing.py` with fully working `fit`, `transform`, and `inverse_transform`.
3. Implement `ForestFlow` in `model.py` with fully working:
   - Constructor (with defaults and xgb_params handling).
   - `_duplicate_and_noise`, `_build_training_pairs_for_t`, `_train_level` helpers.
   - `fit` method (including parallel training across time levels).
   - `sample` method (backward ODE integration).
4. Create the demo script in `examples/` that wires everything together and demonstrates end-to-end usage.
5. Ensure that:
   - All code is clean, well-formatted, and free of redundant or unused parts.
   - All crucial algorithmic steps described above are implemented explicitly and correctly.
   - There are no unimplemented “stubs” for core logic.

Now, start by scaffolding the package structure and then implement these components step by step, following the specification exactly.
