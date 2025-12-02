"""Forest-Flow generative model using XGBoost and Flow Matching."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor


class ForestFlow:
    """Flow-matching generative model with XGBoost vector fields.

    Implements Independent Conditional Flow Matching (I-CFM) where the
    vector field is approximated using Gradient-Boosted Trees at each
    discrete time level.
    """

    def __init__(
        self,
        nt: int = 50,
        n_noise: int = 100,
        n_jobs: int = -1,
        xgb_params: dict | None = None,
        random_state: int = 42,
    ):
        """Initialize ForestFlow.

        Args:
            nt: Number of discrete time levels.
            n_noise: Number of Gaussian noise samples per data point.
            n_jobs: Number of parallel jobs for training (-1 = all CPUs).
            xgb_params: Optional dict of XGBRegressor parameters.
            random_state: Random seed for reproducibility.
        """
        self.nt = nt
        self.n_noise = n_noise
        self.n_jobs = n_jobs
        self.random_state = random_state

        # Default XGBoost parameters (as per paper)
        self.xgb_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_alpha": 0.0,
            "reg_lambda": 0.0,
            "tree_method": "hist",
            "n_jobs": 1,  # Parallelism at time-level, not inside XGB
            "random_state": random_state,
        }
        if xgb_params:
            self.xgb_params.update(xgb_params)

        self.models_: dict[int, MultiOutputRegressor] = {}
        self.t_levels_: np.ndarray | None = None
        self.d_: int | None = None

    def _duplicate_and_noise(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Duplicate data and sample Gaussian noise.

        Args:
            X: Data array of shape (n, d).

        Returns:
            X_dup: Duplicated data of shape (n * n_noise, d).
            Z: Gaussian noise of same shape.
        """
        rng = np.random.default_rng(self.random_state)
        X_dup = np.repeat(X, self.n_noise, axis=0)
        Z = rng.standard_normal(X_dup.shape)
        return X_dup, Z

    def _build_training_pairs_for_t(
        self, X_dup: np.ndarray, Z: np.ndarray, t: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build training pairs for a specific time level.

        Args:
            X_dup: Duplicated data of shape (n * n_noise, d).
            Z: Gaussian noise of same shape.
            t: Time level in (0, 1].

        Returns:
            X_t: Interpolated inputs, shape (n * n_noise, d).
            Y_t: Target vector field, shape (n * n_noise, d).
        """
        X_t = (1 - t) * Z + t * X_dup
        Y_t = X_dup - Z
        return X_t, Y_t

    def _train_level(
        self, X_dup: np.ndarray, Z: np.ndarray, t: float, t_index: int
    ) -> tuple[int, MultiOutputRegressor]:
        """Train model for a single time level.

        Args:
            X_dup: Duplicated data.
            Z: Gaussian noise.
            t: Time level value.
            t_index: Index of this time level.

        Returns:
            Tuple of (t_index, fitted model).
        """
        X_t, Y_t = self._build_training_pairs_for_t(X_dup, Z, t)

        # Filter out rows with NaN targets (sklearn doesn't allow NaN in y)
        # XGBoost can handle NaN inputs, but targets must be finite
        valid_mask = ~np.isnan(Y_t).any(axis=1)
        if not valid_mask.all():
            X_t = X_t[valid_mask]
            Y_t = Y_t[valid_mask]

        model = MultiOutputRegressor(XGBRegressor(**self.xgb_params))
        model.fit(X_t, Y_t)

        return t_index, model

    def fit(self, X: np.ndarray) -> "ForestFlow":
        """Fit the Forest-Flow model on preprocessed data.

        Args:
            X: Preprocessed data array of shape (n, d). NaNs allowed.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")

        self.d_ = X.shape[1]

        # Duplicate data and sample noise
        X_dup, Z = self._duplicate_and_noise(X)

        # Time levels: [1/nt, 2/nt, ..., 1.0]
        self.t_levels_ = np.array([(i + 1) / self.nt for i in range(self.nt)])

        # Train models in parallel across time levels
        results = Parallel(n_jobs=self.n_jobs, verbose=10)(
            delayed(self._train_level)(X_dup, Z, t, t_index)
            for t_index, t in enumerate(self.t_levels_)
        )

        # Collect results
        self.models_ = {t_index: model for t_index, model in results}

        return self

    def sample(self, n_samples: int, random_state: int | None = None) -> np.ndarray:
        """Generate synthetic samples via backward ODE integration.

        Args:
            n_samples: Number of samples to generate.
            random_state: Optional random seed (uses self.random_state if None).

        Returns:
            Synthetic samples of shape (n_samples, d) in preprocessed space.
        """
        if not self.models_:
            raise RuntimeError("ForestFlow must be fit before sampling")

        seed = random_state if random_state is not None else self.random_state
        rng = np.random.default_rng(seed)

        # Initialize from noise
        X_t = rng.standard_normal((n_samples, self.d_)).astype(np.float32)

        # Step size
        h = 1.0 / self.nt

        # Backward integration: t from 1 to 0
        for t_index in reversed(range(self.nt)):
            model = self.models_[t_index]
            Y_hat = model.predict(X_t)
            X_t = X_t + h * Y_hat

        return X_t


def fit_label_conditional(
    X: np.ndarray,
    y: np.ndarray,
    nt: int = 50,
    n_noise: int = 100,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    """Fit separate ForestFlow models per label (for conditional generation).

    Args:
        X: Preprocessed data of shape (n, d).
        y: Label array of shape (n,).
        nt, n_noise, n_jobs, random_state: ForestFlow parameters.

    Returns:
        Dict with 'models' (label -> ForestFlow) and 'label_probs' (label -> prob).
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    label_probs = dict(zip(unique_labels, counts / len(y)))

    models = {}
    for label in unique_labels:
        mask = y == label
        X_label = X[mask]
        ff = ForestFlow(
            nt=nt, n_noise=n_noise, n_jobs=n_jobs, random_state=random_state
        )
        ff.fit(X_label)
        models[label] = ff

    return {"models": models, "label_probs": label_probs}


def sample_label_conditional(
    conditional_result: dict,
    n_samples: int,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample from label-conditional ForestFlow models.

    Args:
        conditional_result: Output from fit_label_conditional.
        n_samples: Total number of samples to generate.
        random_state: Optional random seed.

    Returns:
        X_synth: Synthetic samples of shape (n_samples, d).
        y_synth: Synthetic labels of shape (n_samples,).
    """
    rng = np.random.default_rng(random_state)
    models = conditional_result["models"]
    label_probs = conditional_result["label_probs"]

    labels = list(label_probs.keys())
    probs = np.array([label_probs[label] for label in labels])

    # Sample labels according to empirical distribution
    sampled_labels = rng.choice(labels, size=n_samples, p=probs)

    # Count samples per label
    label_counts = {label: np.sum(sampled_labels == label) for label in labels}

    # Generate samples per label
    X_parts = []
    y_parts = []
    for label, count in label_counts.items():
        if count > 0:
            X_label = models[label].sample(int(count), random_state=random_state)
            X_parts.append(X_label)
            y_parts.append(np.full(count, label))

    X_synth = np.vstack(X_parts)
    y_synth = np.concatenate(y_parts)

    # Shuffle
    idx = rng.permutation(len(X_synth))
    return X_synth[idx], y_synth[idx]
