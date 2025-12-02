# Healthcare Dataset Curation - MIMIC-III

This repository contains scripts for working with MIMIC-III (Medical Information Mart for Intensive Care) clinical database for ML model training and analysis.

## Dataset

### MIMIC-III Clinical Database
**Source**: PhysioNet (requires credentialing)
**Access**: Restricted (requires completion of CITI training)
**Components**: 26 tables including patient stays, events, diagnoses, procedures, and dictionary tables
**Status**: Data combination and analysis scripts available

See `.cursorrules` for detailed database schema information.

## Setup Instructions

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install Python dependencies:
```bash
uv sync
```

This will create a virtual environment and install all dependencies automatically.

### Quick Start

1. **Unzip MIMIC-III data** (if downloaded as .gz files):
```bash
uv run python scripts/unzip_mimic.py
```

2. **Combine all tables** into a single dataset:
```bash
uv run python scripts/combine_mimic_data.py
```

3. **Run exploratory data analysis**:

**Option A: Using Streamlit Dashboard (Recommended)**
```bash
uv run streamlit run app.py
```
Then open your browser to the URL shown (typically http://localhost:8501)

**Option B: Using Command Line**
```bash
# Correlation statistics
uv run python scripts/correlation_stats.py

# Anomaly detection and statistical analysis
uv run python scripts/anomaly_detection.py
```

4. **Preprocess the data** (train/val/test splits, missing value handling, scaling, encoding):
```bash
uv run python scripts/preprocess_data.py --data data/mimic-iii-clinical-database-1.4/combined/mimic_iii_combined.parquet
```

## Directory Structure

```
synth-gen/
├── data/
│   ├── mimic-iii-clinical-database-1.4/  # Raw MIMIC-III data
│   │   ├── ADMISSIONS.csv
│   │   ├── PATIENTS.csv
│   │   ├── CHARTEVENTS.csv
│   │   ├── ... (26 tables)
│   │   └── combined/                     # Combined dataset
│   │       └── mimic_iii_combined.parquet
│   ├── processed/                        # Preprocessed splits (train/val/test)
│   ├── correlation_stats/               # EDA: Correlation visualizations
│   └── anomaly_stats/                    # EDA: Anomaly detection results
├── scripts/
│   ├── unzip_mimic.py                   # Decompress .gz files
│   ├── combine_mimic_data.py            # Combine all MIMIC-III tables
│   ├── correlation_stats.py             # Correlation analysis
│   ├── anomaly_detection.py             # Statistical analysis and anomaly detection
│   └── preprocess_data.py               # Data preprocessing pipeline
├── pyproject.toml                        # Project configuration and dependencies
├── .cursorrules                          # MIMIC-III database schema documentation
├── .venv/                                # Virtual environment (created by uv)
└── README.md
```

## Usage Example

After combining the data, you can load it in Python:

```python
import pandas as pd

# Load the combined dataset
df = pd.read_parquet('data/mimic-iii-clinical-database-1.4/combined/mimic_iii_combined.parquet')

print(f"Dataset shape: {df.shape}")
print(f"Patients: {df['SUBJECT_ID'].nunique()}")
print(f"Admissions: {df['HADM_ID'].nunique()}")
print(f"Variables: {len(df.columns)}")
```

## Key Identifiers

- **SUBJECT_ID**: Unique patient identifier
- **HADM_ID**: Unique hospital admission identifier
- **ICUSTAY_ID**: Unique ICU stay identifier
- **ITEMID**: Concept identifier (join with D_ITEMS for definitions)

## Streamlit Dashboard

An interactive web dashboard for running analyses and viewing visualizations:

```bash
uv run streamlit run app.py
```

The dashboard provides:
- **Correlation Statistics Tab**: Run correlation analysis and view heatmaps
- **Anomaly Detection Tab**: Run statistical analysis and view anomaly detection results
- **Sidebar**: Shows data source status and output file counts
- **Auto-refresh**: Visualizations update automatically after running analyses

## Exploratory Data Analysis

### Correlation Statistics
```bash
uv run python scripts/correlation_stats.py
```

Generates:
- Pearson, Spearman, and Kendall correlation matrices
- Cramer's V for categorical variables
- Theil's U (asymmetric predictability)
- ANOVA and Kruskal-Wallis statistics

Output: `eda/correlation_stats/`

### Anomaly Detection
```bash
uv run python scripts/anomaly_detection.py
```

Calculates:
- Z-scores
- IQR and outlier detection
- Quantile ranks
- Mahalanobis distance
- kNN distance
- Isolation Forest, Local Outlier Factor, One-Class SVM
- Joint distribution measures (mutual information)

Output: `eda/anomaly_stats/`

## Notes

- MIMIC-III requires credentialing through PhysioNet
- CHARTEVENTS table is very large (33GB) and may be excluded from combination by default
- Event tables are aggregated (count, mean, std, min, max) to manage memory
- Preprocessing pipeline uses fixed 60/20/20 train/val/test splits
- All scripts use random seed=50 by default for reproducibility
