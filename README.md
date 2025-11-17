# Healthcare Dataset Curation

This repository contains scripts for downloading and combining NHANES (National Health and Nutrition Examination Survey) data for ML model training.

## Dataset

### NHANES Survey Data (2017-2020)
**Source**: US CDC National Health and Nutrition Examination Survey  
**Access**: Public (direct download)  
**Components**: Demographics, Laboratory, Questionnaire, Examination  
**Status**: Automated download and combination available

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

1. **Download NHANES data** (all components for 2017-2020):
```bash
uv run python scripts/download_nhanes_all_2017_2020.py
```

2. **Combine all data files** by joining on SEQN (participant ID):
```bash
uv run python scripts/combine_nhanes_data.py
```

Or activate the virtual environment first:
```bash
source .venv/bin/activate  # On macOS/Linux
python scripts/download_nhanes_all_2017_2020.py
python scripts/combine_nhanes_data.py
```

The combined dataset will be saved to:
- `data/nhanes/2017-2020/combined/nhanes_2017_2020_combined.parquet` (Parquet format)
- `data/nhanes/2017-2020/combined/nhanes_2017_2020_combined.csv` (CSV format)

## Directory Structure

```
synth-gen/
├── data/
│   └── nhanes/
│       └── 2017-2020/
│           ├── demographics/      # Demographics data files
│           ├── laboratory/       # Laboratory data files
│           ├── questionnaire/     # Questionnaire data files
│           ├── examination/       # Examination data files
│           └── combined/          # Combined dataset
│               ├── nhanes_2017_2020_combined.parquet
│               └── nhanes_2017_2020_combined.csv
├── scripts/
│   ├── download_nhanes_all_2017_2020.py  # Download all NHANES components
│   └── combine_nhanes_data.py            # Combine all files by SEQN
├── pyproject.toml                 # Project configuration and dependencies
├── .venv/                         # Virtual environment (created by uv)
└── README.md
```

## Usage Example

After combining the data, you can load it in Python:

```python
import pandas as pd

# Load the combined dataset
df = pd.read_parquet('data/nhanes/2017-2020/combined/nhanes_2017_2020_combined.parquet')

# Or load from CSV
# df = pd.read_csv('data/nhanes/2017-2020/combined/nhanes_2017_2020_combined.csv')

print(f"Dataset shape: {df.shape}")
print(f"Participants: {df['SEQN'].nunique()}")
print(f"Variables: {len(df.columns)}")
```

## Notes

- NHANES data files are in XPT format (SAS transport format)
- The combined dataset joins all components on SEQN (sequence number), which is the unique participant identifier
- Ensure you have sufficient disk space (~500MB+ for raw files, ~100MB+ for combined dataset)
