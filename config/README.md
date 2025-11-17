# Configuration Files

## schema.yaml

Defines the data schema, constraints, and preprocessing rules for NHANES data:

- **Entity ID**: SEQN (unique participant identifier)
- **Data Types**: Numeric, categorical, ordinal
- **Missing Value Policies**: Threshold-based column dropping, imputation strategies
- **Preprocessing**: Scaling and encoding methods
- **Target Variables**: Primary and secondary targets for stratification
- **Split Configuration**: 60/20/20 train/val/test splits with stratification

## Usage

The schema is automatically loaded by `preprocess_data.py`. To modify preprocessing behavior, edit this file and re-run the preprocessing script.
