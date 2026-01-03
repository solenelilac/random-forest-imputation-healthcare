# random-forest-imputation-healthcare

Iterative Random Forest–based imputation for missing values in healthcare
data. This repository includes a small sample dataset (1,000 observations), 
designed for robust and reusable preprocessing in causal analysis and A/B testing workflows.

This repository demonstrates an iterative Random Forest–based approach
to imputing missing values in healthcare data. The pipeline handles both
numeric and categorical covariates, imputes variables in order of
increasing missingness, and preserves observed values to avoid data
leakage. The implementation is reusable and can be easily adapted to
other datasets.
