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

When to use this method: This approach is appropriate when missing values 
can be reasonably explained by other observed variables in the dataset.

Disclaimer: Random Forest imputation cannot recover information that is 
fundamentally unobserved or systematically hidden. This method should not 
be applied to treatment assignment, outcome variables, performance metrics, post-treatment variables, or primary keys. 
Variables with a very high rate of missingness require special caution and should be carefully reviewed; 
in some cases, exclusion may be more appropriate than heavy imputation.

