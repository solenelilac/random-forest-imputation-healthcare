##################################################################################################
# Healthcare Product Promotion A/B Testing Data Preprocessing
#
# This script prepares user-level panel data for A/B testing of healthcare product promotions.
#
# The focus is on robust covariate preprocessing prior to treatment–control comparison, including:
# 1. Iterative Random-Forest-based imputation of missing covariates
# 2. Preservation of primary keys (user_id, time_stamp)
# 3. Exclusion of outcome variables and key promotion performance metrics from imputation
#
# The resulting dataset is suitable for
# causal analysis and treatment effect estimation in healthcare product promotion experiments.
#
# Author: Lilac Zihui Zhao
##################################################################################################

# -----------------------------
# Load packages
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# -----------------------------
# Load data
# -----------------------------

# load user-level panel data 
df = pd.read_csv("data/rawdata/cardio_track.csv",parse_dates=['time_stamp'])

print(df.head())
print(df.shape)

# inspect missing data
print("missing rate: ", df.isna().mean())

#  remove duplicates
df = df.drop_duplicates(subset=['user_id', 'time_stamp'])
print(df.shape)


# -----------------------------
# Define Random Forest imputation functions
# -----------------------------

# These functions:
# - Train on non-missing values of the target variable
# - Predict missing values using other covariates 
# - Overwrite missing entries

def rf_impute_numeric(df, target, features):
    train = df[df[target].notna()]
    test = df[df[target].isna()]

    # Skip if no missing values
    if test.empty:
        return df
    # model: random forest regressor for numeric values
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=40,
        n_jobs=-1  # parallelize across CPUs
    )

    rf.fit(train[features], train[target])

    df.loc[df[target].isna(), target] = rf.predict(test[features])

    return df


def rf_impute_categorical(df, target, features):
    train = df[df[target].notna()]
    test = df[df[target].isna()]

    if test.empty:
        return df

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=40,
        n_jobs=-1
    )

    rf.fit(train[features], train[target])
    df.loc[df[target].isna(), target] = rf.predict(test[features])

    return df


# -----------------------------
# Prepare data for imputation
# -----------------------------
# df_raw: original data (without imputation)
# df_imp: working copy that will be imputed
df_raw = df.copy()
df_imp = df.copy()

# Covariates eligible for imputation (excluding primary keys and outcomes or key AB-test metrics)
covariates_to_impute = [
    "region",
    "gender",
    "education",
    "traffic_source",
    "peak_exposure_day",
    "pageviews_session",
    "age",
    "product_page_views",
    "time_on_site_sec",
    "ad_exposure_count",
    "peak_exposure_hour"
]

# Identify numeric vs categorical covariates
numeric_covs = (
    df_imp[covariates_to_impute]
    .select_dtypes(include=['int64', 'float64'])
    .columns
    .tolist()
)
print(numeric_covs)

cat_covs = (
    df_imp[covariates_to_impute]
    .select_dtypes(include=['object'])
    .columns
    .tolist()
)
print(cat_covs)


# -----------------------------
# Temporary imputation
# -----------------------------
# median for numeric variables and mode for categoricals
# placeholders so RF models can run
for c in numeric_covs:
    df_imp[c] = df_imp[c].fillna(df_imp[c].median())

for c in cat_covs:
    df_imp[c] = df_imp[c].fillna(df_imp[c].mode().iloc[0])


# -----------------------------
# Determine imputation order
# -----------------------------
# Variables are imputed from smallest to largest missingness to improve predictive accuracy 
missing_rate = df_imp[covariates_to_impute].isna().mean()
impute_order = (
    missing_rate[missing_rate > 0]
    .sort_values()
    .index
    .tolist()
)

print(impute_order)


# -----------------------------
# Iterative RF imputation
# -----------------------------
# For each target variable:
# - Restore NaN where values were originally missing
# - Use remaining covariates as predictors
# - Overwrite temporary median/mode values with RF predictions

for target in impute_order:
    features = [c for c in covariates_to_impute if c != target]

    # Replace placeholders with missings.
    # For column target, keep the current value where the original data df is not missing; 
    # otherwise replace it with NaN, i.e restore original missing values so RF replaces placeholders.
    df_imp[target] = df_imp[target].where(
        df[target].notna(),
        np.nan
    )

    if target in numeric_covs:
        df_imp = rf_impute_numeric(df_imp, target, features)
    else:
        df_imp = rf_impute_categorical(df_imp, target, features)


# Confirm no missing values remain in imputed covariates
print(df_imp[covariates_to_impute].isna().sum())

# Export cleaned data
df_imp.to_csv(
    "data/cleaneddata/cardio_rf_imputation_processed.csv",
    index=False
)

