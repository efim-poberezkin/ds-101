# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: ds-101
#     language: python
#     name: ds-101
# ---

# %% [markdown] {"toc-hr-collapsed": false}
# # Regression analysis of Air Quality dataset
#
# https://archive.ics.uci.edu/ml/datasets/Air+Quality#
#
# **Attribute Information:**
#
# - Date (DD/MM/YYYY)
# - Time (HH.MM.SS)
# - CO(GT) - True hourly averaged concentration CO in mg/m^3 (reference analyzer)
# - NMHC(GT) - True hourly averaged overall Non Metanic HydroCarbons concentration in microg/m^3 (reference analyzer)
# - C6H6(GT) - True hourly averaged Benzene concentration in microg/m^3 (reference analyzer)
# - NOx(GT) - True hourly averaged NOx concentration in ppb (reference analyzer)
# - NO2(GT) - True hourly averaged NO2 concentration in microg/m^3 (reference analyzer)
# - PT08.S1(CO) - (tin oxide) hourly averaged sensor response (nominally CO targeted)
# - PT08.S2(NMHC) - (titania) hourly averaged sensor response (nominally NMHC targeted)
# - PT08.S3(NOx) - (tungsten oxide) hourly averaged sensor response (nominally NOx targeted)
# - PT08.S4(NO2) - (tungsten oxide) hourly averaged sensor response (nominally NO2 targeted)
# - PT08.S5(O3) - (indium oxide) hourly averaged sensor response (nominally O3 targeted)
# - T - Temperature in Â°C
# - RH - Relative Humidity (%)
# - AH - Absolute Humidity
#
# **Target:** С6H6(GT)

# %% [markdown]
# ## Imports

# %%
from statistics import mean

import numpy as np
import pandas as pd
import pandas_profiling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split

# %% [markdown]
# ## EDA

# %% [markdown]
# Since we want to predict C6H6(GT) based on gas multisensor device responses first we'll get rid of other true hourly concentrations coming from reference analyzer. We'll also replace values tagged as -200 with numpy NaN to enable adequate automated Pandas Profiling EDA.

# %%
fpath = "data/AirQualityUCI.xlsx"
df = pd.read_excel(fpath)
df.drop(columns=["CO(GT)", "NMHC(GT)", "NOx(GT)", "NO2(GT)"], inplace=True)
df.replace(-200, np.nan, inplace=True)

profile = df.profile_report(title="Air Quality")
profile.to_file(output_file="air-quality-eda.html")
profile

# %% [markdown]
# Pandas Profiling offers to reject PT08.S2(NMHC) as it's highly correlated with C6H6(GT), but since the latter is our target we won't do that.
#
# Other than that we can see that our numerical columns are more or less normally distributed and are without significant outliers.

# %% [markdown]
# ## Data preparation

# %% [markdown]
# As a part of preprocessing pipeline we'll drop missing values, since there's only 3.9% of them and we'll still have plenty of data left for analysis, and because we don't have sufficient knowledge of subject area to competently fill them in.
#
# We'll also standardize our dataframe except for Date and Time columns.

# %%
df.dropna(inplace=True)


def scale(x):
    centered = x - x.mean()
    return centered / x.std()


numerical_cols = [
    "C6H6(GT)",
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]
df[numerical_cols] = df[numerical_cols].apply(scale)

df.sample(3)


# %% [markdown]
# ## Baseline model

# %% [markdown]
# We'll evaluate training results on cross validation and on a test set. Since our data is well distributed MSE should be a fine metric to perform evaluation on, but as a means to double check and to interpret how well our model explains data we'll be looking at $R^{2}$ metric as well.

# %%
def evaluate(reg, X_train_, X_test_, y_train_, y_test_):
    reg.fit(X_train_, y_train_)

    # Calculate metrics
    y_pred = reg.predict(X_test_)
    mse = mean_squared_error(y_test_, y_pred)
    r2 = r2_score(y_test_, y_pred)

    # Build weights dataframe
    weights = pd.DataFrame(zip(["Bias"] + list(X_train_), [reg.intercept_] + reg.coef_))

    return mse, r2, weights


# %%
def print_metrics_and_weights(mse, r2, weights):
    print(f"MSE: {mse:.4f}")
    print(f"R2: {r2:.4f}")
    print(
        weights.transpose().to_string(
            header=False, index=False, float_format=lambda x: f"{x:.4f}"
        )
    )


# %%
def evaluate_on_cv(reg, X, y):
    cv = KFold(n_splits=5, random_state=37)
    mse_scores, r2_scores = [], []
    k = 0

    for train, test in cv.split(X, y):
        k += 1
        print(f"\n--- Fold {k} ---")
        mse, r2, weights = evaluate(
            reg, X.iloc[train], X.iloc[test], y.iloc[train], y.iloc[test]
        )
        mse_scores.append(mse)
        r2_scores.append(r2)
        print_metrics_and_weights(mse, r2, weights)

    print(f"\nAverage MSE on cross validation: {mean(mse_scores):.4f}")
    print(f"Average R2 on cross validation: {mean(r2_scores):.4f}")


# %%
def evaluate_on_test(reg, X_train_, X_test_, y_train_, y_test_):
    mse, r2, weights = evaluate(reg, X_train_, X_test_, y_train_, y_test_)
    print_metrics_and_weights(mse, r2, weights)


# %% [markdown]
# Before performing cross validation we'll split our data into train and test batches.

# %%
target = "C6H6(GT)"
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns=target), df[target], test_size=0.3, random_state=42
)

# %% [markdown]
# For a baseline model we'll drop Date and Time columns and use a simple linear regression without regularization.

# %%
X_train_baseline = X_train.drop(columns=["Date", "Time"])
X_test_baseline = X_test.drop(columns=["Date", "Time"])

linear_reg = LinearRegression()
evaluate_on_cv(linear_reg, X_train_baseline, y_train)

# %%
evaluate_on_test(linear_reg, X_train_baseline, X_test_baseline, y_train, y_test)

# %% [markdown]
# We can see that even the baseline linear regression model fits our data quite well and according to $R^{2}$ scores we got it explains over 97% of variance in C6H6(GT).
