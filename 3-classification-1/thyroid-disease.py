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

# %% [markdown] {"jupyter": {"outputs_hidden": true}}
# # Thyroid Disease Classification
#
# Thyroid disease records supplied by the Garavan Institute and J. Ross Quinlan, New South Wales Institute, Syndney, Australia, 1987
#
# http://archive.ics.uci.edu/ml/datasets/thyroid+disease
#
# **Attribute Information:**
#
# Classes: replacement therapy, underreplacement, overreplacement, negative
#
# - age: continuous.
# - sex: M, F.
# - on thyroxine: f, t.
# - query on thyroxine: f, t.
# - on antithyroid medication: f, t.
# - sick: f, t.
# - pregnant: f, t.
# - thyroid surgery: f, t.
# - I131 treatment: f, t.
# - query hypothyroid: f, t.
# - query hyperthyroid: f, t.
# - lithium: f, t.
# - goitre: f, t.
# - tumor: f, t.
# - hypopituitary: f, t.
# - psych: f, t.
# - TSH measured: f, t.
# - TSH: continuous.
# - T3 measured: f, t.
# - T3: continuous.
# - TT4 measured: f, t.
# - TT4: continuous.
# - T4U measured: f, t.
# - T4U: continuous.
# - FTI measured: f, t.
# - FTI: continuous.
# - TBG measured: f, t.
# - TBG: continuous.
# - referral source: WEST, STMW, SVHC, SVI, SVHD, other.

# %% [markdown]
# ## Imports

# %%
import numpy as np
import pandas as pd
import pandas_profiling

# %% [markdown]
# ## EDA & preparation

# %%
fpath = "data/dataset_57_hypothyroid.csv"
df = pd.read_csv(fpath)

df.replace("?", np.nan, inplace=True)
numerical_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
df[numerical_cols] = df[numerical_cols].apply(pd.to_numeric)

# %%
profile = df.profile_report(title="Thyroid Disease Dataset", sort="None")
profile.to_file(output_file="thyroid-disease-eda.html")
profile

# %% [markdown]
# We'll perform following steps as data preparation:

# %%
# drop TBG_measured and TBG because they're constant
df.drop(columns=["TBG_measured", "TBG"], inplace=True)
numerical_cols.remove("TBG")


# %%
# standardize numerical columns
def scale(x):
    centered = x - x.mean()
    return centered / x.std()


df[numerical_cols] = df[numerical_cols].apply(scale)

# %%
df.sample(3)

# %%
