import pandas as pd
import numpy as np

# read full data
raw_data = pd.read_csv("accepted_2007_to_2017.csv")

# make a subset of data by randomly sampling (to work on it)
subset = raw_data.sample(n=int(len(raw_data)/100))

# pre-processing steps

# 1 - find out null values
subset.isnull().sum()
for i in subset.columns:
    if subset[i].isnull().sum() > (len(subset)/2):
        # drop columns that contains more than half null values
        subset = subset.drop(i, axis=1)
len(subset.columns)

# dropping columns that will be useless for EDA
subset = subset.drop(["sub_grade", "pymnt_plan", "purpose", "zip_code",
                      "addr_state"], axis=1)
