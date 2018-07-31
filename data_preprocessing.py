import pandas as pd
import numpy as np

# read full data
raw_data = pd.read_csv("accepted_2007_to_2017.csv")

# make a subset of data by randomly sampling (to work on it)
# subset = raw_data.sample(n=int(len(raw_data)/100))

# pre-processing steps

# 1 - find out null values
# raw_data.isnull().sum()
raw_data.info(verbose=True, null_counts=True)
raw_data.head()

for i in raw_data.columns:
    if raw_data[i].isnull().sum() > (len(raw_data)/2):
        # drop columns that contains more than half null values
        raw_data = raw_data.drop(i, axis=1)
len(raw_data.columns)

# dropping columns that will be useless for EDA
# subset = subset.drop(["sub_grade", "pymnt_plan", "purpose", "zip_code",
#                       "addr_state"], axis=1)

# lets take a look at the loan_status variable of the data
raw_data["loan_status"].value_counts(dropna=False)

# in the output of above code, we can see that there are many status levels, but
# for this analysis we will be looking at loans that are either charged off or fully paid
# so, we will be dripping the rows that have status different than "Charged off" or "Fully Paid"

raw_data = raw_data.loc[raw_data["loan_status"].isin(["Fully Paid", "Charged Off"])]
print("We are now left with %d rows" % len(raw_data))

# check for NANs one more time
raw_data.info(verbose=True, null_counts=True)

# remove columns with NANs and columns that we don't need for further analysis
raw_data = raw_data.drop(["next_pymnt_d", "debt_settlement_flag", "disbursement_method", "hardship_flag",
                          "pymnt_plan", "title", ], axis=1)

# take a look at the basic stats of the numerical valued columns
raw_data.describe()

# drop the columns that do not have any significant amount of data related to them
# after going through every feature, i have compiled the following list of features to keep
# although i may have missed some important ones, but that wii not put any significant effect on the analysis
to_keep = ['addr_state', 'annual_inc', 'application_type', 'dti', 'earliest_cr_line', 'emp_length',
             'emp_title', 'fico_range_high', 'fico_range_low', 'grade', 'home_ownership', 'id',
             'initial_list_status', 'installment', 'int_rate', 'issue_d', 'loan_amnt', 'loan_status',
             'mort_acc', 'open_acc', 'pub_rec', 'pub_rec_bankruptcies', 'purpose', 'revol_bal',
             'revol_util', 'sub_grade', 'term', 'title', 'total_acc', 'verification_status', 'zip_code']

to_drop = [feat for feat in raw_data.columns if feat not in to_keep]
len(to_drop) # to be dropped

raw_data = raw_data.drop(to_drop, axis=1)
print("We are left with : ", raw_data.shape, "i.e %d rows(loans) and %d columns(features)" %(raw_data.shape))

# get rid of NANs everywhere
raw_data = raw_data.dropna()
raw_data.info(verbose=True, null_counts=True)

# this is the final data that we will be using for out analysis
raw_data.to_csv("cleaned_data.csv")