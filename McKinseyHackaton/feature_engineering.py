import pandas as pd
pd.options.mode.chained_assignment = None

data_train = pd.read_csv('./data/train.csv')
data_test = pd.read_csv('./data/test.csv')

#print(pd.DataFrame({'NoNaNs': data_train.isna().sum(),
#                    'Nodistinct': data_train.nunique()
#                   }))

# --------------------------
# Select features
# --------------------------
features = ['perc_premium_paid_by_cash_credit',
            'age_in_days',
            'Income',
            'Count_3-6_months_late',
            'Count_6-12_months_late',
            'Count_more_than_12_months_late',
            'no_of_premiums_paid',
#            'premium',
	    'application_underwriting_score'
           ]

new_data_train = data_train[features+['renewal']]
new_data_test  = data_test[features]

# --------------------------
# NaNs Strategy
# --------------------------
# I have checked earlier that NaNs are in Count...
# Mode fill

#new_data_train.fillna(new_data_train.mode().to_dict(), inplace=True)

#from scipy.stats import mode
#import numpy as np
for name in new_data_train.columns.tolist():
    val = new_data_train[name].mean()#.iloc[0]
    new_data_train[name].fillna(val, inplace=True)

for name in new_data_test.columns.tolist():
    val = new_data_test[name].mean()#.iloc[0]
    new_data_test[name].fillna(val, inplace=True)

# --------------------------
# Export data
# --------------------------
new_data_train.to_csv('./data/data_train.csv', index=False)
new_data_test.to_csv('./data/data_test.csv', index=False)
