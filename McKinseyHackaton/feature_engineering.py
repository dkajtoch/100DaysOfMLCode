import pandas as pd

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
            'premium'
           ]

new_data_train = data_train[features+['renewal']]
new_data_test  = data_test[features]

# --------------------------
# NaNs Strategy
# --------------------------
# I have checked earlier that Nans are in Count...
# Mode fill
new_data_train = new_data_train.fillna( new_data_train.mode() )
new_data_test = new_data_test.fillna( new_data_test.mode() )

# --------------------------
# Export data
# --------------------------
new_data_train.to_csv('./data/data_train.csv', index=False)
new_data_test.to_csv('./data/data_test.csv', index=False)
