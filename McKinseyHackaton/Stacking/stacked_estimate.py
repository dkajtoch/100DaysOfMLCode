# ---------------------------
# General packages
# ---------------------------
import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from tqdm import tqdm
from scipy.stats import boxcox

# ---------------------------
# training collector
# ---------------------------
train_stack = pd.read_csv('../data/train.csv', usecols=['id', 'renewal'])
train_stack_name = 'stacked_train_proba.csv'

train_stack = train_stack.reindex(columns=['id', 'xgboost', 'RF', 'NN', 'renewal'])

# ---------------------------
# testing collector
# ---------------------------
test_stack = pd.read_csv('../data/test.csv', usecols=['id', 'premium'])
test_stack_name = 'stacked_test_proba.csv'

test_stack = test_stack.reindex(columns=['id', 'xgboost', 'RF', 'NN', 'premium'])

# ---------------------------
# Out-of-sample predictions
# ---------------------------
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, random_state=1234)

# ----------------------------
# training data
# ----------------------------
data_train = pd.read_csv('../data/train.csv')

# predictor
y = data_train['renewal']

# ----------------------------
# testing data
# ----------------------------
data_test = pd.read_csv('../data/test.csv')

# ----------------------------
# XGBoost
# ----------------------------
def features(data):
    X = data[[
    	'perc_premium_paid_by_cash_credit',
        'age_in_days',
        'Income',
        'Count_3-6_months_late',
        'Count_6-12_months_late',
        'Count_more_than_12_months_late',
        'no_of_premiums_paid',
        'application_underwriting_score'
    ]]

    # no strategy for missing values

    return X

X = features(data_train)

# model
from xgboost import XGBClassifier

clf = XGBClassifier(n_estimators=250,
    reg_alpha=12.,
    reg_lambda=0.,
    random_state=654
)

# fiting
ProgressBar = tqdm(range(10), desc='xgboost')

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)

    proba = np.float64(clf.predict_proba(X_test))[:,1]

    train_stack['xgboost'].iloc[test_index] = proba

    try:
        ProgressBar.update()
    except Exception as e:
        print(e)
        pass

try:
    train_stack.to_csv(train_stack_name, index=False)
    print('Successfully saved xgboost')
except Exception as e:
    print(e)
    pass

del ProgressBar

print('Refiting XGBoost on the whole training data')
clf.fit(X, y)
X_test = features(data_test)
proba = np.float64(clf.predict_proba(X_test))[:,1]
test_stack['xgboost'] = proba

try:
    test_stack.to_csv(test_stack_name, index=False)
    print('Successfully saved xgboost')
except Exception as e:
    print(e)
    pass

# ----------------------------
# Random Forest
# ----------------------------
def features(data):
    X = data[[
    	'perc_premium_paid_by_cash_credit',
        'age_in_days',
        'Income',
        'Count_3-6_months_late',
        'Count_6-12_months_late',
        'Count_more_than_12_months_late',
        'no_of_premiums_paid',
        'application_underwriting_score'
    ]]

    # Missing values strategy
    for name in X.columns.tolist():
        val = X[name].mean()#.iloc[0]
        X[name].fillna(val, inplace=True)

    return X

X = features(data_train)

# model
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=250,
    criterion='gini',
    min_samples_leaf=100,
    min_samples_split=3,
    random_state=1563
)

# fiting
ProgressBar = tqdm(range(10), desc='RF')

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train)

    proba = np.float64(clf.predict_proba(X_test))[:,1]

    train_stack['RF'].iloc[test_index] = proba

    try:
        ProgressBar.update()
    except Exception as e:
        print(e)
        pass

try:
    train_stack.to_csv(train_stack_name, index=False)
    print('Successfully saved RF')
except Exception as e:
    print(e)
    pass

del ProgressBar

print('Refiting RF on the whole training data')
clf.fit(X, y)
X_test = features(data_test)
proba = np.float64(clf.predict_proba(X_test))[:,1]
test_stack['RF'] = proba

try:
    test_stack.to_csv(test_stack_name, index=False)
    print('Successfully saved RF')
except Exception as e:
    print(e)
    pass

# ----------------------------
# Neural Network
# ----------------------------
def features(data):
    X = data[[
    	'perc_premium_paid_by_cash_credit',
        'Count_3-6_months_late',
        'Count_6-12_months_late',
        'Count_more_than_12_months_late',
        'no_of_premiums_paid',
        'application_underwriting_score'
    ]]

    X['Income'] = np.log(data['Income'])
    X['age_in_days'], lam = boxcox(data['age_in_days'])

    # Missing values strategy
    for name in X.columns.tolist():
        val = X[name].mean()#.iloc[0]
        X[name].fillna(val, inplace=True)

    return(X, lam)

X, lam = features(data_train)

# model
# ---------------------------------------
import random as rn
import tensorflow as tf
np.random.seed(654)
rn.seed(654)
tf.set_random_seed(654)
import os
os.environ['PYTHONHASHSEED'] = '0'
from keras import backend as K
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
# ---------------------------------------
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

input_dim = len(X.columns)

model = Sequential()
model.add(Dense(32, input_dim=input_dim, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(16, kernel_initializer='random_uniform', activation='relu'))
model.add(Dense(1, kernel_initializer='random_uniform', activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# combine scaler with neural network
clf = Pipeline([('feature', StandardScaler()), ('model', model)])

# fiting
ProgressBar = tqdm(range(10), desc='NN')

for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf.fit(X_train, y_train,
	    model__epochs=9,
	    model__batch_size=200,
	    model__verbose=False
	   )

    proba = np.float64(clf.predict_proba(X_test))

    train_stack['NN'].iloc[test_index] = proba.ravel()

    try:
        ProgressBar.update()
    except Exception as e:
        print(e)
        pass

try:
    train_stack.to_csv(train_stack_name, index=False)
    print('Successfully saved NN')
except Exception as e:
    print(e)
    pass

del ProgressBar

print('Refiting NN on the whole training data')
clf.fit(X, y,
        model__epochs=9,
        model__batch_size=200,
        model__verbose=False
       )
X_test, _ = features(data_test)
X_test['age_in_days'] = boxcox(data_test['age_in_days'], lmbda=lam)
proba = np.float64(clf.predict_proba(X_test))
test_stack['NN'] = proba

try:
    test_stack.to_csv(test_stack_name, index=False)
    print('Successfully saved NN')
except Exception as e:
    print(e)
    pass
