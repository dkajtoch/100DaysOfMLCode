import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss
import numpy as np
import pandas as pd
import sys

# ---------------------------
# read already prepared data
# ---------------------------
data_train = pd.read_csv('./data/data_train.csv')
y = data_train['renewal']
X = data_train.drop('renewal', axis=1)

if str(sys.argv[1]) == 'tuning':
    print('Preparing for hyperparameter tuning')

    from sklearn.model_selection import GridSearchCV

    param_grid = {'reg_alpha':np.linspace(0.,10.,5),
                  'reg_lambda':np.linspace(0.,10.,5),
                  'n_estimators':[15,20,40,60],
                  'seed':[1337]
                 }

    clf = GridSearchCV(estimator,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=10
    )

    clf.fit(X, y)

    # print results
    print('Best AUC: %.5f' % clf.best_score_)
    print('Best parameter set: %s' % clf.best_params_)

    # read test data
    X_test = pd.read_csv('./data/data_test.csv')
    proba = np.float64(clf.best_estimator_.predict_proba(X_test))

    # export to a file
    export_data = pd.read_csv('./data/test.csv', usecols=['id'])
    export_data['renewal'] = proba[:,1]
    export_data['premium'] = X_test['premium']

    export_data.to_csv('./data/test_proba.csv', index=False)

elif str(sys.argv[1]) == 'testing':
    print('Preparing for testing')

    from sklearn.cross_validation import train_test_split

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.2, random_state=1543)

    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from xgboost import XGBClassifier

    clf = XGBClassifier(n_estimators=150)

    clf.fit(X_train, y_train,
            eval_metric=['auc','logloss'],
            eval_set=[(X_train, y_train),(X_test, y_test)],
            early_stopping_rounds=None,
            verbose=False
           )

    results = estimator.evals_result()

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_context('notebook')
    
    plt.close()

    fig, ax = plt.subplots(figsize=(15,5),
        nrows=1, ncols=2, gridspec_kw={'wspace':0.2}
    )

    ax[0].plot(results['validation_0']['auc'], label='train')
    ax[0].plot(results['validation_1']['auc'], label='test')
    ax[0].set_xlabel('Number of iterations')
    ax[0].set_ylabel('AUC')
    ax[0].legend()

    ax[1].plot(results['validation_0']['logloss'], label='train')
    ax[1].plot(results['validation_1']['logloss'], label='test')
    ax[1].set_xlabel('Number of iterations')
    ax[1].set_ylabel('Log Loss')
    ax[1].legend()

    plt.show()

elif str(sys.argv[1] == 'predict'):
    print('Preparing for prediction')

    clf = XGBClassifier(n_estimators=50)

    clf.fit(X, y)

    # read test data
    X_test = pd.read_csv('./data/data_test.csv')
    proba = np.float64(clf.predict_proba(X_test))

    # export to a file
    export_data = pd.read_csv('./data/test.csv', usecols=['id'])
    export_data['renewal'] = proba[:,1]
    export_data['premium'] = X_test['premium']

    export_data.to_csv('./data/test_proba.csv', index=False)
