import pandas as pd
import numpy as np
import sys

# --------------------------------------------------
# Custom scorer
# --------------------------------------------------
def probability_change(p, x):
    from numpy import exp

    res = p*(1. + 0.2*( 1. - exp(-2.)*exp( 2.*exp(-x/400.) ) ) )
    return res if res<=1.0 else 1.0

def optimal_incentives(prob, premium):
    from scipy.optimize import minimize_scalar

    opt = []
    for p, amount in zip(prob, premium):
        # formula given by McKinsey
        revenue = lambda x: -( amount*probability_change(p, x) - x )

        res=minimize_scalar(revenue,
                            bounds=(0., 1.0E+05),
                            method='bounded'
                           )

        opt.append(res.fun)

    opt = np.array(opt)
    return -np.mean(opt)

def custom_score(y_true, proba, premium, lam=1./9000.):

    from sklearn.metrics import roc_auc_score

    #res = 0.7 * roc_auc_score(y_true, proba) + \
    #    0.3 * optimal_incentives(proba, premium) * lam

    res = optimal_incentives(proba, premium)

    return res
# --------------------------------------------------
# --------------------------------------------------

data_train = pd.read_csv('./stacked_train_proba.csv', usecols=['xgboost','RF','NN','renewal'])

X = data_train[['xgboost','RF','NN']]
y = data_train['renewal']

# premium
premium = pd.read_csv('../data/train.csv', usecols=['premium'])
premium = np.float64(premium['premium'].tolist())

# add extra feature
#dat = pd.read_csv('../data/train.csv', usecols=['perc_premium_paid_by_cash_credit'])
#X['extra'] = dat['perc_premium_paid_by_cash_credit']
# meta classifier
# --------------------------------------------------
if str(sys.argv[1]) == 'validate':

    print('Preparing for cross-validation')

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, log_loss

    clf = LogisticRegression(penalty='l2',
        C=1.0E-04
    )

    auc_tab  = np.array([])
    loss_tab = np.array([])
    custom_tab = np.array([])

    skf = StratifiedKFold(n_splits=4, random_state=1234)
    for train_index, test_index in skf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        premium_test = premium[test_index]
    
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_test)
    
        auc = roc_auc_score(y_test, proba[:,1])
        loss = log_loss(y_test, proba)
        custom = custom_score(y_test.tolist(), proba[:,1], premium_test)
    
        auc_tab  = np.append(auc_tab, auc)
        loss_tab = np.append(loss_tab, loss)
        custom_tab = np.append(custom_tab, custom)
        
    print('AUC: %.8f +/- %.8f' % (np.mean(auc_tab), np.std(auc_tab)))
    print('Loss: %.8f +/- %.8f' % (np.mean(loss_tab), np.std(loss_tab)))
    print('Custom: %.8f +/- %.8f' % (np.mean(custom_tab), np.std(custom_tab)))


elif str(sys.argv[1]) == 'predict':

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    clf = LogisticRegression(C=0.0001, penalty='l2')

    clf.fit(X, y)

    # Predict
    X_test = pd.read_csv('./stacked_test_proba.csv', usecols=['xgboost','RF','NN'])
    # add extra feature
    #dat = pd.read_csv('../data/test.csv', usecols=['perc_premium_paid_by_cash_credit'])
    #X_test['extra'] = dat['perc_premium_paid_by_cash_credit']

    print('Writing predictions')

    #proba = clf.best_estimator_.predict_proba(X_test)[:,1]
    proba = clf.predict_proba(X_test)[:,1]

    # export to a file
    export_data = pd.read_csv('../data/test.csv', usecols=['id', 'premium'])
    export_data.insert(loc=1, column='renewal', value=proba)

    export_data.to_csv('../data/test_proba.csv', index=False)

elif str(sys.argv[1]) == 'majority':

    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score, log_loss


    proba = np.float64(X.mean(axis=1).tolist())
    proba = np.column_stack((1.-proba, proba))
    
    auc_tab  = np.array([])
    loss_tab = np.array([])
    custom_tab = np.array([])

    skf = StratifiedKFold(n_splits=4, random_state=1234)
    for train_index, test_index in skf.split(np.zeros(len(y)),y):
        p = proba[test_index]
        y_test = y.iloc[test_index]
        premium_test = premium[test_index]
    
        auc = roc_auc_score(y_test, p[:,1])
        loss = log_loss(y_test, p)
        custom = custom_score(y_test.tolist(), p[:,1], premium_test)
    
        auc_tab  = np.append(auc_tab, auc)
        loss_tab = np.append(loss_tab, loss)
        custom_tab = np.append(custom_tab, custom)
        
    print('AUC: %.8f +/- %.8f' % (np.mean(auc_tab), np.std(auc_tab)))
    print('Loss: %.8f +/- %.8f' % (np.mean(loss_tab), np.std(loss_tab)))
    print('Custom: %.8f +/- %.8f' % (np.mean(custom_tab), np.std(custom_tab)))

    # Predict
#    X_test = pd.read_csv('./stacked_test_proba.csv', usecols=['xgboost','RF','NN'])
    # add extra feature
    #dat = pd.read_csv('../data/test.csv', usecols=['perc_premium_paid_by_cash_credit'])
    #X_test['extra'] = dat['perc_premium_paid_by_cash_credit']

#    print('Writing predictions')

    #proba = clf.best_estimator_.predict_proba(X_test)[:,1]
#    proba = np.float64(X_test.mean(axis=1))

    # export to a file
#    export_data = pd.read_csv('../data/test.csv', usecols=['id', 'premium'])
#    export_data.insert(loc=1, column='renewal', value=proba)

#    export_data.to_csv('../data/test_proba.csv', index=False)
