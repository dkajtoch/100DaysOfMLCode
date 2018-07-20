import pandas as pd

# read probability and premium and Ids
data = pd.read_csv('./data/test_proba.csv')

def probability_change(p, x):
    from numpy import exp

    res = p*(1. + 0.2*( 1. - exp(-2.)*exp( 2.*exp(-x/400.) ) ) )
    return res if res<=1.0 else 1.0

def optimal_incentives(prob, premium):
    from scipy.optimize import minimize_scalar

    # ProgressBar
    from tqdm import tqdm
    ProgressBar = tqdm(range(len(prob)))

    opt = []
    for p, amount in zip(prob, premium):
        # formula given by McKinsey
        revenue = lambda x: -( amount*probability_change(p, x) - x )

        res=minimize_scalar(revenue,
                            bounds=(0., 1.0E+05),
                            method='bounded'
                           )

        opt.append(res.x)

        ProgressBar.update()

    return opt

# find opimal incentives
data['incentives'] = \
    optimal_incentives(data['renewal'].tolist()[:5], data['premium'].tolist()[:5])

data = data.drop('premium', axis=1)

# export submission
data.to_csv('./data/submission.csv', index=False)
