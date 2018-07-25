from __future__ import print_function
import numpy as np
from scipy.optimize import minimize_scalar

__all__ = ['effort_incentives', 'renewal_effort', 'renewal_incentives',
           'revenue', 'optimal_revenue'
          ]

def effort_incentives(x):
    """
    Equation for the effort-incentives curve

    Parameters
    ==========

    x: float
        Effort value

    Returns
    =======

    res: float
        Incentives amount
    """
    res = 10. * (1. - np.exp(-x/4. * 1.0E-02))

    return res

def renewal_effort(x):
    """
    Equation for % improvements in renewal prob vs effort

    Parameters
    ==========

    x: float
        Effort value

    Returns
    =======

    res: float
        Percent improvement
    """
    res = 20. * (1. - np.exp(-x/5.))

    return res

def renewal_incentives(x):
    """
    Equation for % improvements in renewal prob vs incentives

    Parameters
    ==========

    x: float
        Incentives

    Returns
    =======

    res: float
        Percent improvement
    """
    res = renewal_effort(effort_incentives(x))

    return res

def revenue(x, premium, p=1.):
    """
    Net revenue vs incentives for given premium and base probability p

    Parameters
    ==========

    x: float
        Incentives

    premium: float
        Premium value

    p: float
        Base probability

    Returns
    =======

    res: float
        Net revenue
    """

    # change in probability
    proba = p*(1. + renewal_incentives(x)/100.)
    proba = proba if proba<=1.0 else 1.0

    return premium * proba - x

def optimal_revenue(premium, p=1.):
    """
    Optimal revenue over incentives

    Parameters
    ==========

    premium: float
        Premium value

    p: float
        Base probability

    Returns
    =======
    
    (res.x, res.fun): (float, float)
        optimal argument, optimal value
    """

    fun = lambda x: -revenue(x,premium,p)

    res = minimize_scalar(fun,
        bounds=(0., 1.0E+05),
        method='bounded'
    )

    return(res.x, -res.fun)

if __name__ == '__main__':
    print(effort_incentives(0.1))
    print(renewal_effort(0.1))
    print(renewal_incentives(0.1))
    print(revenue(0.1, 200, 0.5))
    print(optimal_revenue(200,0.5))
