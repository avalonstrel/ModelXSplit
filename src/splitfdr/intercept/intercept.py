import numpy as np

from .split_lasso_sci import cv_split_lasso

def fixed_intercept(X, y, D, 
                    lambdas=None, nus=None, 
                    is_screen=False, n2=None,
                    nfolds=5, parallel_num=64):
    # if lambdas is None:
    #     lambdas = np.power(10, np.arange(0, -8, -0.4))
    # if nus is None:
    #     nus = np.power(10, np.arange(0, 2, 0.4))

    return cv_split_lasso(X, y, D, 
                          lambdas=lambdas, nus=nus,
                          is_screen=is_screen, n2=n2, 
                          nfolds=nfolds)

def path_intercept(X, y, D, 
                    nu, lambdas=None):
    # if lambdas is None:
    #     lambdas = np.power(10, np.arange(0, -8, -0.4))
    # if nus is None:
    #     nus = np.power(10, np.arange(0, 2, 0.4))
    p = D.shape[1]
    betas = split_lasso(X, y, D, nu, lambdas=lambdas, is_screen=False)
    betas = betas[:p, :]
    return betas




