from concurrent.futures import ProcessPoolExecutor
from functools import partial
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from glum import GeneralizedLinearRegressor, GeneralizedLinearRegressorCV

# from glmnet_python import glmnet
from splitfdr.construct import construct_splitknockoffs_matrix
from splitfdr.utils.misc import gamma_supp_select, get_W, hittingpoint


def split_lasso(data):
    X, y, lambda_val = data["X"], data["y"], data["lambda_val"]
    X, y = np.asarray(X), np.asarray(y)
    glm = GeneralizedLinearRegressor(
            family="normal",
            alpha=lambda_val,  # default
            l1_ratio=1.0,
            fit_intercept=True,
            max_iter=300,
        )
    glm.fit(X, y)
    return glm.coef_

class SplitKnockoffSolver:
    def __init__(self) -> None:
        pass

    def solve(self, X, y, lambdas, parallel_num=16):
        # fit = glmnet(x=X, y=y, alpha=1, 
        #                 lambdau=lambdas)
        # coef1 = np.array(fit['beta'])
        # coef1 = coef1.transpose()
        # return coef1
        coefs = np.stack([split_lasso({
                            "X":X, 
                            "y":y, 
                            "lambda_val":lambda_val
                        }) for lambda_val in lambdas])
        # with ProcessPoolExecutor(max_workers=128) as executor:
        #     results = list(
        #         executor.map(
        #             split_lasso,
        #             [
        #                 {
        #                     "X":X, 
        #                     "y":y, 
        #                     "lambda_val":lambda_val
        #                 }
        #                 for lambda_val in lambdas
        #             ],
        #         )
        #     )
        # coefs = np.stack(results)
        
        return coefs

def get_r_Z(tilde_y, A_beta, A_gamma, 
            beta_intercepts, lambdas, beta_fixed=False):
    m = A_gamma.shape[1]
    spkf_solver = SplitKnockoffSolver()
    if beta_fixed:
        assert len(beta_intercepts.shape) == 1, 'The length of beta intercepts should be 1, if beta_fixed is True'
        y_new = tilde_y - A_beta @ beta_intercepts
        coef1 = spkf_solver.solve(A_gamma, y_new, lambdas)
        
    else:
        coef1 = np.zeros((len(lambdas), m))
        for i in range(len(lambdas)):
            y_new = tilde_y - A_beta @ beta_intercepts[i, :]
            coef1[i] = spkf_solver.solve(A_gamma, y_new, lambdas[i])

    # print(coef1.shape)
    # calculate r and Z
    r, Z = np.zeros(m), np.zeros(m)
    for i in range(m):
        r_, Z_ = hittingpoint(coef1[:, i], lambdas)
        # print('coef1', coef1[:, i], r_, Z_)
        r[i], Z[i] = r_, Z_
    return r, Z


def compute_W_solpath_statistics(X, y, D, 
                        beta_intercepts, nu, lambdas, 
                        stats, W_types):

    A_beta, A_gamma, tilde_y, tilde_A_gamma = construct_splitknockoffs_matrix(X, y, D, nu)
    
    beta_fixed = (len(beta_intercepts.shape) == 1)
    # @TODO maybe parallel
    r, Z = get_r_Z(tilde_y, A_beta, A_gamma, 
                    beta_intercepts, lambdas, beta_fixed=beta_fixed)
    
    tr, tZ = get_r_Z(tilde_y, A_beta, tilde_A_gamma, 
                    beta_intercepts, lambdas, beta_fixed=beta_fixed)
    if 'gamma_supp' in stats:
        r, Z = gamma_supp_select(r, Z, stats['gamma_supp'])
        tr, tZ = gamma_supp_select(tr, tZ, stats['gamma_supp'])
    # store tilde_Z when it is positive
    # plt.plot(Z, tZ, 'ro')
    # plt.savefig('test.png')
    # W_types: ['bc','st'...]
    # methods:['knockoff', 'knockoff+']
    Ws = {W_type:get_W(r, tr, Z, tZ, W_type=W_type) for W_type in W_types}
    return Ws
