import numpy as np
from glum import GeneralizedLinearRegressorCV, GeneralizedLinearRegressor
from multiprocessing import Pool
from functools import partial
from splitfdr.utils.misc import normalize_func


def trans_data(X, y, D, nu):
    """
    To solve the 1/n |y - X beta|^2 + 1/nu |D beta - gamma|^2 + lambda |gamma|_1. 
    By translating the lasso with transformational sparsity into a partial penalty lasso problem
    and solve it by glmnet_python
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix
    y : np.ndarray
        ``(n,)``-shaped response vector
    D : np.ndarrays
        ``(m, p)``-shaped transformation matrix
    nu: float
        the weight parameter of transformational sparisity
    lambdau: np.ndarrays
        potential lambda for cross validation.
    """
    # Transform the original problem in new a new way
    # A_beta = [X / \sqrt{n}; D/\sqrt{\nu}]
    # A_gamma = [0;-I_m/\sqrt{\nu}]
    # tilde_y = [y/\sqrt{n};0]
    n, p = X.shape
    m = D.shape[0]
    print(X.shape, y.shape, D.shape)
    # A_beta shape [n+m, p]
    A_beta = np.concatenate([X / np.sqrt(n), D / np.sqrt(nu)], axis=0)
    # A_gamma shape [n+m, m]
    A_gamma = np.concatenate([np.zeros((n, m)), - np.eye(m) / np.sqrt(nu)], axis=0)
    # tilde_y shape [n+m]
    tilde_y = np.concatenate([y / np.sqrt(n), np.zeros(m)], axis=0)

    penalty = np.ones(p+m)
    penalty[:p] = 0
    
    X = np.concatenate([A_beta, A_gamma], axis=1)
    y = tilde_y

    return X, y, penalty

def split_lasso(X, y, D, nu, lambdas=None, is_screen=False):
    """
    To solve the 1/n |y - X beta|^2 + 1/nu |D beta - gamma|^2 + lambda |gamma|_1. 
    By translating the lasso with transformational sparsity into a partial penalty lasso problem
    and solve it by glmnet_python
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix
    y : np.ndarray
        ``(n,)``-shaped response vector
    D : np.ndarrays
        ``(m, p)``-shaped transformation matrix
    nu: float
        the weight parameter of transformational sparisity
    lambdau: np.ndarrays
        potential lambda for cross validation.
    """

    X, y, penalty = trans_data(X, y, D, nu)
    glm = GeneralizedLinearRegressorCV(
            family="normal",
            alphas=lambdas,  # default
            l1_ratio=1.0,
            fit_intercept=True,
            max_iter=300,
            P1=penalty,
        )
    glm.fit(X, y)
    betas = glm.coef_
    return betas

def cv_split_lasso(X, y, D, 
                   lambdas=None, nus=None, 
                   is_screen=False, n2=None,
                   nfolds=5):
    """
    To solve the 1/n |y - X beta|^2 + 1/nu |D beta - gamma|^2 + lambda |gamma|_1. 
    By translating the lasso with transformational sparsity into a partial penalty lasso problem
        and solve it by glmnet_python
    Cross validation choose on lambda and nu.
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix
    y : np.ndarray
        ``(n,)``-shaped response vector
    D : np.ndarrays
        ``(m, p)``-shaped transformation matrix
    lambdas: np.ndarrays
        potential lambda for cross validation. 
        If `None`, will set it to np.power(10, np.arange(-1, -8, -0.1))
    nus: np.ndarrays
        potential nus for cross validation. 
        If `None`, will set it to np.power(10, np.arange(-1, -8, -0.1))
    nfolds: int
        the number of folds of the cross validation.
    """
    if is_screen:
        X, y, beta_supp = screen(X, y, D)

    if lambdas is None:
        lambdas = np.power(10, np.arange(0, -8, -0.4))
    if nus is None:
        nus = np.power(10, np.arange(0, 2, 0.4))

    n, m, p = X.shape[0], D.shape[0], X.shape[1]

    # Construct train and test based on the original data.
    test_num = int(n / nfolds)
    shuffled_index = np.random.permutation(np.arange(n))
    test_index = shuffled_index[(test_num):(2)*(test_num)]
    train_index = np.ones(n, dtype=np.bool8)
    train_index[test_index] = False
    X_train, y_train = X[train_index], y[train_index]
    X_test, y_test = X[test_index], y[test_index]
    X_train, y_train = normalize_func(X_train), normalize_func(y_train)
    X_test, y_test = normalize_func(X_test), normalize_func(y_test)
    
    loss_cv = np.zeros((len(nus), len(lambdas))) + 1e+8
    
    for nu_i, nu_val in enumerate(nus):
        X_, y_, penalty = trans_data(X_train, y_train, D, nu_val)
        glmcv = GeneralizedLinearRegressorCV(
            family="normal",
            alphas=lambdas,  # default
            l1_ratio=1.0,
            fit_intercept=True,
            max_iter=300,
            P1=penalty,
            cv=nfolds,
        )
        glmcv.fit(X_, y_)
        # print(f"Chosen alpha:    {glmcv.alpha_}")
        # print(f"Chosen l1 ratio: {glmcv.l1_ratio_}")
        betas = glmcv.coef_
        # print(betas)
        # sss
        lam_i = np.argwhere(lambdas == glmcv.alpha_)[0, 0]

        # use the model to predict on our test data
        # print("Shape", betas.shape, lam_i)
        y_hat = X_test @ betas[:p] 
        # print(y_hat.shape, y_test.shape)
        loss_cv[nu_i, lam_i] = np.mean((y_hat - y_test)**2)
    return standard_cv_select(loss_cv, X, y, D, lambdas, nus)

def screen(X, y, D):
    # screen for a estimated support set of beta
    lambdas = np.power(10, np.arange(0, -6, -0.1))
    cvfit = cvglmnet(x=X, y=y, ptype='mse',
                     lambdau=lambdas,
                     nfolds=5, parallel=True)
    beta_hat = cvglmnetCoef(cvfit, s='lambda_min')
    beta_supp = np.where(np.abs(beta_hat) >= 10**(-2))
    return X[:, beta_supp], D[:, beta_supp], beta_supp


def standard_cv_select(loss_cv, X, y, D, lambdas, nus):
    p = X.shape[1]
    min_index = np.argwhere(loss_cv == np.min(loss_cv))
    
    nu_ind, lam_ind = min_index[0]
    nu_val, lambda_val = nus[nu_ind], lambdas[lam_ind]
    X, y, penalty = trans_data(X, y, D, nu_val)
    # print('loss', loss_cv, min_index, nu_val, lambda_val)
    glm = GeneralizedLinearRegressor(
            family="normal",
            alpha=lambda_val,  # default
            l1_ratio=1.0,
            fit_intercept=True,
            max_iter=300,
            P1=penalty,
        )
    glm.fit(X, y)
    betas = glm.coef_
    # print('Chosen', betas, lambdas, nus, lambda_val, nu_val)
    # print('beta', betas[:p])
    return betas[:p], nu_val, lambda_val

def screen_cv_select(loss_cv, X, y, D, beta_supp, n2, lambdas, nus):
    mean_loss_cv = np.mean(loss_cv, axis=0)
    std_loss_cv = np.std(loss_cv, axis=0)
    p = X.shape[1]
    min_index = np.argwhere(mean_loss_cv <= np.min(mean_loss_cv + std_loss_cv))
    nu_ind, lam_ind = min_index[0]
    nu_val, lambda_val = nus[nu_ind], lambdas[lam_ind]
    betas = split_lasso(X, y, D, nu_val, lambdas=np.array([lambda_val]), screen=True)
    beta = betas[:p, 0]
    gamma = betas[p:, 0]
    gamma_supp = np.where(np.abs(gamma) >= 10**(-2))
    if np.sum(beta_supp) + np.sum(gamma_supp) > n2:
        gamma_sort = sorted(np.abs(gamma), reverse=True)
        th = gamma_sort[n2 - np.sum(beta_supp)]
        gamma_supp = np.where(np.abs(gamma) > th)
    return beta, nu_val, lambda_val, beta_supp, gamma_supp

if __name__ == "__main__":
    n, m, p = 500, 100, 100
    X = np.random.randn(n, p)
    y = np.random.randn(n)
    D = np.random.randn(m, p)
    nus, lambdas = None, None
    # split_lasso(X, y, D, nu)
    import time
    start_time = time.time()
    cv_split_lasso(X, y, D, lambdas=None, nus=None, nfolds=10, parallel_num=128)
    print('used time', time.time()-start_time)