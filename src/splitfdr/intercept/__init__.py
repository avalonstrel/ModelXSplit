from .intercept import fixed_intercept, path_intercept
import numpy as np

def get_intercept(X, y, D, nu, lambdas, 
                  n2=None, method='cv_all'):
    """
    Return:
    betas[len(lambdas), p]
    """
    assert method in ['cv_all', 'path']
    n1, m, p = X.shape[0], D.shape[0], X.shape[1]
    is_screen = False

    if n1 < p or (n2) < p + m:
        is_screen = True
    # print(n1, n2, p, m, is_screen)
    if method == 'cv_all':
        if is_screen:
            # @TODO 
            # include the nu into the search range
            beta, nu_val, lambda_val, beta_supp, gamma_supp = fixed_intercept(X, y, D, 
                                                                            lambdas=None, nus=None,
                                                                            is_screen=True, n2=n2)
            return beta, {'lambda':lambda_val, 'nu':nu_val,
                            'beta_supp':beta_supp, 'gamma_supp':gamma_supp}
        else:
            # @TODO 
            # include the nu into the search range
            beta, nu_val, lambda_val = fixed_intercept(X, y, D, lambdas=None, nus=None)
            return beta, {'lambda':lambda_val, 'nu':nu_val,}
    
    elif method == 'path':
        betas = path_intercept(X, y, D, 
                                nu, lambdas=None)
        return betas, {}
        

def get_group_intercept(X, y, D, groups, nu, lambdas, 
                  n2=None, method='cv_all'):
    """
    Return:
    betas[len(lambdas), p]
    """
    assert method in ['cv_all', 'path']
    n1, m, p = X.shape[0], D.shape[0], X.shape[1]
    is_screen = False

    if n1 < p or (n2) < p + m:
        is_screen = True
    # print(n1, n2, p, m, is_screen)
    if method == 'cv_all':
        if is_screen:
            # @TODO 
            # include the nu into the search range
            beta, nu_val, lambda_val, beta_supp, gamma_supp = fixed_intercept(X, y, D, 
                                                                            lambdas=None, nus=None,
                                                                            is_screen=True, n2=n2)
            return beta, {'lambda':lambda_val, 'nu':nu_val,
                            'beta_supp':beta_supp, 'gamma_supp':gamma_supp}
        else:
            # @TODO 
            # include the nu into the search range
            beta, nu_val, lambda_val = fixed_intercept(X, y, D, lambdas=None, nus=None)
            return beta, {'lambda':lambda_val, 'nu':nu_val,}
    
    elif method == 'path':
        betas = path_intercept(X, y, D, 
                                nu, lambdas=None)
        return betas, {}
        




    
    
    



    
    
    