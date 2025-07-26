import numpy as np
# from glmnet_python import glmnet, cvglmnet, glmnetPlot, glmnetPrint, cvglmnetCoef


def sigmoid(x):
    "Numerically-stable sigmoid function."
    if x >= 0:
        z = np.exp(-x)
        return 1 / (1 + z)
    else:
        z = np.exp(x)
        return z / (1 + z)
    
def normalize_func(X):
    if len(X.shape) == 2:
        X_mean = np.mean(X, axis=0, keepdims=True)
    else:
        X_mean = np.mean(X)
    X_cen = X - X_mean

    return X_cen

 
def normalize_col_func(X, norm_type='l2'):
    if len(X.shape) <= 2:
        X_mean = np.mean(X, axis=0, keepdims=True)
    X_cen = X - X_mean

    if norm_type == 'maxmin':
        X_range = X_cen.max() - X_cen.min()
        X_normed = (X_cen - X_cen.min()) / X_range
    elif norm_type == 'l2':
        n = X_cen.shape[0]
        X_norm = np.linalg.norm(X_cen, axis=0, keepdims=True)
        X_normed = X_cen / X_norm * np.sqrt(n-1)
        # X_normed = X_cen / X_norm 
    else:
        X_normed = X_cen
    return X_normed

def canonical_svd(mat):
    """
    Reduced SVD with canonical sign choice
       [U,S,V] = KNOCKOFFS.PRIVATE.CANONICALSVD(X)
    Computes a reduced SVD without sign ambiguity. Our convention is that
       the sign of each vector in U is chosen such that the coefficient
       with largest absolute value is positive.
    """
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    V = Vh.T
    n, m = mat.shape
    for j in range(min(n, m)):
        i = np.argmax(np.abs(U[:, j]))
        if U[i, j] < 0:
            U[:, j] = -U[:, j]
            V[:, j] = -V[:, j]
    return U, S, V

def decompose(X, randomize=False):
    """
    KNOCKOFFS.PRIVATE.DECOMPOSE  Decompose design matrix X for knockoff creation
    [U,S,V,U_perp] = KNOCKOFFS.PRIVATE.DECOMPOSE(X)
    [U,S,V,U_perp] = KNOCKOFFS.PRIVATE.DECOMPOSE(X, randomize)
    """
    n, p = X.shape
    assert n >= 2 * p, f'Data matrix must have n >= 2p, now n={n}, p={p}'
    
    U, S, V = canonical_svd(X)

    # Construct an orthogonal matrix U_perp such that U_perp'*X = 0.
    zeros_matrix = np.zeros((n, n - p))
    Q, _ = np.linalg.qr(np.concatenate((U, zeros_matrix), axis=1))
    U_perp = Q[:, p:] # n, n-p
    # if randomize:
    #     Q, _ = np.linalg.qr(np.random.randn(p, p), mode='economic')
    #     U_perp = U_perp @ Q
    
    return U, S, V, U_perp

def coef_norm(coef):
    return np.sqrt(np.sum(coef**2))

def group_hittingpoint(coef, lambdas):
    r, Z = 0, 0

    for i in range(len(lambdas)):
        if np.abs(coef_norm(coef[i])) > 1e-5:
            r = np.sign(coef_norm(coef[i]))
            Z = lambdas[i]
            break

    return r, Z

def hittingpoint(coef, lambdas):
    r, Z = 0, 0

    for i in range(len(lambdas)):
        if np.abs(coef[i]) != 0:
            r = np.sign(coef[i])
            Z = lambdas[i]
            break

    return r, Z

def get_W(r, tr, Z, tZ, W_type='st'):
    # print(r[:3], tr[:3], Z[:3], tZ[:3])
    Z_tilde = tZ * (r == tr)
    if W_type == 's':
        W = Z * np.sign(Z - tZ)
    elif W_type == 'st':
        W = Z * np.sign(Z - Z_tilde)
    elif W_type == 'bc':
        W = np.maximum(Z, tZ) * np.sign(Z - tZ)
    elif W_type == 'bct':
        W = np.maximum(Z, Z_tilde) * np.sign(Z - Z_tilde)
    return W

def gamma_supp_select(r, Z, gamma_supp):
    tmp = np.zeros_like(gamma_supp)
    tmp[gamma_supp] = r
    r = tmp

    tmp = np.zeros_like(gamma_supp)
    tmp[gamma_supp] = Z
    Z = tmp
    return r, Z

def random_split(X, y, split_ratio):
    n = X.shape[0]
    n1 = int(n * split_ratio)
    shuffled_index = np.random.permutation(n)
    split1_index = shuffled_index[:n1]
    split2_index = shuffled_index[n1:]
    X1, y1 = X[split1_index, :], y[split1_index] 
    X2, y2 = X[split2_index, :], y[split2_index] 
    return X1, y1, X2, y2

def random_split_all(*datas, split_ratio=0.2):
    n = datas[0].shape[0]
    n1 = int(n * split_ratio)
    shuffled_index = np.random.permutation(n)
    split1_index = shuffled_index[:n1]
    split2_index = shuffled_index[n1:]
    datas1 = [d[split1_index] for d in datas]
    datas2 = [d[split2_index] for d in datas]
    return *datas1, *datas2
    # X1, y1 = X[split1_index, :], y[split1_index] 
    # X2, y2 = X[split2_index, :], y[split2_index] 
    # return X1, y1, X2, y2


def threshold(W, q, method='knockoff'):
    """
    KNOCKOFFS.THRESHOLD  Compute the threshold for variable selection
      T = KNOCKOFFS.THRESHOLD(W, q) threshold using 'knockoff' method
      T = KNOCKOFFS.THRESHOLD(W, q, method) threshold with given method
          Inputs:
          W - statistics W_j for testing null hypothesis beta_j = 0.
          q - target FDR
          method - either 'knockoff' or 'knockoff+'
                   Default: 'knockoff'
          Outputs:
          T - threshold for variable selection
          See also KNOCKOFFS.SELECT.
    """
    offset = 1 if method in ['knockoff+', 'k+'] else 0
    # print(q, W)
    W_tmp = list(np.abs(W[W!=0])) + [0,]
    ts = sorted(W_tmp)
    ratios = np.array([(offset + np.sum(W <= -ts[i])) / max(1, np.sum(W >= ts[i]))
                       for i in range(len(ts))])
    # print('ratios', ratios, q)
    index = np.argwhere(ratios <= q)
    
    if len(index) == 0:
        T = np.inf
    else:
        T = ts[index[0][0]]
    # print('aaa', W,index[0][0], T)
    return T


def eval_threshold(W, q, method='knockoff'):
    """
    KNOCKOFFS.THRESHOLD  Compute the threshold for variable selection
      T = KNOCKOFFS.THRESHOLD(W, q) threshold using 'knockoff' method
      T = KNOCKOFFS.THRESHOLD(W, q, method) threshold with given method
          Inputs:
          W - statistics W_j for testing null hypothesis beta_j = 0.
          q - target FDR
          method - either 'knockoff' or 'knockoff+'
                   Default: 'knockoff'
          Outputs:
          T - threshold for variable selection
          See also KNOCKOFFS.SELECT.
    """ 
    # First original T determination. (1 + ...) / (...) <= alpha
    offset = 1 if method in ['knockoff+', 'k+'] else 0
    # print(q, W)
    W_tmp = list(np.abs(W[W!=0])) + [0,]
    ts = sorted(W_tmp)
    ratios = np.array([(offset + np.sum(W <= -ts[i])) / max(1, np.sum(W >= ts[i]))
                       for i in range(len(ts))])
    # print('ratios', ratios, q)
    index = np.argwhere(ratios <= q)
    
    if len(index) == 0:
        T1 = np.inf
    else:
        T1 = ts[index[0][0]]

    # Second T determination. sum(W_j >= T) < 1 / alpha
    ratios = np.array([np.sum(W >= ts[i]) 
                       for i in range(len(ts))])
    index = np.argwhere(ratios < (1 / q))
    if len(index) == 0:
        T2 = np.inf
    else:
        T2 = ts[index[0][0]]
    # print("Times:", T1, T2)
    return np.minimum(T1, T2)


def evalue_func(W, q, method):
    T = eval_threshold(W, q, method)

    evals = len(W) * (W >= T).astype(np.int32) / (1 + (W <= -T).sum())
    # print("lenW", len(W), "T", T, "Num -T", (1 + (W <= -T).sum()), "evals", evals)
    return evals 

def select_evalue(evals, q_e):
    p = len(evals)    
    evals_sort = sorted(evals, reverse=True)
    indexes_sort = (p / ((np.arange(p) + 1) * q_e))
    conditions = (evals_sort >= indexes_sort)
    # print("evals", evals_sort, "index", indexes_sort, "cond", conditions)

    rev_indices = np.flatnonzero(conditions[::-1])
    if len(rev_indices) > 0:
        max_k = p - rev_indices[0]
    else:
        max_k = 1e-8
    return np.argwhere(evals >= (p / (q_e * max_k))).reshape(-1)

def select(W, q, method):
    T = threshold(W, q, method)
    return np.argwhere(W >= T).reshape(-1)

