import numpy as np

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