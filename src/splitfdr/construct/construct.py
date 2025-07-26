import numpy as np
from splitfdr.utils.misc import canonical_svd, decompose
from scipy.sparse import csr_matrix
import cvxpy as cp

def construct_splitknockoffs_matrix(X, y, D, nu):
    """
    This function gives the variable splitting design matrix
    [A_beta, A_gamma] and response vector tilde_y. It will also create a
    knockoff copy for A_gamma if required.
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix
    y : np.ndarray
        ``(n,)``-shaped response vector
    D : np.ndarrays
        ``(m, p)``-shaped transformation matrix
    nus: np.ndarrays
        potential nus for cross validation. 
        If `None`, will set it to np.power(10, np.arange(-1, -8, -0.1))
    """
    n, m, p = X.shape[0], D.shape[0], X.shape[1]
    assert m > 0, f"Not valid transformed dimension {m}"

    # A_beta shape [n+m, p]
    A_beta = np.concatenate([X / np.sqrt(n), D / np.sqrt(nu)], axis=0)
    # A_gamma shape [n+m, m]
    A_gamma = np.concatenate([np.zeros((n, m)), - np.eye(m) / np.sqrt(nu)], axis=0)
    # tilde_y shape [n+m]
    tilde_y = np.concatenate([y / np.sqrt(n), np.zeros(m)], axis=0)

    # construct knockoff copy
    # s_size = 2 - eta
    Sigma_bb = A_beta.T @ A_beta

    rank = np.linalg.matrix_rank(Sigma_bb)
    if rank == len(Sigma_bb):
        Sigma_bb_inv = np.linalg.inv(Sigma_bb)
    else:
        Sigma_bb_inv = np.linalg.pinv(Sigma_bb)
    

    Sigma_gg = A_gamma.T @ A_gamma
    Sigma_gb = A_gamma.T @ A_beta
    Sigma_bg = Sigma_gb.T

    # calculate C_nu
    C = Sigma_gg - Sigma_gb @ Sigma_bb_inv @ Sigma_bg
    C = (C + C.T) / 2
    C_inv = np.linalg.inv(C)

    # generate s
    C_eigvals, _ = np.linalg.eigh(C)
    C_eig = np.min(C_eigvals)
    diag_s = csr_matrix(np.diag(min(2 * C_eig, 1 / nu) * np.ones(C.shape[0])))
    
    # calculate K^T K = 2S-S C_nu^{-1} S
    KK = 2 * diag_s - diag_s @ C_inv @ diag_s
    KK = (KK + KK.T) / 2
    KK_eval, KK_evec = np.linalg.eigh(KK)
    
    K = KK_evec @ np.diag(np.sqrt(KK_eval+1e-10)) @ KK_evec.T
    
    # calculate U=[U1;U_2] where U_2 = 0_m* m
    # U_1 is an orthogonal complement of X
    # @TODO make sure the utilization of the decompose n-p ? p
    _, _, _, U_perp = decompose(X)
    U_1 = U_perp[:, :m]
    U = np.concatenate([U_1, np.zeros((m,m))], axis=0)
    
    # calculate sigma_beta beta^{-1} sigma_beta gamma
    short = Sigma_bb_inv @ Sigma_bg
    tilde_A_gamma = A_gamma @ (np.eye(m) - C_inv @ diag_s) \
                    + A_beta @ short @ C_inv @ diag_s \
                    + U @ K
    A_gamma = np.real(A_gamma)
    tilde_A_gamma = np.real(tilde_A_gamma)
    
    return A_beta, A_gamma, tilde_y, tilde_A_gamma


def optimize_alpha(Sigma, Sigma_M, D):
    alpha = cp.Variable(1)
    constraints = [(Sigma - alpha * D.T @ Sigma_M @ D) >> 0]
    # DTM = alpha * D.T @ Sigma_M  # (p, m)
    # Sigma_inv = np.linalg.inv((Sigma + Sigma.T) / 2)
    # constraints = [(alpha * Sigma_M - DTM.T @ Sigma_inv @ DTM) >> 0]
    prob = cp.Problem(cp.Maximize(alpha), constraints)

    prob.solve()

    return alpha.value[0] 


def generate_M(X, D, Sigma, Sigma_M, alpha=0, is_modelX=False):
    """
    This function will sample a M given X,
    Now only give a normal distribution case.
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix. It should be a random designed matrix
    D:  np.ndarray
            ``(m, p)``-shaped transformal matrix. 
    Sigma: np.ndarray
            ``(p, p)`` covariance matrix of X
    Sigma_M: np.ndarray
            ``(m, m)`` covariance matrix of M
    alpha: the maximum constant 
    is_modelX: whether use the modelX
    """
    if is_modelX:
        alpha = optimize_alpha(Sigma, Sigma_M, D)
        print("Treat the algo in ModelX. alpha:", alpha)
        # xx
        return X, alpha
    n, p = X.shape
    m = Sigma_M.shape[0]
    if alpha == 0:
        alpha = optimize_alpha(Sigma, Sigma_M, D)
    
    DTM = alpha * D.T @ Sigma_M  # (p, m)
    # aM = alpha * Sigma_M  #(m, m)
    
    Sigma_inv = np.linalg.inv((Sigma + Sigma.T) / 2)

    mean = X @ Sigma_inv @ DTM
    Cov = alpha * Sigma_M - DTM.T @ Sigma_inv @ DTM
    
    M = np.stack([np.random.multivariate_normal(mean[i], Cov, size=1)[0] for i in range(len(mean))])
    return M, alpha
    
    
def generate_M_copy(M, Sigma_M, alpha):
    n, m = M.shape
    eigvals = np.linalg.eigvalsh(Sigma_M)
    s = 2*alpha*np.min(eigvals)
    S = np.diag([s for _ in range(m)])
    
    aSM = alpha * Sigma_M
    aSM_inv = np.linalg.inv(aSM)
    # aSMS = alpha * Sigma_M - S
    
    mean = M - M @ aSM_inv @ S
    Cov = 2 * S - S @ aSM_inv @ S
    
    M_tilde = np.stack([np.random.multivariate_normal(mean[i], Cov, size=1)[0] for i in range(len(mean))])
    return M_tilde

from .pairwise import generate_pariwise_M, generate_pairwise_M_copy  
def construct_spllitmodelx_matrix(X, D, Sigma, Sigma_M, is_modelX=False, con_type="normal", aug_n=0):
    """
    This function gives the modelx knockoff copy matrix
    [] and response vector tilde_y. It will also create a
    knockoff copy for A_gamma if required.
    Parameters
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix. It should be a random designed matrix
    D : np.ndarrays
        ``(m, p)``-shaped transformation matrix
    Sigma: np.ndarray
            ``(p, p)`` covariance matrix of X
    Sigma_M: np.ndarray
            ``(m, m)`` covariance matrix of M
    nu: float
        potential nu for cross validation. 
        If `None`, will set it to np.power(10, np.arange(-1, -8, -0.1))
    """
    # assert con_type in ["normal", "pairwise"]
    if con_type == "normal":
        M, alpha = generate_M(X, D, Sigma, Sigma_M, is_modelX=is_modelX)
        M_tilde = generate_M_copy(M, Sigma_M, alpha)
    elif "pairwise" in con_type:
        ptype = "randM"
        if "fixM" in con_type:
            ptype = "fixM"
        elif "fixaugM" in con_type:
            ptype = "fixaugM"
        elif "seqM" in con_type:
            ptype = "seqM"
        
        M = generate_pariwise_M(X, D, ptype, aug_n)
        M_tilde = generate_pairwise_M_copy(M, X, D, ptype, aug_n)
        alpha = 0
    return alpha, M, M_tilde

