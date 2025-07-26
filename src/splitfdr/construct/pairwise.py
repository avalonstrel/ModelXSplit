import numpy as np

def generate_pariwise_M(X, D, ptype="randM", aug_n=0):
    """
    This function will sample a M given X,
    Currently it only support the independent generation method for M.
    Parameters:
    ----------
    X : np.ndarray
            ``(n, p)``-shaped design matrix. It should be a random designed matrix
    D:  np.ndarray
            ``(m, p)``-shaped transformal matrix. 
    ptype:pairwise type, randomnes on M: "randM"
                        fixed M: "fixM"
    """
    n, p = X.shape
    m, p = D.shape
    M = np.zeros((n, m))
    
    # For each row, randomly choose one column to set to 1
    if ptype == "randM":
        M[np.arange(n), np.random.randint(0, m, size=n)] = 1
    elif ptype == "fixM":
        D_dict = {}
        for i_ in range(m):
            d_ = D[i_]
            inds = np.argwhere(d_ != 0).reshape(-1).tolist()        
            D_dict[(inds[0], inds[1])] = i_
        
        for i in range(n):
            inds = np.argwhere(X[i] != 0).reshape(-1).tolist()
            M[i, D_dict[(inds[0], inds[1])]] = 1       
    elif ptype == "fixaugM":
        D_dict = {}
        for i_ in range(m):
            d_ = D[i_]
            inds = np.argwhere(d_ != 0).reshape(-1).tolist()        
            D_dict[(inds[0], inds[1])] = i_

        for i in range(n - aug_n):
            inds = np.argwhere(X[i] != 0).reshape(-1).tolist()
            M[i, D_dict[(inds[0], inds[1])]] = 1
    elif ptype == "seqM":
        M = np.random.binomial(1, 0.5, (n, m))
    else:
        return None
    return M

def generate_pairwise_M_copy(M, X, D, ptype="randM", aug_n=0):
    n, p = X.shape
    m, p = D.shape
    if ptype == "randM":
        return generate_pairwise_M_copy_randM(M, X, D)
    elif ptype == "fixM":
        M_tilde = np.zeros((n, m))
        M_tilde[np.arange(n), np.random.randint(0, m, size=n)] = 1
        return M_tilde
    elif ptype == "fixaugM":
        M_tilde = np.zeros((n, m))
        M_tilde[np.arange(n-aug_n, n), np.random.randint(0, m, size=aug_n)] = 1
        
        return M_tilde
    elif ptype == "seqM":
        return generate_pairwise_M_copy_seqM(M, X, D)
    else:
        return None
    
def generate_pairwise_M_copy_seqM(M, X, D):
    n, p = X.shape
    m, p = D.shape

    M_tilde = np.zeros((n, m))
    for i_ in range(n):
        M_tilde[i_] = M_copy_single_dp(M[i_], X[i_], D)

    return M_tilde




def M_copy_single_dp(M, X, D):
    m = len(M)
    
    def fast_init_t_func(k, A_ik):
        # Convert A_ik to numpy array
        A_ik_array = np.array([int(bit) for bit in A_ik])
        A_ko = A_ik_array.copy()
        
        # A_ko[k] = 1  # Set kth position to 1
        # if A_ko[k] = 1
        R = X - D[k] if A_ko[k] == 1 else X
        # R = X - A_ko @ D
        # R = X - D[k]
        inds = np.argwhere(R != 0).reshape(-1).tolist()

        if len(inds) > 2:
            p = 1
        elif len(inds) == 2:
            iks = np.argwhere(D[k] != 0).reshape(-1).tolist()
            
            ik1, ik2 = iks
            if R[ik1] == 0 and R[ik2] == 1:
                js = np.argwhere(R == -1).reshape(-1).tolist()
                j = js[0]
                if j < ik2:
                    p = 1
                else:
                    p = 0.5
            elif R[ik1] == -1 and R[ik2] == 0:
                js = np.argwhere(R == 1).reshape(-1).tolist()
                j = js[0]
                if j < ik1:
                    p = 0.5
                else:
                    p = 1
            elif (R[ik1] == R[ik2] == 0 or
                  R[ik1] == 1 and R[ik2] == -1 or 
                  R[ik1] == 1 and R[ik2] == 0 or 
                  R[ik1] == 0 and R[ik2] == -1):
                p = 0
            else:
                # p = 1
                raise NotImplemented
        else:
            p = 1
        return p
    
    # DP table: key is (k, A_tk_state), value is probability
    DP = {}
    M_str = ''.join(str(int(bit)) for bit in M)
    
    M_tilde = np.zeros(m, dtype=int)
    for k in range(m):
        # For each k, compute the probability p
        # A_ik is M_str
        # A_tk is M_tilde up to k as a string
        A_tk = ''.join(str(bit) for bit in M_tilde[:k])
        
        # Compute p using DP
        p = dp_probability(k, M_str, A_tk, DP, fast_init_t_func)
        
        # Sample M_tilde[k]
        M_tilde[k] = np.random.binomial(1, p)
    return M_tilde


def dp_probability(k, A_ik, A_tk, DP, fast_init_t_func):
    # Corresponding to CY description
    # Key for DP table
    state_key = (k, A_tk)
    if state_key in DP:
        return DP[state_key]
    
    if k == 0 or len(A_tk) == 0:
        p = fast_init_t_func(k, A_ik)
        DP[state_key] = p
        return p
    
    A_tkm = A_tk[-1]
    # Case when A_tk[k] = 0
    q = len(A_tk)
    A_ik_0 = A_ik[:q-1] + '0' + A_ik[q:]
    A_tk_prev = A_tk[:-1]
    p0 = dp_probability(k-1, A_ik_0, A_tk_prev, DP, fast_init_t_func)
    if A_tkm == '0':
        p0 = 1 - p0

    # Case when A_tk[k] = 1
    A_ik_1 = A_ik[:q-1] + '1' + A_ik[q:]
    p1 = dp_probability(k-1, A_ik_1, A_tk_prev, DP, fast_init_t_func)
    if A_tkm == '0':
        p1 = 1 - p1

    # 
    A_ik_0 = A_ik[:k-1] + '0' + A_ik[k:]
    t0 = 1 - dp_probability(k, A_ik_0, A_tk_prev, DP, fast_init_t_func)   
    
    # Case when A_tk[k] = 1
    A_ik_1 = A_ik[:k-1] + '1' + A_ik[k:]
    t1 = dp_probability(k, A_ik_1, A_tk_prev, DP, fast_init_t_func)
    
    numerator = p1 * t1
    denominator = p0 * t0 + p1 * t1
    if denominator == 0:
        p = 0.0
    else:
        p = numerator / denominator
    
    DP[state_key] = p
    
    return p


def generate_pairwise_M_copy_randM(M, X, D):
    n, p = X.shape
    m, p = D.shape
    
    D_dict = {}
    for i_ in range(m):
        d_ = D[i_]
        inds = np.argwhere(d_ != 0).reshape(-1).tolist()        
        D_dict[(inds[0], inds[1])] = i_
    
    M_tilde = np.zeros((n, m))
    
    X_tilde = X - M @ D  # n x p
    # 1 must first
    for i_ in range(n):
        x_tilde = X_tilde[i_]
        inds = np.argwhere(x_tilde != 0).reshape(-1).tolist()
        if len(inds) == 2:
            # Means only (2, -2), (1, -1)
            if x_tilde[inds[0]] == 2:
                m_i = D_dict[(inds[0], inds[1])]
            elif x_tilde[inds[0]] == 1:  
                if inds[0] < inds[1] - 1: 
                    rand_idx = np.random.choice([r_ for r_ in range(inds[0]+1, inds[1])])
                    if np.random.choice([True, False]):
                        m_i = D_dict[(inds[0], rand_idx)]    
                    else:
                        m_i = D_dict[(rand_idx, inds[1])]
                else:
                    raise NotImplemented
            else:
                
                raise NotImplemented
        elif len(inds) == 3:
            #  (1, 1, -2), (2, -1, -1)
            if x_tilde[inds[0]] == 1:
                rand_idx = np.random.choice(inds[:2])
                m_i = D_dict[(rand_idx, inds[-1])] 
            elif x_tilde[inds[0]] == 2:
                rand_idx = np.random.choice(inds[1:])
                m_i = D_dict[(inds[0], rand_idx)]
            else:
                raise NotImplemented
            
        elif len(inds) == 4:
            # (1,1,-1,-1), (1,-1,1,-1)
            if x_tilde[inds[1]] == 1:
                first_idx, second_idx = np.random.choice(inds[:2]), np.random.choice(inds[2:])
                m_i = D_dict[(first_idx, second_idx)]
            elif x_tilde[inds[1]] == -1:
                first_idx = np.random.choice([inds[0], inds[2]])
                if first_idx == inds[0]:
                    m_i = D_dict[(first_idx, inds[1])]
                elif first_idx == inds[2]:
                    m_i = D_dict[(first_idx, inds[3])]
                else:
                    raise NotImplemented
            else:
                raise NotImplemented
        else:
            m_i = np.random.choice(m, 1)
            
        M_tilde[i_, m_i] = 1
    
    return M_tilde
    
