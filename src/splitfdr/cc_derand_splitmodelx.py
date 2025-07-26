from splitfdr.utils.misc import normalize_func, random_split, evalue_func, select_evalue
from splitfdr.construct import construct_spllitmodelx_matrix, generate_M_copy
from splitfdr.W_statistics.modelx_statistics import compute_W_stat
import copy
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl
from multiprocessing import Pool


def each_evalue_func(data):
    # paarms unpackage
    X, y, D, M = (
        data["X"],
        data["y"],
        data["D"],
        data["M"]
    )
    alpha = data["alpha"]
    Sigma, Sigma_M = (
        data["Sigma"],
        data["Sigma_M"],
    )
    lambdas, nus = (
        data["lambdas"],
        data["nus"],
    )
    model_type, Z_types, W_types, T_types = (
        data["model_type"],
        data["Z_types"],
        data["W_types"],
        data["T_types"],
    )
    is_modelX = data["is_modelX"]
    q = data["q"]

    # Generate M and M_tilde
    M_tilde = generate_M_copy(M, Sigma_M, alpha)
    # alpha, M, M_tilde = construct_spllitmodelx_matrix(
    #     X, D, Sigma, Sigma_M, is_modelX=is_modelX
    # )

    # Compute the W statistics
    Ws, cv_lambda_vals, cv_nu_vals = compute_W_stat(
        model_type, Z_types, W_types, X, y, D, M, M_tilde, lambdas, nus
    )

    # Compute the chosen S and results
    results = {}
    evals = {}
    for Z_type in Z_types:
        for W_type in W_types:
            for T_type in T_types:
                type_key = f"{Z_type}_{W_type}"
                # print(f'W:{W.shape}, S:{S.shape}')
                results[f"lambda_{type_key}"] = cv_lambda_vals[type_key]
                results[f"nu_{type_key}"] = cv_nu_vals[type_key]
                results[f"W_{type_key}"] = Ws[type_key]
                evals_nr = evalue_func(Ws[type_key], q, T_type)
                evals[type_key] = evals_nr
                results[f"E_{type_key}"] = evals_nr
    return evals, results


def get_fixM_suff_stat(idx, X, M, D, y):
    m = D.shape[0]
    neg_idx = np.array(range(m)) != idx
    return (X, M[:,neg_idx], y)

def get_randM_suff_stat(idx, X, D, y, Sigma, Sigma_M, is_modelX=False):
    m = D.shape[0]
    alpha, M, M_tilde = construct_spllitmodelx_matrix(
        X, D, Sigma, Sigma_M, is_modelX=is_modelX
    )
    neg_idx = np.array(range(m)) != idx
    return (X, M[:,neg_idx], y, alpha)

def resample_data(variables, idx):
    """
    We instead take the sufficient statistic as Z_{-j}, and sample
    Z_j | Z_{-j} by taking its knockoff Ztilde_j and reconstructing the data under H_j
    as (Ztilde_j, Z_{-j}).
    """
    # new_variables = copy.deepcopy(variables)
    X, M, y = variables["X"], variables["M"], variables["y"]
    D = variables["D"]
    alpha = variables["alpha"]
    suff_type = variables["suff_type"]
    Sigma, Sigma_M = variables["Sigma"], variables["Sigma_M"]

    if suff_type == "fixM":
        X_neg, M_neg, y_neg = get_fixM_suff_stat(idx, X, M, D, y)
    elif suff_type == "randM":
        X_neg, M_neg, y_neg, alpha = get_randM_suff_stat(idx, X, D, y, Sigma, Sigma_M, ["is_modelX"])
    
    n, m = M.shape
    neg_idx = np.array(range(m)) != idx
    
    inverse = np.linalg.inv(Sigma_M[neg_idx][:,neg_idx])
    cond_Sigma = Sigma_M[idx,idx] - Sigma_M[idx,neg_idx] @ inverse @ Sigma_M[neg_idx,idx] # 1x1
    cond_mu_vec = Sigma_M[idx,neg_idx] @ inverse @ (M_neg.T)
    
    M_new = np.zeros(shape=(n, m))
    M_new[:,idx] = np.random.normal(
        loc=cond_mu_vec, scale=np.sqrt(cond_Sigma), size=(n,)  # setting the slice of nx1 to new copies under the null
    )  
    M_new[:,neg_idx] = M_neg
    
    return X_neg, M_new, y_neg, alpha

def parse_results(eval_results, W_types, Z_types, T_types, q_e):
    results = {}
    num_derand = len(eval_results)
    type_tags = [f"{Z_type}_{W_type}" for Z_type in Z_types for W_type in W_types]
    for type_key in type_tags:
        results[f"lambda_{type_key}"] = [
            eval_results[i][1][f"lambda_{type_key}"] for i in range(num_derand)
        ]
        results[f"nu_{type_key}"] = [
            eval_results[i][1][f"nu_{type_key}"] for i in range(num_derand)
        ]
        evals_tmp = np.stack(
            [eval_results[i][0][type_key] for i in range(num_derand)]
        )
        print("Shape:", np.stack(evals_tmp).shape)
        eval_tmp = evals_tmp.mean(axis=0)
        Ws_tmp = np.stack(
            [eval_results[i][1][f"W_{type_key}"] for i in range(num_derand)]
        )
        results[f"E_{type_key}"] = eval_tmp
        results[f"W_{type_key}"] = Ws_tmp
        for T_type in T_types:
            results[f"S_{type_key}_{T_type}"] = select_evalue(eval_tmp, q_e)
    return results

def parse_cc_results(ori_results, cc_results, idx, q, W_types, Z_types, T_types):
    type_tags = [f"{Z_type}_{W_type}" for Z_type in Z_types for W_type in W_types]
    #cc_results [K, ]
    num_cc = len(cc_results)
    e_cc_vals = {}  #E_j, 1,...,k
    for idx_cc in range(num_cc):
        for type_tag in type_tags:
            for T_type in T_types:
                f_tag = f"{type_tag}_{T_type}"
                if not f_tag in e_cc_vals:
                    e_cc_vals[f_tag] = []
                tmp_cc_res = cc_results[idx_cc]
                ori_S = set(list(ori_results[f"S_{type_tag}_{T_type}"]) + [idx,])
                cc_S = set(list(tmp_cc_res[f"S_{type_tag}_{T_type}"]) + [idx,])
                ori_evals = ori_results[f"E_{type_tag}"]
                cc_evals = tmp_cc_res[f"E_{type_tag}"]
                m = len(ori_evals)
                ind = int((cc_evals[idx] / ori_evals[idx]) >= (len(ori_S) / len(cc_S)))
                e_cc_vals[f_tag].append(m / q * ind / len(cc_S) - cc_evals[idx])  # K number
    return e_cc_vals

def get_boost_eval(final_cc_results):
    boost_idxs = {}
    for key in final_cc_results:
        mean_cc_evals = final_cc_results[key].mean(axis=1)
        select_idxs = np.argwhere(mean_cc_evals < 0)
        boost_idxs[key] = select_idxs
    return boost_idxs

class CCDerandomizedSplitModelXFilter:
    """
    Performs ModelX-based inference, from start to finish.

    Parameters
    """

    def __init__(
        self,
    ):
        """
        Initialize the class.
        """

        # self.q = q
        # self.method = method
        pass

    def forward(
        self,
        X,
        y,
        D,
        Sigma,
        Sigma_M,
        num_cc,
        num_derand,
        split_ratio,
        model_type,
        lambdas,
        nus,
        q=0.10,
        q_e=0.2,
        Z_types=("LCD"),
        W_types=("st"),
        T_types=("knockoff"),
        suff_type="ramdM",
        normalize=True,
        norm_type="l2",
        is_modelX=False,
        use_avcs=False,
        tmp_save_path=None,
    ):
        """
        Runs the modelx filter; returns whether each feature was rejected.

        Parameters
        ----------
        X : np.ndarray
            ``(n, p)``-shaped design matrix
        y : np.ndarray
            ``(n,)``-shaped response vector
        D : np.ndarrays
            ``(m, p)``-shaped transformation matrix
        split_ratio: float
            The split ratio of the dataset
        fdr : float
            The desired level of false discovery rate control.

        num_derand: int
            The number of repeat times for computing e-value
        fstat_kwargs : dict
            Extra kwargs to pass to the feature statistic ``fit`` function,
            excluding the required arguments.
        knockoff_kwargs : dict
            Extra kwargs for instantiating the knockoff sampler argument if
            the ksampler argument is a string identifier. This can be
            the empty dict for some identifiers such as "gaussian" or "fx",
            but additional keyword arguments are required for complex samplers
            such as the "metro" identifier. Defaults to {}
        stats_type : str
            The type of W statistics used in the method. Now support ['LCD', 'LSM']
            LCD: lasso coefficent difference
            LSM: Lasso signal max (solution path)
        is_ModelX: bool
            Whether to treat the model as original ModelX which just ignore the D.

        """
        # normalize the data
        if normalize:
            X, y = normalize_func(X), normalize_func(y)
        n, m, p = X.shape[0], D.shape[0], X.shape[1]

        # Get a solved beta first which will used in the W statistics computation
        # TODO Fake split now
        X1, y1, X2, y2 = random_split(X, y, split_ratio=split_ratio)
        n1 = len(X1)

        # Generate M and M_tilde
        # Derandomized need repeat multiple times Concurrent
        from concurrent.futures import ProcessPoolExecutor
        
        alpha, M, M_tilde = construct_spllitmodelx_matrix(
            X, D, Sigma, Sigma_M, is_modelX=is_modelX
        )
        variables = {
            "X": X,
            "y": y,
            "D": D,
            "M":M, 
            "alpha":alpha,
            "Sigma": Sigma,
            "Sigma_M": Sigma_M,
            "lambdas": lambdas,
            "nus": nus,
            "model_type": model_type,
            "Z_types": Z_types,
            "W_types": W_types,
            "T_types": T_types,
            "is_modelX": is_modelX,
            "suff_type": suff_type,
            "q": q,
        }
        ori_eval_results = [each_evalue_func(variables) for nr_i in range(num_derand)]
        ori_results = parse_results(ori_eval_results, W_types, Z_types, T_types, q_e)
        neg_variables = copy.deepcopy(variables)
        # Conditional Calibration Need run m loop
        records = []
        final_cc_results = {f"{Z_}_{W_}_{T_}":np.zeros((m, num_cc)) for W_ in W_types for Z_ in Z_types for T_ in T_types}
        for idx in range(m):
            resamped_data = resample_data(variables, idx)
            for key, val in zip(["X", "M", "y", "alpha"], resamped_data):
                neg_variables[key] = val
            cc_results = []
            for idx_cc in range(num_cc):
                eval_results = [each_evalue_func(neg_variables) for nr_i in range(num_derand)]
                # Combine all results
                # Compute the chosen S and results
                results = parse_results(eval_results, W_types, Z_types, T_types, q_e)
                cc_results.append(results)
            records.append(cc_results)
            e_cc_vals = parse_cc_results(ori_results, cc_results, idx, q, W_types, Z_types, T_types)
            for key in final_cc_results:
                final_cc_results[key][idx] = np.array(e_cc_vals[key])  # [m x K]
            if tmp_save_path is not None:
                boost_idxs = get_boost_eval(final_cc_results)
                tmp_ori_path = tmp_save_path.replace(".pkl", "_ori.pkl")
                pkl.dump(records, open(tmp_save_path, "wb"))
                pkl.dump(ori_results, open(tmp_save_path.replace(".pkl", "_ori.pkl"), "wb"))
                pkl.dump(boost_idxs, open(tmp_save_path.replace(".pkl", "_idxs.pkl"), "wb"))

        
        # Direct mean E
        if use_avcs:
            # @TODO Method AVCS 
            pass
        else:
            # Direct mean E
            boost_idxs = get_boost_eval(final_cc_results)
            print(boost_idxs)

        # Have Get E_{j,k}
        return boost_idxs, ori_results, records
