from splitfdr.utils.misc import normalize_func, random_split, evalue_func, select_evalue
from splitfdr.construct import construct_spllitmodelx_matrix
from splitfdr.W_statistics.modelx_statistics import (
    compute_W_stat,
)
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool


def each_evalue_func(data):
    # paarms unpackage
    X, y, D = (
        data["X"],
        data["y"],
        data["D"],
    )
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
    alpha, M, M_tilde = construct_spllitmodelx_matrix(
        X, D, Sigma, Sigma_M, is_modelX=is_modelX
    )
    
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


class DerandomizedSplitModelXFilter:
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
        normalize=True,
        norm_type="l2",
        is_modelX=False,
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
        # X1, y1 = X, y
        # X2, y2 = X, y
        n1 = len(X1)

        # Generate M and M_tilde
        # Derandomized need repeat multiple times Concurrent

        from concurrent.futures import ProcessPoolExecutor

        variables = {
            "X": X,
            "y": y,
            "D": D,
            "Sigma": Sigma,
            "Sigma_M": Sigma_M,
            "lambdas": lambdas,
            "nus": nus,
            "model_type": model_type,
            "Z_types": Z_types,
            "W_types": W_types,
            "T_types": T_types,
            "is_modelX": is_modelX,
            "q": q,
        }
        # Use ProcessPoolExecutor to parallelize the solving process
        # pool = Pool(processes=1)
        # eval_results = pool.map(
        #     each_evalue_func,
        #     [variables for nr_i in range(num_derand)],
        # )
        eval_results = [each_evalue_func(variables) for nr_i in range(num_derand)]
        # with ProcessPoolExecutor(max_workers=1) as executor:
        #     eval_results = list(
        #         executor.map(
        #             each_evalue_func,
        #             [variables for nr_i in range(num_derand)],
        #         )
        #     )
        # eval_results: [(evals, results)]
        # Combine all results
        # Compute the chosen S and results
        results = {}

        for Z_type in Z_types:
            for W_type in W_types:
                type_key = f"{Z_type}_{W_type}"

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
