import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from functools import partial
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from splitfdr.utils.misc import random_split_all
from splitfdr.utils.misc import hittingpoint, get_W
from splitfdr.W_statistics.sovlers.cvx_solver import (
    ModelXCVXCFSolver,
    ModelXCVXSolver,
    ModelXCVXSepSolver,
)
from splitfdr.W_statistics.sovlers.mosek_solver import ModelXMosekSolver
from splitfdr.W_statistics.sovlers.lgb_solver import LGBModel
from splitfdr.W_statistics.sovlers.ebm_solver import EBMModel
from splitfdr.W_statistics.sovlers.nn_solver import NeuralNetworkModel


def W_stat_func(Z, Z_tilde, W_type):
    tmp_func = globals()[f"{W_type}_stat"]
    return tmp_func(Z, Z_tilde)


def bc_stat(Z, Z_tilde):
    return np.maximum(Z, Z_tilde) * np.sign(Z - Z_tilde)


def lcd_stat(Z, Z_tilde):
    return np.abs(Z) - np.abs(Z_tilde)


def each_model_solve(data):
    beta, gamma, gamma_tilde = None, None, None
    try:
        # Create a new solver
        # model_type = data["model_type"]
        # solver_class = data["solver_class"]
        # X1, y1, D, M1, M1_tilde = data["X"], data["y"], data["D"], data["M"], data["M_tilde"]
        # modelx_solver = solver_class(model_type, X1, y1, D, M1, M1_tilde)

        modelx_solver = data["modelx_solver"]
        lambda_val, nu_inv_val = data["lambda_val"], data["nu_inv_val"]
        beta, gamma, gamma_tilde = modelx_solver.solve(lambda_val, nu_inv_val)
    except BaseException as err:
        print(err)
    return beta, gamma, gamma_tilde


def compute_W_stat(
    model_type, Z_types, W_types, X, y, D, M, M_tilde, lambda_vals, nu_vals
):
    Ws, cv_lambda_vals, cv_nu_vals = {}, {}, {}
    for Z_type in Z_types:
        # for solution path
        if "sp" not in Z_type:
            # get Z, Z_tilde first
            Z, Z_tilde, cv_lambda_val, cv_nu_val = compute_Z_value_stat(
                model_type, Z_type, X, y, D, M, M_tilde, lambda_vals, nu_vals
            )
        else:
            # @TODO not implemented for solution path now 
            # for solution path
            raise NotImplemented
        for W_type in W_types:
            type_key = f"{Z_type}_{W_type}"
            # get W
            W = W_stat_func(Z, Z_tilde, W_type)
            Ws[type_key] = W
            cv_lambda_vals[type_key], cv_nu_vals[type_key] = cv_lambda_val, cv_nu_val
    return Ws, cv_lambda_vals, cv_nu_vals


def compute_Z_value_stat(model_type, Z_type, X, y, D, M, M_tilde, lambda_vals, nu_vals):
    """
    Compute the LCD W statistics of modelX inference
    Parameters
    ----------

    """
    # Add CV
    num_cv = 1
    whole_records = []
    assert Z_type in [
        "",
        "db",
        "cf",
        "sep",
        "mos",
        "lgb",
        "ebm",
        "nn"
    ], f"Not supported solver {Z_type}."  # db means the original one, abbreviated by the double

    if Z_type in ["", "db"]:
        solver_class = ModelXCVXSolver
    elif Z_type == "cf":
        solver_class = ModelXCVXCFSolver
    elif Z_type == "sep":
        solver_class = ModelXCVXSepSolver
    elif Z_type == "mos":
        solver_class = ModelXMosekSolver
    elif Z_type == "lgb":
        solver_class = LGBModel
    elif Z_type == "ebm":
        solver_class = EBMModel
    elif Z_type == "nn":
        solver_class = NeuralNetworkModel
    
    for cv_i in range(num_cv):
        X1, y1, M1, M1_tilde, X2, y2, M2, M2_tilde = random_split_all(
            X, y, M, M_tilde, split_ratio=0.8
        )
        # find the statistics from solving the minimization problems
        betas, gammas, gammas_tilde = {}, {}, {}
        Ws = {}
        train_losses, eval_losses = {}, {}
        modelx_solver = solver_class(model_type, X1, y1, D, M1, M1_tilde)
        keys = [
            (lambda_val, 1 / nu_val if nu_val != np.inf else 0)
            for lambda_val in lambda_vals
            for nu_val in nu_vals
        ]
        results = [each_model_solve({
                            "modelx_solver": modelx_solver,
                            "lambda_val": key[0],
                            "nu_inv_val": key[1],
                        }) for key in keys ]
        # pool = Pool(processes=8)
        # results = pool.map(
        #     each_model_solve,
        #     [
        #         {
        #             "modelx_solver": modelx_solver,
        #             "lambda_val": key[0],
        #             "nu_inv_val": key[1],
        #         }
        #         for key in keys
        #     ],
        # )

        # with ThreadPoolExecutor(max_workers=8) as executor:
        #     results = list(
        #         executor.map(
        #             each_model_solve,
        #             [
        #                 {
        #                     "X": X1,
        #                     "y": y1,
        #                     "D": D,
        #                     "M": M1,
        #                     "M_tilde": M1_tilde,
        #                     "model_type": model_type,
        #                     "solver_class":solver_class,
        #                     # "modelx_solver": modelx_solver,
        #                     "lambda_val": key[0],
        #                     "nu_inv_val": key[1],
        #                 }
        #                 for key in keys
        #             ],
        #         )
        #     )

        for k_i, key in enumerate(keys):
            lambda_val, nu_inv_val = key
            beta, gamma, gamma_tilde = results[k_i]
            train_losses[key] = modelx_solver.cv_evaluate_loss(
                beta,
                gamma,
                gamma_tilde,
                X1,
                y1,
                D,
                M1,
                M1_tilde,
                lambda_val,
                nu_inv_val,
            )
            eval_losses[key] = modelx_solver.cv_evaluate_loss(
                beta,
                gamma,
                gamma_tilde,
                X2,
                y2,
                D,
                M2,
                M2_tilde,
                lambda_val,
                nu_inv_val,
            )
            betas[key], gammas[key], gammas_tilde[key] = beta, gamma, gamma_tilde
        whole_records.append((eval_losses, train_losses, betas, gammas, gammas_tilde))

    eval_losses = {
        key: np.mean([whole_records[cv_i][0][key] for cv_i in range(num_cv)])
        for key in keys
    }
    train_losses = {
        key: np.mean([whole_records[cv_i][1][key] for cv_i in range(num_cv)])
        for key in keys
    }
    
    sorted_losses = sorted(eval_losses.items(), key=lambda x: x[1])
    
    lambda_val, nu_inv_val = sorted_losses[0][0]

    try:
        modelx_solver = solver_class(model_type, X, y, D, M, M_tilde)
        beta, gamma, gamma_tilde = modelx_solver.solve(lambda_val, nu_inv_val)
    except:
        print("Best solved wrong: just set beta, gamma, gamma_tilde as 0")
    return gamma, gamma_tilde, lambda_val, 1 / nu_inv_val
    

def compute_W_solpath_statistics(
    model_type,
    W_types,
    X,
    y,
    D,
    M,
    M_tilde,
    lambda_vals,
    nu_val,
):
    """
    Compute the solution path of W statistics of modelX inference.
    Parameters
    ----------

    """
    nu_val = 1
    # find the statistics from solving the minimization problems
    gammas, gammas_tilde = [], []

    # Sequential Version
    modelx_solver = ModelXCVXSolver(model_type, X, y, D, M, M_tilde)
    for lambda_val in lambda_vals:
        gamma, gamma_tilde = modelx_solver.solve(lambda_val, nu_val)
        gammas.append(gamma), gammas_tilde.append(gamma_tilde)
    gammas, gammas_tilde = np.stack(gammas), np.stack(gammas_tilde)
    gammas[np.abs(gammas) < 1e-5] = 0
    gammas_tilde[np.abs(gammas_tilde) < 1e-5] = 0
    # Compute the W

    
    def get_r_Z(coef, lambdas):
        m = coef.shape[1]
        r, Z = np.zeros(m), np.zeros(m)
        for i in range(m):
            r_, Z_ = hittingpoint(coef[:, i], lambdas)
            
            r[i], Z[i] = r_, Z_
        return r, Z

    r, Z = get_r_Z(gammas, lambda_vals)
    tr, tZ = get_r_Z(gammas_tilde, lambda_vals)
    # W_types: ['bc','st'...]
    Ws = {W_type: get_W(r, tr, Z, tZ, W_type=W_type) for W_type in W_types}
    return Ws
