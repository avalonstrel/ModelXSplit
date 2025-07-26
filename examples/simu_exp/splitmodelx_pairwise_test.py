import os
import json
import numpy as np
import time
import pickle as pkl
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Literal
from pathlib import Path
import logging
from splitfdr.split_modelx import SplitModelXFilter
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """Configuration class for simulation parameters."""
    D_types: str = "0"
    q: float = 0.2
    data_type: str = "normal"
    con_type: str = "pairwise"
    model_type: str = "linear"
    Z_types: str = "db"
    n: int = 500
    p: int = 100
    c: float = 0.3
    k: float = 0.1
    A: float = 1.0
    lambdas: Optional[str] = None
    nus: Optional[str] = None
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'SimulationConfig':
        return cls(**{k: v for k, v in vars(args).items() if k in SimulationConfig.__annotations__})

def setup_argparse() -> argparse.ArgumentParser:
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Split ModelX Pairwise Test Simulation")
    parser.add_argument("--D_types", type=str, default="0", help="Which D matrix to be used")
    parser.add_argument("--q", type=float, default=0.2, help="The q ratio used")
    parser.add_argument("--data_type", type=str, default="normal", help="The simulated data type")
    parser.add_argument("--con_type", type=str, default="pairwise", help="How to construct M and M copy")
    parser.add_argument("--model_type", type=str, default="linear", help="Which model type is used")
    parser.add_argument("--Z_types", type=str, default="db", help="Which Z type is used")
    parser.add_argument("--n", type=int, default=500, help="The sample size")
    parser.add_argument("--p", type=int, default=100, help="The number of features")
    parser.add_argument("--c", type=float, default=0.3, help="The feature correlation")
    parser.add_argument("--k", type=float, default=0.1, help="The sparsity level")
    parser.add_argument("--A", type=float, default=1, help="The signal noise ratio")
    parser.add_argument("--lambdas", type=str, default=None, help="Lambda range in format 'start:end:step'")
    parser.add_argument("--nus", type=str, default=None, help="Nu range in format 'start:end:step'")
    return parser

parser = setup_argparse()
args = parser.parse_args()
PROJECT_DIR = os.getcwd()


def fdr_func(result, gamma_true):
    """
    Params:
    result: support set in [Ture, False...]
    gamma_ture: the
    """
    return np.sum(gamma_true[result] == 0) / max(len(result), 1)


def power_func(result, gamma_true):
    return np.sum(gamma_true[result] != 0) / np.sum(gamma_true != 0)


def fdr_func_cv(results, gamma_true):
    """
    Params:
    results: support set in [Ture, False...]
    gamma_ture: the
    """
    fdrs = {
        lambda_val: fdr_func(results[lambda_val], gamma_true) for lambda_val in results
    }
    sorted_fdrs = sorted(fdrs.items(), lambda x: x[1])
    return sorted_fdrs[0]


def power_func_cv(results, gamma_true):
    powers = {
        lambda_val: power_func(results[lambda_val], gamma_true)
        for lambda_val in results
    }
    sorted_powers = sorted(powers.items(), lambda x: x[1])
    return sorted_powers[0]


def fdr_power_func_cv(results, gamma_true):
    fdr_powers = {
        lambda_val: [
            fdr_func(results[lambda_val], gamma_true),
            power_func(results[lambda_val], gamma_true),
        ]
        for lambda_val in results
    }
    sorted_fdr_powers = sorted(fdr_powers.items(), key=lambda x: x[1][1])
    return sorted_fdr_powers[0]


def logistic_model(X, gen_type="random"):
    def sigmoid(x):
        """Numerically stable sigmoid function"""
        return 1 / (1 + np.exp(-x))

    y_ = sigmoid(X)
    y = np.zeros_like(y_)
    assert gen_type in ["deterministic", "random"]
    if gen_type == "deterministic":
        y[y_ > 0.5] = 1
        y[y_ <= 0.5] = 0
    elif gen_type == "random":
        for i in range(len(y)):
            y[i] = np.random.binomial(1, y_[i])
    return y

def data_simulation(n, p, D, A, c, k, seed):
    # Pairwise
    # n samples/edges
    # p as the number of object
    #  gamma truea as p scores
    np.random.seed(seed)
    # beta_true = np.random.randint(0, int(A), p) / A
    beta_true = np.random.rand(p) * A

    beta_true[int(p * k) :] = 0

    X = np.zeros((n, p))

    for i in range(n):
        rand_idx1, rand_idx2 = np.random.choice(p, 2, replace=False)
        if rand_idx1 > rand_idx2:
            rand_idx1, rand_idx2 = rand_idx2, rand_idx1
        X[i, rand_idx1] = 1
        X[i, rand_idx2] = -1

    gamma_true = D @ beta_true
    print("True", beta_true, gamma_true)
    Sigma = np.zeros((p, p))
    return X, Sigma, beta_true, gamma_true


def data_simulation_aug(n, p, D, A, c, k, seed, data_type):
    # Pairwise
    # n samples/edges
    # p as the number of object
    #  gamma truea as p scores
    np.random.seed(seed)
    # beta_true = np.random.randint(0, int(A), p) / A
    beta_true = np.random.rand(p) * A

    beta_true[int(p * k) :] = 0

    X = np.zeros((n, p))

    for i in range(n):
        rand_idx1, rand_idx2 = np.random.choice(p, 2, replace=False)
        if rand_idx1 > rand_idx2:
            rand_idx1, rand_idx2 = rand_idx2, rand_idx1
        X[i, rand_idx1] = 1
        X[i, rand_idx2] = -1

    gamma_true = D @ beta_true
    print("True", beta_true, gamma_true)
    Sigma = np.zeros((p, p))
    
    # augment n 
    aug_n = n
    if "augment" in data_type:
        if "rand" in data_type:
            aug_n = np.random.negative_binomial(n, 0.5)
        X = np.concat([X, np.zeros((aug_n, p))], axis=0)
    elif "resample" in data_type:
        X, aug_n = data_resample(X, n * 2)
    else:
        raise NotImplemented
    return X, Sigma, beta_true, gamma_true, aug_n

def data_resample(X, res_n):
    n, p = X.shape
    res_idxs = np.random.randint(0, 2, res_n)
   
    res_num = np.sum(res_idxs)
    # print(res_idxs, res_num)
    # sss
    aug_n = len(res_idxs) - res_num
    sel_idxs = np.random.choice(len(X), res_num, replace=True)
    X = np.concat([X[sel_idxs], np.zeros((aug_n, p))], axis=0)
    return X, aug_n


def unit_simulation_cv(
    exp_name,
    data_type,
    con_type,
    model_type,
    Z_types,
    W_types,
    T_types,
    n,
    p,
    D,
    A,
    c,
    k,
    sigma,
    seed,
    num_tests,
    split_ratio,
    lambdas,
    nus,
    q,
    normalize,
    is_modelX=False,
):
    # data simulation'
    # assert data_type in ["normal", ]
    aug_n = 0
    if ("augment" in data_type or 
        "resample" in data_type):
        X, Sigma, beta_true, gamma_true, aug_n = data_simulation_aug(n, p, D, A, c, k, seed, data_type)
    else:
        X, Sigma, beta_true, gamma_true = data_simulation(n, p, D, A, c, k, seed)
    print("Gamma True Ratio:", np.sum(gamma_true != 0) / len(gamma_true))

    # Estimate Sigma
    if "_est_" in exp_name:
        Sigma_hat = X.T @ X / n
    elif "_gt_" in exp_name:
        Sigma_hat = Sigma
    else:
        raise NotImplemented

    # Some Choice of Sigma M
    Sigma_M = np.eye(D.shape[0])
    # Sigma_M = Sigma
    print("Whether treat the model as original modelX: Is ModelX:", is_modelX)
    # Sigma_M = np.linalg.pinv(D).T @ Sigma @ np.linalg.pinv(D) + 2 * np.eye(D.shape[0])
    # Sigma_M = D @ Sigma @ D.T
    modelx_filter = SplitModelXFilter()
    # test with multiple random noise
    type_keys = [
        f"{Z_type}_{W_type}_{T_type}"
        for Z_type in Z_types
        for W_type in W_types
        for T_type in T_types
    ]
    fdr_split, power_split = {key: [] for key in type_keys}, {
        key: [] for key in type_keys
    }

    # Settings for save
    print(exp_name)
    result_dir = f"{PROJECT_DIR}/results/splitmodelx/data_{data_type}/con_{con_type}"
    os.makedirs(f"{result_dir}/{exp_name}", exist_ok=True)
    lambda_tag = f"lam{np.log10(lambdas.min()):.3f}-{np.log10(lambdas.max()):.3f}"
    nu_tag = f"nu{np.log10(nus.min()):.3f}-{np.log10(nus.max()):.3f}"
    durations = []
    for n_test in range(num_tests):
        print(f"Start test {n_test}.")
        np.random.seed(n_test)
        
        varepsilon = np.random.randn(len(X)) * np.sqrt(sigma)
        y = X @ beta_true + varepsilon
        # if normal: loss and model according to model_type
        # others: y model according to data_type, loss according to model_type
    
        if data_type in ["normal", "augment", "augment_rand", "resample"] :
            if model_type == "logistic":
                y = logistic_model(X @ beta_true, "random")
                normalize = False
        elif data_type == "uniform":
            pass
        elif data_type == "BT":
            y = logistic_model(X @ beta_true, "random")
            normalize = False
        else:
            raise NotImplemented 
        start_time = time.time()   
        whole_results = modelx_filter.forward(
            X,
            y,
            D,
            Sigma_hat,
            Sigma_M,
            split_ratio,
            con_type,
            model_type,
            lambdas,
            nus,
            q=q,
            Z_types=Z_types,
            W_types=W_types,
            T_types=T_types,
            normalize=normalize,
            is_modelX=is_modelX,
            aug_n=aug_n,
        )
        duration = time.time() - start_time
        durations.append(duration)
        for key in fdr_split:

            fdr_split[key].append(fdr_func(whole_results[f"S_{key}"], gamma_true))
            power_split[key].append(power_func(whole_results[f"S_{key}"], gamma_true))
            # print(whole_results[f"S_{key}"], gamma_true)

        pkl.dump(
            {
                "Sigma_hat": Sigma_hat,
                "Sigma_M": Sigma_M,
                "whole_results": whole_results,
                "D": D,
                "lambdas": lambdas,
                "nus": nus,
                "n": n,
                "p": p,
                "A": A,
                "c": c,
                "k": k,
                "fdr": fdr_split,
                "power": power_split,
                "num_tests": num_tests,
                "curr_test": n_test,
                "seed": seed,
            },
            open(f"{result_dir}/{exp_name}/{lambda_tag}_{nu_tag}.pkl", "wb"),
        )
        json.dump(
            {
                "lambdas": lambdas.tolist(),
                "nus": nus.tolist(),
                "cv_lambdas": [
                    whole_results[f"lambda_{Z_type}_{W_type}"]
                    for Z_type in Z_types
                    for W_type in W_types
                ],
                "cv_nus": [
                    whole_results[f"nu_{Z_type}_{W_type}"]
                    for Z_type in Z_types
                    for W_type in W_types
                ],
                "n": n,
                "p": p,
                "A": A,
                "c": c,
                "k": k,
                "fdr": fdr_split,
                "power": power_split,
                "mean_fdr": {key:sum(fdr_split[key]) / len(fdr_split[key]) for key in fdr_split},
                "mean_power": {key:sum(power_split[key]) / len(power_split[key]) for key in power_split},
                "num_tests": num_tests,
                "curr_test": n_test,
                "durations": durations,
                "mean_duration":(sum(durations) / len(durations)),
                "seed": seed,
            },
            open(f"{result_dir}/{exp_name}/{lambda_tag}_{nu_tag}.json", "w"),
        )

    mean_fdr_split, mean_power_split = {}, {}
    std_fdr_split, std_power_split = {}, {}
    for key in fdr_split:
        mean_fdr_split[key] = np.array(fdr_split[key]).mean(axis=0)
        mean_power_split[key] = np.array(power_split[key]).mean(axis=0)

        std_fdr_split[key] = np.array(fdr_split[key]).std(axis=0)
        std_power_split[key] = np.array(power_split[key]).std(axis=0)
    print("Finish Test.")
    print(
        "FDR:",
        mean_fdr_split,
        std_fdr_split,
        "Power:",
        mean_power_split,
        std_power_split,
    )
    with open(f"{result_dir}/{exp_name}/{lambda_tag}_{nu_tag}.log", "w") as wf:
        wf.write(
            f"FDR Mean:{mean_fdr_split}\nFDR STD:{std_fdr_split}\nPower Mean:{mean_power_split}\nPower STD:{std_power_split}"
        )


def simulation():
    k = args.k  # the sparsity level
    A = args.A  # the signal noise ratio
    n = args.n  # the sample size
    p = args.p  # the number of features
    c = args.c  # the feature correlation
    param_tag = f"n{n}p{p}c{c}A{A}k{k}"

    q = args.q
    split_ratio = 0.4
    normalize = False
    num_tests = 200
    step_size = 0.5

    data_type = args.data_type
    con_type = args.con_type
    model_type = args.model_type
    # W_types = ('s', 'st')
    Z_types = tuple(args.Z_types.split("_")) # how to compute the Z, used for the solver, like different optimization problem
    W_types = (
        "bc",
        "lcd",
    )  # how to compute the W, used for the solver, how to derive W from Z
    T_types = (
        "k",
        "k+",
    )  # how to compute the Z, used for the solver, hot the decide the T

    sigma = 1
    seed = 109

    # generate D1, D2, D3
    D_G = np.zeros((p * (p - 1) // 2, p))
    count = 0
    for i_ in range(p):
        for j_ in range(i_+1, p):
            D_G[count][i_] = 1
            D_G[count][j_] = -1
            count += 1
            
    Ds = {"0": D_G}
    
    type_key = f"{model_type}_" + "_".join(Z_types + W_types)
    whole_lambdas, whole_nus = [], []
    if args.lambdas is not None and args.nus is not None:
        lambdas = np.power(10, np.arange(*[float(t) for t in args.lambdas.split(':')]))
        nus = np.power(10, np.arange(*[float(t) for t in args.nus.split(':')]))
        whole_lambdas.append(lambdas)
        whole_nus.append(nus)
    else:
        raise NotImplemented
    
    for lambdas, nus in zip(whole_lambdas, whole_nus):
        for D_i in (args.D_types).split(","):
            D = Ds[D_i]
            exp_name = f"{type_key}_est_N{num_tests}_{param_tag}_mosek_D{D_i}"
            print(f"Start test {exp_name} D_{D_i}.")
            unit_simulation_cv(
                exp_name,
                data_type,
                con_type,
                model_type,
                Z_types,
                W_types,
                T_types,
                n,
                p,
                D,
                A,
                c,
                k,
                sigma,
                seed,
                num_tests,
                split_ratio,
                lambdas,
                nus,
                q,
                normalize,
            )
            print(f"Finish test {exp_name} D_{D_i}.")


if __name__ == "__main__":
    simulation()
