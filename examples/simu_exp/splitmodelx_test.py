import os
import json
import numpy as np
import pickle as pkl
from splitfdr.split_modelx import SplitModelXFilter
import argparse

parser = argparse.ArgumentParser(description="Description of your script")
parser.add_argument(
    "--D_types", type=str, default="0,1,2", help="which D matrix to be used."
)
parser.add_argument("--q", type=float, default=0.2, help="The q ratio used.")
parser.add_argument(
    "--test_type", type=str, default="noX", help="The simulated Data type."
)
parser.add_argument(
    "--data_type", type=str, default="normal", help="The simulated Data type."
)
parser.add_argument(
    "--con_type", type=str, default="normal", help="How to construct M and M copy."
)
parser.add_argument(
    "--model_type", type=str, default="logistic", help="which model type is used."
)
parser.add_argument(
    "--Z_types", type=str, default="db", help="which model type is used."
)
parser.add_argument("--lambdas", type=str, default=None, help="lambdas used to cross validation.")
parser.add_argument("--nus", type=str, default=None, help="lambdas used to cross validation.")
parser.add_argument("--n", type=int, default=500, help="the sample size.")
parser.add_argument("--p", type=int, default=100, help="the number of features.")
parser.add_argument("--c", type=float, default=0.3, help="the feature correlation.")
parser.add_argument("--k", type=int, default=20, help="the sparsity level.")
parser.add_argument("--A", type=float, default=1, help="the signal noise ratio.")

args = parser.parse_args()

PROJECT_DIR = os.getcwd()
print(PROJECT_DIR)


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
    # generate X
    Sigma = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = c ** (np.abs(i - j))

    np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    # generate beta and gamma
    m = D.shape[0]
    beta_true = np.zeros(p)
    for i in range(k):
        beta_true[i] = A
        if (i + 1) % 3 == 1:
            beta_true[i] = 0
    print("beta_true", beta_true)
    gamma_true = D @ beta_true
    return X, Sigma, beta_true, gamma_true


def colinear_data_simulation(n, p, D, A, c, k, seed):
    # generate X with some specific features with colinearity
    Sigma = np.zeros((p, p))
    colinear_pairs = np.random.choice(k, 2 * (k // 5), replace=False).reshape(-1, 2)
    
    for i in range(p):
        for j in range(p):
            Sigma[i, j] = c ** (np.abs(i - j))
    for i, j  in colinear_pairs:
        Sigma[i, j] = 0.9 
        
    np.random.seed(seed)
    X = np.random.multivariate_normal(np.zeros(p), Sigma, n)

    # generate beta and gamma
    m = D.shape[0]
    beta_true = np.zeros(p)
    for i in range(k):
        beta_true[i] = A
        if (i + 1) % 3 == 1:
            beta_true[i] = 0
            
    for i, j  in colinear_pairs:
        beta_true[i] = A
        beta_true[j] = A
    
    print("beta_true", beta_true, colinear_pairs)
    gamma_true = D @ beta_true
    return X, Sigma, beta_true, gamma_true


def unit_simulation_cv(
    exp_name,
    test_type,
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
    assert data_type in ["normal", "colinear"]
    if data_type == "normal":
        X, Sigma, beta_true, gamma_true = data_simulation(n, p, D, A, c, k, seed)
    elif data_type == "colinear":
        X, Sigma, beta_true, gamma_true = colinear_data_simulation(n, p, D, A, c, k, seed)

    # Estimate Sigma
    if "_est_" in exp_name:
        Sigma_hat = X.T @ X / n
    elif "_gt_" in exp_name:
        Sigma_hat = Sigma
    else:
        raise NotImplemented

    # Some Choice of Sigma M
    Sigma_M = np.eye(D.shape[0])
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
    
    result_dir = f"{PROJECT_DIR}/results/splitmodelx/data_{data_type}/con_{con_type}"
    os.makedirs(f"{result_dir}/{exp_name}", exist_ok=True)
    lambda_tag = f"lam{np.log10(lambdas.min()):.3f}-{np.log10(lambdas.max()):.3f}"
    nu_tag = f"nu{np.log10(nus.min()):.3f}-{np.log10(nus.max()):.3f}"

    for n_test in range(num_tests):
        print(f"Start test {n_test}.")
        np.random.seed(n_test)

        if test_type == "withX":
            X = np.random.multivariate_normal(np.zeros(p), Sigma, n)
            # Estimate Sigma
            if "_est_" in exp_name:
                Sigma_hat = X.T @ X / n
            elif "_gt_" in exp_name:
                Sigma_hat = Sigma
            else:
                raise NotImplemented
            
        varepsilon = np.random.randn(n) * np.sqrt(sigma)
        y = X @ beta_true + varepsilon
        if model_type == "logistic":
            y = logistic_model(X @ beta_true, "random")
            normalize = False
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
        )

        for key in fdr_split:
            fdr_split[key].append(fdr_func(whole_results[f"S_{key}"], gamma_true))
            power_split[key].append(power_func(whole_results[f"S_{key}"], gamma_true))

        pkl.dump(
            {
                "Sigma_hat": Sigma_hat,
                "Sigma_M": Sigma_M,
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
                "num_tests": num_tests,
                "curr_test": n_test,
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
    
    split_ratio = 0.2
    normalize = False
    num_tests = 200
    sigma = 1

    test_type = args.test_type
    data_type = args.data_type
    con_type = args.con_type
    model_type = args.model_type
    
    Z_types = tuple(args.Z_types.split("_")) # how to compute the Z, used for the solver, like different optimization problem
    W_types = (
        "bc",
        "lcd"
    )  # how to compute the W, used for the solver, how to derive W from Z
    T_types = (
        "k",
        "k+",
    )  # how to compute the Z, used for the solver, hot the decide the T
    lambdas = np.power(10, np.arange(*[float(t) for t in args.lambdas.split(':')]))
    nus = np.power(10, np.arange(*[float(t) for t in args.nus.split(':')]))
    type_key = f"{model_type}_" + "_".join(Z_types + W_types)
    
    seed = 109

    # generate D1, D2, D3
    D_G = np.zeros((p - 1, p))
    for i in range(p - 1):
        D_G[i, i] = 1
        D_G[i, i + 1] = -1
    D_1 = np.eye(p)
    D_2 = D_G
    D_3 = np.concatenate([np.eye(p), D_G], axis=0)

    Ds = {"0": D_1, "1": D_2, "2": D_3}
    for D_i in (args.D_types).split(","):
        D = Ds[D_i]
        for cov_type in ["gt", "est"]:
            exp_name = f"{type_key}_{cov_type}_N{num_tests}_{param_tag}_{test_type}_D{D_i}"
            print(f"Start test {exp_name} D_{D_i}.")
            unit_simulation_cv(
                exp_name,
                test_type,
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
