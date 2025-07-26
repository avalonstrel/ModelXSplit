import numpy as np
from splitfdr.split_modelx import SplitModelXFilter
from scipy.io import loadmat
import os
import argparse

parser = argparse.ArgumentParser(description="Description of your script")

parser.add_argument("--q", type=float, default=0.2, help="The q ratio used.")
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
args = parser.parse_args()


PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))

def fdr_func(result, gamma_true):
    """
    Params:
    result: support set in [Ture, False...]
    gamma_ture: the 
    """
    # print(gamma_true, result)
    return np.sum(gamma_true[result] == 0) / max(len(result), 1)

def power_func(result, gamma_true):
    return np.sum(gamma_true[result] != 0) / np.sum(gamma_true != 0)

def fdr_func_cv(results, gamma_true):
    """
    Params:
    results: support set in [Ture, False...]
    gamma_ture: the 
    """
    # print(result, gamma_true)
    fdrs = {lambda_val:fdr_func(results[lambda_val], gamma_true) for lambda_val in results}
    sorted_fdrs = sorted(fdrs.items(), lambda x:x[1])
    return sorted_fdrs[0]
    
def power_func_cv(results, gamma_true):
    powers = {lambda_val:power_func(results[lambda_val], gamma_true) for lambda_val in results}
    sorted_powers = sorted(powers.items(), lambda x:x[1])
    return sorted_powers[0]

def fdr_power_func_cv(results, gamma_true):
    fdr_powers = {lambda_val:[fdr_func(results[lambda_val], gamma_true), 
                                       power_func(results[lambda_val], gamma_true)] for lambda_val in results}
    sorted_fdr_powers = sorted(fdr_powers.items(), key=lambda x: x[1][1])
    return sorted_fdr_powers[0]

def load_data(data_type):
    feats = ['AD', 'MCI', 'NC']
    Xs, ys = [], []
    for feat in feats:
        X_path = f"{DATA_DIR}/data/AALfeat/AAL_{data_type}_{feat}_feature_TIV.mat"
        y_path = f"{DATA_DIR}/data/AALfeat/ADAS_{data_type}_{feat}.mat"
        X, y = loadmat(X_path), loadmat(y_path)
        Xs.append(np.array(X['feature_TIV']))
        ys.append(np.array(y['ADAS']))
    Xs = np.concatenate(Xs, axis=0)
    ys = np.concatenate(ys, axis=0)
    return Xs, ys

def data_preprocess():
    X1s, y1s = load_data('15')
    X2s, y2s = load_data('30')
    Xs = np.concatenate([X1s, X2s], axis=0)
    ys = np.concatenate([y1s, y2s], axis=0)
    Xs = Xs[:, :90]  # select Cerebrum brain regions
    
    # filter y < 0
    filter_indexs = ys.reshape(-1) > 0
    Xs = Xs[filter_indexs, :]
    ys = ys[filter_indexs]
    from splitfdr.utils.misc import normalize_col_func
    Xs = normalize_col_func(Xs, norm_type='l2')
    ys = normalize_col_func(ys, norm_type='l2')

    return Xs, ys.reshape(-1)

def get_edge_D():
    # limit to 90
    selected_indexs = np.arange(90)
    edges = []
    with open(f"{DATA_DIR}/data/AALfeat-python/edges.txt", "r") as rf:
        for line in rf:
            terms = [int(t) - 1 for t in line.strip().split()]
            if terms[0] in selected_indexs and terms[1] in selected_indexs:
                edges.append(terms)
    m = len(edges)
    p = len(selected_indexs)
    D = np.zeros((m, p))
    for i in range(m):
        D[i, edges[i][0]] = 1
        D[i, edges[i][1]] = -1
    return D

def unit_adni_cv(prob_type, W_types, T_types, split_ratio, lambdas, nus, q, normalize, seed):
    # data load & preprocess
    X, y = data_preprocess()
    n, p  = X.shape
    if prob_type == 'direct':
        D = np.eye(p)
    elif prob_type == 'connect':
        D = get_edge_D()
    # Estimate Sigma
    Sigma_hat = X.T @ X / n
    # Some Choice of Sigma M
    Sigma_M = np.eye(D.shape[0])
   
    con_type = args.con_type
    model_type = 'linear'
    modelx_filter = SplitModelXFilter()
    # test with multiple random noise
    Z_types = tuple(args.Z_types.split("_")) # how to compute the Z, used for the solver, like different optimization problem
    
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
        is_modelX=False,
    )
    print("Finish Test.")

    os.makedirs(f"{PROJECT_DIR}/results/splitmodelx/final/{prob_type}", exist_ok=True)
    save_path = f"{PROJECT_DIR}/results/splitmodelx/final/{prob_type}/results.txt"
    with open(save_path, "w") as wf:
        wf.write(str(whole_results))
    

def region_simulation():    
    split_ratio = 150 / 752
    normalize = False
    q = args.q
    prob_type = 'direct'
    W_types = (
        "bc",
    )  # how to compute the W, used for the solver, how to derive W from Z
    T_types = (
        "k",
    )  # how to compute the Z, used for the solver, hot the decide the T
    

    seed = 65
    np.random.seed(seed)
    lambdas = np.power(10, np.arange(-2.0, -3.1, -0.2)) # np.power(10, )
    nus = np.power(10, np.arange(-2.0, -1.0, 0.2)) # np.power(10, )
    unit_adni_cv(prob_type, W_types, T_types, split_ratio, lambdas, nus, q, normalize, seed)    


def connection_simulation():
    split_ratio = 600 / 752
    normalize = False
    q = args.q
    prob_type = 'connect'
    W_types = (
        "lcd",
    )  # how to compute the W, used for the solver, how to derive W from Z
    T_types = (
        "k",
    )  # how to compute the Z, used for the solver, hot the decide the T
    
    seed = 65
    np.random.seed(seed)
    lambdas = np.power(10, np.arange(-2.5, -1, 0.5)) 
    nus = np.power(10, np.arange(0.5, 1.5, 0.2)) 
    unit_adni_cv(prob_type, W_types, T_types, split_ratio, lambdas, nus, q, normalize, seed)  

if __name__ == '__main__':
    region_simulation()
    connection_simulation()










        