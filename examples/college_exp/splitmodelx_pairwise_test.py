import os
import json
import pandas as pd
import numpy as np
import pickle as pkl
from splitfdr.split_modelx import SplitModelXFilter
import argparse

parser = argparse.ArgumentParser(description="Description of your script")

parser.add_argument("--q", type=float, default=0.2, help="The q ratio used.")
parser.add_argument(
    "--data_type", type=str, default="normal", help="The simulated Data type."
)
parser.add_argument(
    "--sel_type", type=str, default="max_degree", help="How to choose the node."
)
parser.add_argument(
    "--con_type", type=str, default="pairwise", help="How to construct M and M copy."
)
parser.add_argument(
    "--model_type", type=str, default="logistic", help="which model type is used."
)
parser.add_argument(
    "--D_type", type=str, default="full", help="which D type is used."
)
parser.add_argument("--n", type=int, default=500, help="the sample size.")
parser.add_argument("--res_scale", type=int, default=3, help="the sample size.")
parser.add_argument("--max_num", type=int, default=50, help="the size of selected ids.")
parser.add_argument("--lambdas", type=str, default="-7:-8.5:-0.5", help="a.")
parser.add_argument("--nus", type=str, default="0:1.5:0.5", help="b.")
# parser.add_argument("--lambdas", type=str, default="0:-2:-0.4", help="a.")
# parser.add_argument("--nus", type=str, default="2:4:0.4", help="b.")

args = parser.parse_args()


PROJECT_DIR = os.getcwd()
DATA_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))
    
COLLEGE_MAP = {
    "University of Tokyo":"The University of Tokyo",
    "London School of Economics and Political Science":"London School of Economics",
    "Massachusetts Institute of Technology":"Massachusetts Institute of Technology - MIT",
    "École Polytechnique Fédérale de Lausanne":"Swiss Federal Institute of Technology Lausanne - EPFL",
    "California Institute of Technology":"California Institute of Technology - Caltech",
    "University of Maryland, College Park": "University of Maryland at College Park",
    "University of California, Berkeley": "University of California - Berkeley",
    "ETH Zürich – Swiss Federal Institute of Technology Zürich": "Swiss Federal Institute of Technology Zurich - ETHZ", 
    "University of California, Los Angeles":"University of California - Los Angeles",
    "The University of Hong Kong":"University of Hong Kong",
    "University of California, San Diego":"University of California - San Diego",
    "University of California, Davis":"University of California - Davis",
    "University of Colorado Boulder":"University of Colorado at Boulder",
    "University of Michigan":"University of Michigan - Ann Arbor",
}

def process_name(name):
    terms = name.split(',')
    re_name = name
    if len(terms) > 1: re_name = ",".join(terms[:-1])
    if re_name in COLLEGE_MAP:
        return COLLEGE_MAP[re_name]
    return re_name

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

def get_rank(path):
    rank_dict = {}
    with open(path, 'r') as rf:
        for i, line in enumerate(rf):
            if i == 0: continue
            terms = line.strip().split(",")
            r, uni, cou = terms[:3]
            rank_dict[uni.strip()] = i + 1
    return rank_dict

def get_college_dict():
    # data/worldcollege
    # wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv
    path = f"{DATA_DIR}/data/worldcollege/wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv"
    college = pd.read_csv(path)
    ids = set(college["Winner ID"].unique().tolist() + college["Loser ID"].unique().tolist())
    ids = sorted(ids)
    id2name = {}
    for i in range(len(college.index)):
        row = college.iloc[i, :]
        w_id, l_id = row["Winner ID"], row["Loser ID"]
        w_name, l_name = row["Winner Text"], row["Loser Text"]
        if w_id not in id2name:
            id2name[w_id] = w_name
        if l_id not in id2name:
            id2name[l_id] = l_name
        
    return id2name

def get_max_degree_ids(college, max_num):
    ids = set(college["Winner ID"].unique().tolist() + college["Loser ID"].unique().tolist())
    ids = sorted(ids)
    # Count by edges
    edges_count = {id:0 for id in ids}
    for i in range(len(college.index)):
        row = college.iloc[i, :]
        winner, loser = row["Winner ID"], row["Loser ID"]
        edges_count[winner] += 1
        edges_count[loser] += 1
    sorted_edges_count = sorted(edges_count.items(), key=lambda x: x[1])
    
    # max_num = 40
    ids = [term[0] for term in sorted_edges_count[-max_num:]]
    ids = sorted(ids)
    return ids


def get_rank_ids(college, max_num):
    ids = set(college["Winner ID"].unique().tolist() + college["Loser ID"].unique().tolist())
    ids = sorted(ids)
    rank_path = f"{DATA_DIR}/data/worldcollege/results-ranking-QS-region-World-year-2017-ranking-QS-region-World-year-2017-export.csv"
    rank_dict = get_rank(rank_path)
    max_rank = sorted(rank_dict.items(), key=lambda x: x[1])[:max_num]
    id2name = get_college_dict()
    
    name2id = {process_name(val):key for key, val in id2name.items()}
    ids = []
    for college, r in max_rank:
        if college not in name2id:
            print("Not exist", college, r)
        else:
            ids.append(name2id[college])
    ids = sorted(ids)
    return ids   

def get_random_ids(college, max_num):
    ids = set(college["Winner ID"].unique().tolist() + college["Loser ID"].unique().tolist())
    ids = np.random.choice(ids, max_num, replace=False)
    ids = sorted(ids)
    return ids

def data_resample(X, y, res_n):
    n, p = X.shape
    res_idxs = np.random.randint(0, 2, res_n)
   
    res_num = np.sum(res_idxs)
    # print(res_idxs, res_num)
    # sss
    aug_n = len(res_idxs) - res_num
    sel_idxs = np.random.choice(len(X), res_num, replace=True)
    
    X = np.concat([X[sel_idxs], np.zeros((aug_n, p))], axis=0)
    y = np.concat([y[sel_idxs], np.random.binomial(n=1, p=0.5, size=aug_n)], axis=0)
    # print(X, y, X.shape, y.shape)
    # sss
    return X, y, aug_n

def data_load(path=None, max_num=50, sel_type="max_degree", D_type="full"):
    # data/worldcollege
    # wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv
    if path is None:
        path = "/home/hlinbh/Projects/SplitKnockoffs-Python/data/worldcollege/wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv"
    college = pd.read_csv(path)
    # Random
    if sel_type == "max_degree":
        ids = get_max_degree_ids(college, max_num)
    elif sel_type == "rank":
        ids = get_rank_ids(college, max_num)
    else:
        ids = get_random_ids(college, max_num)
    
    id2ind = {id:i for i, id in enumerate(ids)}
    Xs, ys = [], []
    Ds = {}
    for i in range(len(college.index)):
        row = college.iloc[i, :]
        winner, loser = row["Winner ID"], row["Loser ID"]
        if not (winner in id2ind and loser in id2ind):
            continue
            
        x_comp = np.zeros(len(ids))
        x_comp[id2ind[winner]], x_comp[id2ind[loser]] = 1, -1
        comp = (id2ind[winner], id2ind[loser])
        y = 1.0
        # Make Sure that 1 before -1
        if winner > loser:
            x_comp =  -x_comp
            comp = (id2ind[loser], id2ind[winner])
            y = 0.0
        Xs.append(x_comp), ys.append(y)
        if comp not in Ds:
            Ds[comp] = len(Ds)

    
    p = len(ids)
    if D_type == "full":
        D_matrix = np.zeros((p * (p - 1) // 2, p))
        count = 0
        for i_ in range(p):
            for j_ in range(i_+1, p):
                D_matrix[count][i_] = -1
                D_matrix[count][j_] = 1
                count += 1
    elif D_type == "sampled":
        for D_item in Ds:
            a, b = D_item
            D_comp = np.zeros(len(ids))
            D_comp[a], D_comp[b] = 1, -1
            D_matrix.append(D_comp)
    return np.stack(Xs).astype(np.float32), np.array(ys).astype(np.float32), D_matrix.astype(np.float32), id2ind

def data_load_aug(path=None, max_num=50, res_scale=2,
                  data_type="resample", sel_type="max_degree", D_type="full"):
    # data/worldcollege
    # wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv
    if path is None:
        path = f"{DATA_DIR}/data/worldcollege/wikisurvey_colleges_votes_2017-11-26T07_15_02Z.csv"
    college = pd.read_csv(path)
    # Random
    if sel_type == "max_degree":
        ids = get_max_degree_ids(college, max_num)
    elif sel_type == "rank":
        ids = get_rank_ids(college, max_num)
    else:
        ids = get_random_ids(college, max_num)
    
    id2ind = {id:i for i, id in enumerate(ids)}
    Xs, ys = [], []
    Ds = {}
    for i in range(len(college.index)):
        row = college.iloc[i, :]
        winner, loser = row["Winner ID"], row["Loser ID"]
        if not (winner in id2ind and loser in id2ind):
            continue
            
        x_comp = np.zeros(len(ids))
        x_comp[id2ind[winner]], x_comp[id2ind[loser]] = 1, -1
        comp = (id2ind[winner], id2ind[loser])
        y = 1.0
        # Make Sure that 1 before -1
        if winner > loser:
            x_comp =  -x_comp
            comp = (id2ind[loser], id2ind[winner])
            y = 0.0
        Xs.append(x_comp), ys.append(y)
        if comp not in Ds:
            Ds[comp] = len(Ds)
    
    # Create D
    p = len(ids)
    if D_type == "full":
        D_matrix = np.zeros((p * (p - 1) // 2, p))
        count = 0
        for i_ in range(p):
            for j_ in range(i_+1, p):
                D_matrix[count][i_] = -1
                D_matrix[count][j_] = 1
                count += 1
    elif D_type == "sampled":
        D_matrix = []
        for D_item in Ds:
            a, b = D_item
            D_comp = np.zeros(len(ids))
            D_comp[a], D_comp[b] = 1, -1
            D_matrix.append(D_comp)
        D_matrix = np.array(D_matrix)
    # Augment X
    n = len(Xs)
    Xs, ys, D = np.stack(Xs).astype(np.float32), np.array(ys).astype(np.float32), D_matrix.astype(np.float32)
    aug_n = n
    if "augment" in data_type:
        if "rand" in data_type:
            aug_n = np.random.negative_binomial(n, 0.5)
        Xs = np.concat([Xs, np.zeros((aug_n, p))], axis=0)
        aug_ys = np.random.binomial(n=1, p=0.5, size=len(ys))
        ys = np.concatenate([ys, aug_ys], axis=0)
    
    elif "resample" in data_type:
        Xs, ys, aug_n = data_resample(Xs, ys, n * res_scale)
    else:
        raise NotImplemented
     
    return Xs, ys, D, id2ind, aug_n

def unit_simulation_cv(
    exp_name,
    data_type,
    con_type,
    model_type,
    D_type,
    sel_type,
    Z_types,
    W_types,
    T_types,
    seed,
    max_num,
    res_scale,
    split_ratio,
    lambdas,
    nus,
    q,
    normalize,
    num_tests,
    is_modelX=False,
):  
    pkl_dumps, json_dumps = [], []
    for num_test in range(num_tests):
        aug_n = 0
        if ("augment" in data_type or 
            "resample" in data_type):
            X, y, D, id2ind, aug_n = data_load_aug(max_num=max_num, res_scale=res_scale, data_type=data_type, sel_type=sel_type, D_type=D_type)

        else:
            X, y, D, id2ind = data_load(max_num=max_num, sel_type=sel_type, D_type=D_type)
        print("Shape Check", X.shape, y.shape)
        n = X.shape[0]
        # Estimate Sigma    
        Sigma_hat = X.T @ X / n
        
        Sigma_M = np.eye(D.shape[0])
        
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


        # if normal: loss and model according to model_type
        # others: y model according to data_type, loss according to model_type
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
            aug_n=aug_n,
        )
        pkl_dumps.append({
                "Sigma_hat": Sigma_hat,
                "Sigma_M": Sigma_M,
                "id2ind":id2ind,
                "whole_results": whole_results,
                "D": D,
                "X":X,
                "y":y,
                "lambdas": lambdas,
                "nus": nus,
                "fdr": fdr_split,
                "power": power_split,
                "seed": seed,
                "num_test":num_test,
            })  
        pkl.dump(
            pkl_dumps,
            open(f"{result_dir}/{exp_name}/{lambda_tag}_{nu_tag}.pkl", "wb"),
        )
        print(f"Finish Test {num_test}.")
    


def run():
    q = args.q
    split_ratio = 0.2
    normalize = False
    step_size = 0.5
    num_tests = 100

    max_num = args.max_num
    res_scale = args.res_scale
    data_type = args.data_type
    con_type = args.con_type
    model_type = args.model_type
    D_type = args.D_type
    sel_type = args.sel_type
    
    Z_types = (
        "db",
        # "mos",
    )  # how to compute the Z, used for the solver, like different optimization problem
    W_types = (
        "bc",
        "lcd",
    )  # how to compute the W, used for the solver, how to derive W from Z
    T_types = (
        "k",
        "k+",
    )  # how to compute the Z, used for the solver, hot the decide the T

    seed = 65  
    type_key = f"{model_type}_" + "_".join(Z_types + W_types)
    whole_lambdas, whole_nus = [np.power(10, np.arange(-7, -7.6, -0.1))], [np.power(10, np.arange(0, 0.6, 0.1))]

    for lambdas, nus in zip(whole_lambdas, whole_nus):
        exp_name = f"{type_key}_max{max_num}_{sel_type}_{D_type}_res{res_scale}_N{num_tests}"
        unit_simulation_cv(
            exp_name,
            data_type,
            con_type,
            model_type,
            D_type,
            sel_type,
            Z_types,
            W_types,
            T_types,
            seed,
            max_num,
            res_scale,
            split_ratio,
            lambdas,
            nus,
            q,
            normalize,
            num_tests
        )
        print(f"Finish test {exp_name}.")


if __name__ == "__main__":
    run()
    