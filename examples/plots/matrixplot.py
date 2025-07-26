import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import os

from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.dirname(os.path.dirname(PROJECT_DIR))

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

def get_res(path, s_type="S_db_bc_k"):
    datas = pkl.load(open(path, 'rb'))
    results = []
    for data in datas:
        # print(data.keys())
        whole_results = data["whole_results"]
        # select = whole_results[f"S_db_bc_k"]
        select = whole_results[s_type]
        id2ind = data["id2ind"]
        ind2id = {val:key for key, val in id2ind.items()}
        D = data["D"]
        X = data["X"]
        
        comps = []
        for sel_idx in select:
            x_comp = D[sel_idx]
            inds = np.argwhere(x_comp != 0).tolist()
            ids = (ind2id[inds[0][0]], ind2id[inds[1][0]])
            comps.append(ids)
        
        sampled_comps = []
        X_num = 0
        for i in range(len(D)):
            X_comp = D[i]
            inds = np.argwhere(X_comp != 0).tolist()
            if len(inds) == 0: continue
            X_num += 1
            ids = (ind2id[inds[0][0]], ind2id[inds[1][0]])
            if ids not in sampled_comps:sampled_comps.append(ids)
        # print("X, D shape", X.shape, D.shape, len(D), X_num)
        results.append({"id2ind":id2ind, "comps":comps, "sampled_comps":sampled_comps})
    return results

def get_res_base(path, s_type="S_db_bc_k"):
    data = pkl.load(open(path, 'rb'))
    print(data.keys())
    whole_results = data["whole_results"]
    # select = whole_results[f"S_db_bc_k"]
    select = whole_results[s_type]
    id2ind = data["id2ind"]
    # print("id2ind", id2ind, len(id2ind))
    ind2id = {val:key for key, val in id2ind.items()}
    D = data["D"]
    # print("D shape", D.shape)
    comps = []
    for sel_idx in select:
        x_comp = D[sel_idx]
        inds = np.argwhere(x_comp != 0).tolist()

        ids = (ind2id[inds[0][0]], ind2id[inds[1][0]])
        comps.append(ids)
    return id2ind, comps
    # print(select, id2ind, X)

    
def get_rank(path):
    rank_dict = {}
    with open(path, 'r') as rf:
        r_i = 1
        for i, line in enumerate(rf):
            if i == 0: continue
            if "World Rank" in line: continue
            terms = line.strip().split(",")
            r, uni, cou = terms[:3]
            rank_dict[uni.strip()] = r_i
            r_i += 1
    return rank_dict


def plot_matrix(counts, whole_counts, rank_dict, save_path):
    max_num = max(rank_dict.values())
    sel_matrix = np.zeros((max_num, max_num))
    sel_matrix[np.arange(len(sel_matrix)), np.arange(len(sel_matrix))] = 1.0

    plt.figure()
    comps = list(whole_counts.keys())
    for (rank1, rank2) in comps:
        if rank1 > rank2:
            rank1, rank2 = rank2, rank1
        if (counts[(rank1, rank2)] + counts[(rank2, rank1)]) > 0:
            sel_matrix[rank1-1, rank2-1] = 0.7  
        else:
            sel_matrix[rank1-1, rank2-1] = 0.3  

    sns.heatmap(sel_matrix, cmap='Blues', cbar=False, square=True,)
    plt.xlabel('College Relative Rank')
    plt.ylabel('College Relative Rank')
    plt.savefig(save_path)
    plt.close()

REVERSE_COLLEGE_MAP = {
    "Massachusetts Institute of Technology - MIT":"Massachusetts Institute of Technology",
    "California Institute of Technology - Caltech":"California Institute of Technology",
    "Swiss Federal Institute of Technology Zurich - ETHZ":"ETH Zürich – Swiss Federal Institute of Technology Zürich",
    "Swiss Federal Institute of Technology Lausanne - EPFL":"École Polytechnique Fédérale de Lausanne, Switzerland",
    "University of Michigan - Ann Arbor":"University of Michigan",
    "University of Hong Kong":"The University of Hong Kong",
    "University of California - Berkeley":"University of California, Berkeley",
    "University of California - Los Angeles":"University of California, Los Angeles",
    "The University of Tokyo": "University of Tokyo",
    "London School of Economics":"London School of Economics and Political Science"
}
    
    
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

def main():   
    pre_tag = "data_resample/con_pairwise_fixaugM/logistic_db_bc_lcd_max80_rank_sampled_res3_N100"
    tags = [f"{pre_tag}/lam-7.500--7.000_nu0.000-0.500.pkl"]

    for tag in tags:
        #  "S_db_bc_k+", "S_db_lcd_k", "S_db_lcd_k+"
        for s_type in ["S_db_bc_k",]:
            res_path = f"{PROJECT_DIR}/college_exp/results/splitmodelx/{tag}"
            rank_path = f"{DATA_DIR}/data/worldcollege/results-ranking-QS-region-World-year-2017-ranking-QS-region-World-year-2017-export.csv"
            id2name = get_college_dict()
            rank_dict = get_rank(rank_path)   
            results = get_res(res_path, s_type)

            counts = defaultdict(int)  ## {"i-j": time}
            whole_counts = defaultdict(int)
            sampled_counts = defaultdict(int)
            comp_record = defaultdict(int)
            max_num = 0
            for res in results:
                selid2ind, comps, sampled_comps = res["id2ind"], res["comps"], res["sampled_comps"]
                sel_names = [process_name(id2name[key]) for key in selid2ind]
                
                names_rank = {}
                for sel_name in sel_names:
                    if sel_name not in rank_dict:
                        print("no", sel_name)
                        rank_dict[sel_name] = len(rank_dict)
                    names_rank[sel_name] = rank_dict[sel_name]
                
                for name1 in names_rank:
                    for name2 in names_rank:
                        if name1 != name2:
                            rank1 = names_rank[name1]
                            rank2 = names_rank[name2]
                            whole_counts[(rank1, rank2)] += 1
                

                for comp in sampled_comps:
                    # print(id2name[comp[0]], id2name[comp[1]])
                    name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                    rank1, rank2 = names_rank[name1], names_rank[name2]
                    sampled_counts[(rank1, rank2)] += 1


                for comp in comps:
                    name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                    rank1, rank2 = names_rank[name1], names_rank[name2]
                    comp_record[(name1, name2, rank1, rank2)] += 1
                    counts[(rank1, rank2)] += 1
               
                if max_num < len(comps):
                    # max_counts, max_sampled_counts = counts, sampled_counts
                    max_counts, max_sampled_counts = defaultdict(int), defaultdict(int)
                    for comp in sampled_comps:
                        # print(id2name[comp[0]], id2name[comp[1]])
                        name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                        rank1, rank2 = names_rank[name1], names_rank[name2]
                        max_sampled_counts[(rank1, rank2)] += 1

                    for comp in comps:
                        # print(id2name[comp[0]], id2name[comp[1]])
                        name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                        rank1, rank2 = names_rank[name1], names_rank[name2]
                        max_counts[(rank1, rank2)] += 1

            save_dir =  f"{PROJECT_DIR}/college_exp/results/matrixplots"
            save_path = f"{save_dir}/" + tag.replace("pkl", "png").replace("lam", f"{s_type}_lam")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plot_matrix(max_counts, max_sampled_counts, names_rank, save_path)
                       
main()