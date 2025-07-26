import numpy as np
import pandas as pd
import pickle as pkl
import seaborn as sns
import matplotlib.pyplot as plt
import os

from collections import defaultdict

PROJECT_DIR = os.path.dirname(os.getcwd())
DATA_DIR = os.path.dirname(PROJECT_DIR)

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
        # print("id2ind", id2ind, len(id2ind))
        ind2id = {val:key for key, val in id2ind.items()}
        D = data["D"]
        X = data["X"]

        comps = []
        for sel_idx in select:
            x_comp = D[sel_idx]
            inds = np.argwhere(x_comp != 0).tolist()
            if len(inds) != 2: continue
            ids = (ind2id[inds[0][0]], ind2id[inds[1][0]])
            comps.append(ids)

        sampled_comps = []
        for i in range(len(D)):
            D_comp = D[i]
            inds = np.argwhere(D_comp != 0).tolist()
            if len(inds) == 0: continue
            ids = (ind2id[inds[0][0]], ind2id[inds[1][0]])
            sampled_comps.append(ids)

        # comps = [comp for comp in comps if comp in sampled_comps or reversed(comp) in sampled_comps]
        results.append({"id2ind":id2ind, "comps":comps, "sampled_comps":sampled_comps})
    return results

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

def plot_prob(counts, whole_counts, save_path):
    # To ensure consistent style
    sns.set_theme(style="whitegrid")

    # grouped = {}
    # step_size = 3
    if len(counts) == 0:
        return
    
    for step_size in [3]:
        max_num = max(counts.keys()) + 2
        grouped = {}
        for i in range(0, max_num, step_size): 
            group_sum = sum(counts.get(j, 0) for j in range(i, i+step_size))
            whole_group_sum = sum(whole_counts.get(j, 0) for j in range(i, i+step_size))
            if whole_group_sum != 0 and group_sum != 0:
                if whole_group_sum < group_sum:
                    print(i, group_sum, whole_group_sum)
                    sss
                grouped[f"{i}-{i+step_size-1}"] = group_sum / whole_group_sum


        # Extract keys and values
        labels = list(grouped.keys())
        values = list(grouped.values())

        # Create the plot
        plt.figure(figsize=(8, 5))
        ax = sns.barplot(x=labels, y=values, palette="pastel")

        plt.xlabel("Rank Difference Group", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.xticks(rotation=90)
        plt.tight_layout()
        os.makedirs(os.path.join(os.path.dirname(save_path), f"step{step_size}"), exist_ok=True)
        save_path_ =  os.path.join(os.path.dirname(save_path), f"step{step_size}", os.path.basename(save_path))
        plt.savefig(save_path_)
        print(f"Finish {save_path_}.")

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

    rank_path = f"{DATA_DIR}/data/worldcollege/results-ranking-QS-region-World-year-2017-ranking-QS-region-World-year-2017-export.csv"
    rank_dict = get_rank(rank_path)   
    for tag in tags:
        print(f"Tag:{tag}")
        # "S_db_bc_k+", "S_db_lcd_k", "S_db_lcd_k+"
        for s_type in ["S_db_bc_k", ]:
            res_path = f"{PROJECT_DIR}/college_exp/results/splitmodelx/{tag}"
            id2name = get_college_dict()
            results = get_res(res_path, s_type)

            counts = defaultdict(int)  ## {"i-j": time}
            whole_counts = defaultdict(int)
            sampled_counts = defaultdict(int)
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
                            whole_counts[np.abs(rank1 - rank2)] += 1
                
                for comp in sampled_comps:
                    name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                    rank1, rank2 = names_rank[name1], names_rank[name2]
                    sampled_counts[np.abs(rank1 - rank2)] += 1

                for comp in comps:
                    name1, name2 = process_name(id2name[comp[0]]), process_name(id2name[comp[1]])
                    rank1, rank2 = names_rank[name1], names_rank[name2]
                    counts[np.abs(rank1 - rank2)] += 1

            save_dir =  f"{PROJECT_DIR}/college_exp/results/probplots"
            save_path = f"{save_dir}/" + tag.replace("pkl", "png").replace("lam", f"{s_type}_lam")        
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plot_prob(counts, whole_counts, save_path)
            plot_prob(counts, sampled_counts, save_path.replace(".png", "_sampled.png"))

if __name__ == "__main__":
    main()
