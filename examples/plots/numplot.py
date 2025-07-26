import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl
import re

PROJECT_DIR = os.path.dirname(os.getcwd())

def plot(tag, save_path, nums, mean_fdrs, mean_powers, std_fdrs, std_powers):
    # Set up the figure
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    # Define the keys and their corresponding colors and styles
    
    keys = ["db_lcd_k", "db_lcd_k+"]
    plot_keys = ["MSX", "MSX+"]
    fdr_colors = ['#1b9e77', '#7570b3']  # Teal, Purple-blue
    power_colors = ['#d95f02', '#e7298a']  # Burnt orange, Magenta
    power_styles = ['--', '--']  # Dashed lines for Power
    # Plot each method
    for i, key in enumerate(keys):
        # Plot FDR
        plt.plot(nums, mean_fdrs[key], color=fdr_colors[i], label=f'FDR for {plot_keys[i]}')
        plt.fill_between(nums, 
                         np.clip(mean_fdrs[key] - std_fdrs[key], 0, 1.0), 
                         np.clip(mean_fdrs[key] + std_fdrs[key], 0, 1.0), 
                         color=fdr_colors[i], alpha=0.2)
        
        # Plot Power
        plt.plot(nums, mean_powers[key], color=power_colors[i], 
                 linestyle=power_styles[i], label=f'Power for {plot_keys[i]}')
        plt.fill_between(nums, 
                         np.clip(mean_powers[key] - std_powers[key], 0, 1.0), 
                         np.clip(mean_powers[key] + std_powers[key], 0, 1.0), 
                         color=power_colors[i], alpha=0.1)
    
    
    # Set axis limits and labels
    plt.ylim(0, 1.05)
    plt.xlabel("Num of Samples", fontsize=15)
    # Add legend and save
    # plt.title(tag)
    plt.legend(loc="center right", fontsize=15)
    # plt.legend(loc="center right", fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def parse_plot_tag(tag):
    matches = re.findall(r'([a-zA-Z])([\d.]+)', tag)
    key_dict = {'p':'p', 'k':'sparsity', 'A':'scale'}
    result = ', '.join([f"{key_dict[key]} = {value}" for key, value in matches if key in [ 'p', 'A', 'k']])
    return result

def normal_info(covtag, dtag):
    tag_pat = "logistic_db_bc_lcd_" + covtag + "_N200_n{}p100c0.3A1.0k20_mosek_D" + dtag
    data_dir = f"{PROJECT_DIR}/simu_exp/results/splitmodelx/data_normal/con_normal/" + "{}"
    save_dir = "{}/simu_exp/collect_results/splitmodelx/numplots/data_normal/con_normal/{}".format(PROJECT_DIR, tag_pat.format("x"))
    os.makedirs(save_dir, exist_ok=True)

    para_tag = "lam-4.000--2.000_nu2.000-4.000"
    nums = np.arange(100, 1100, 100)
    return tag_pat, data_dir, save_dir, para_tag, nums

def lin_normal_info(covtag, dtag):
    tag_pat = "linear_db_bc_lcd_" + covtag + "_N200_n{}p100c0.3A1.0k20_mosek_D" + dtag
    data_dir = f"{PROJECT_DIR}/simu_exp/results/splitmodelx/data_normal/con_normal/" + "{}"
    save_dir = "{}/simu_exp/collect_results/splitmodelx/numplots/data_normal/con_normal/{}".format(PROJECT_DIR, tag_pat.format("x"))
    os.makedirs(save_dir, exist_ok=True)

    para_tag = "lam-4.000--2.000_nu2.000-4.000"
    nums = np.arange(100, 1100, 100)
    return tag_pat, data_dir, save_dir, para_tag, nums

def pairwise_info(mtag, covtag, dtag):
    tag_pat = f"{mtag}_db_bc_lcd_{covtag}_N200_" + "n{}" + f"p10c0.3A5.0k{dtag}_mosek_D0"
    data_dir = f"{PROJECT_DIR}/simu_exp/results/splitmodelx/data_resample/con_pairwise_fixaugM/" + "{}"
    save_dir = "{}/simu_exp/collect_results/splitmodelx/numplots/data_resample/con_pairwise_fixaugM/{}".format(PROJECT_DIR, tag_pat.format("x"))
    os.makedirs(save_dir, exist_ok=True)

    para_tag = "lam-6.000--1.000_nu2.000-3.500"
    nums = np.arange(100, 1100, 100)
    return tag_pat, data_dir, save_dir, para_tag, nums

def save_and_plot(tag_pat, data_dir, save_dir, para_tag, nums):
    save_path = f"{save_dir}/final.png"

    mean_fdrs, mean_powers = {}, {}
    std_fdrs, std_powers = {}, {}

    for num in nums:
        tag = tag_pat.format(num)
        data_path = f"{data_dir.format(tag)}/{para_tag}.pkl"
        result = pkl.load(open(data_path, 'rb'))
        fdr_split, power_split = result["fdr"], result["power"]
        if len(mean_fdrs) == 0:
            for key in fdr_split:
                mean_fdrs[key], mean_powers[key] = [], []
                std_fdrs[key], std_powers[key] = [], []
        for key in fdr_split:
            mean_fdrs[key].append(np.array(fdr_split[key]).mean(axis=0))
            mean_powers[key].append(np.array(power_split[key]).mean(axis=0))
            
            std_fdrs[key].append(np.array(fdr_split[key]).std(axis=0))
            std_powers[key].append(np.array(power_split[key]).std(axis=0))
        
    for key in mean_fdrs:
        mean_fdrs[key] = np.array(mean_fdrs[key])
        mean_powers[key] = np.array(mean_powers[key])
        std_fdrs[key] = np.array(std_fdrs[key])
        std_powers[key] = np.array(std_powers[key])


    pkl.dump({"nums":nums, 
            "mean_fdrs":mean_fdrs, "std_fdrs":std_fdrs,
            "mean_powers":mean_powers, "std_powers":std_powers}, open(save_path.replace(".png", ".pkl"), "wb"))

    plot(parse_plot_tag(tag), save_path, nums, mean_fdrs, mean_powers, std_fdrs, std_powers)

################################################################################################
# Plots for Nuplots Normal distribution
for covtag in ["est", "gt"]:
    for dtag in ["0", "1", "2"]:
        tag_pat, data_dir, save_dir, para_tag, nums = normal_info(covtag, dtag)
        save_and_plot(tag_pat, data_dir, save_dir, para_tag, nums)

################################################################################################
# Plots for Nuplots Pairwise distribution
for mtag in ["linear", "logistic"]: 
    for covtag in ["est"]:
        for dtag in ["0.1", "0.3", "0.5"]:
            tag_pat, data_dir, save_dir, para_tag, nums = pairwise_info(mtag, covtag, dtag)
            save_and_plot(tag_pat, data_dir, save_dir, para_tag, nums)


