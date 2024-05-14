import pickle, gzip, os
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

folder = "./outputs/2024-05-04/"
egfn_data = []
seeds = [0, 1, 2]

modes = [
    'db_egfn_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedFalseri0.5_rpFalse_fbTrue',
    'db_egfn_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedFalseri0.5_rpTrue_fbFalse',
    'db_egfn_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedFalseri0.5_rpFalse_fbFalse',
    'db_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedFalseri0.5_rpFalse_fbFalse',
    'tb_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedTrueri0.5_rpFalse_fbFalse',
    'sac_ntrain2500_prbTrue_mutation_Truereplay_size16gamma1augmentedFalseri0.5_rpFalse_fbFalse'
]
labels = ['EGFN', 'GFN', 'GAFN', 'SAC']
labels = ['EGFN', 'EGFN w/ RE', 'EGFN w/o feedback', 'GFN ', 'GAFN', 'SAC']
# labels = ['EGFN w/o feedback', 'GFN0', 'GFN 16', 'SAC', 'TB EGFN', 'TB']
n_methods = len(labels)
n_sparsity = 1
n_dim = 1

for i in modes:
    data = []
    for j in seeds:
        with gzip.open(f"{folder}seed{j}/{i}/result.json", "rb") as f:
            data.append(pickle.load(f))
        print(i)
    egfn_data.append(data)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
for i in range(n_methods * n_sparsity * n_dim):
    
    x = egfn_data[i][0]['modes_dict'].keys()

    y = []
    for j in range(len(seeds)):
        y.append(list(egfn_data[i][j]['modes_dict'].values()))
        # if len(y[-1]) < 26: 
        #     y[-1].append(y[-1][-1])
    y = np.array(y)
    # calculate the 95% confidence interval
    y_mean = np.mean(y, axis=0)
    yerr = np.std(y, axis=0) * 1.96 / np.sqrt(len(seeds)) 
    axes.plot(x, y_mean, label=labels[i%n_methods])
    axes.fill_between(x, y_mean-yerr, y_mean+yerr, alpha=0.2)


# for i in range(7*3):
#     x = egfn_data[i][1]['modes_dict'].keys()
#     y = []
#     for j in range(len(seeds)):
#         y.append(list(egfn_data[i][j]['modes_dict'].values()))
#         if len(y[-1]) < 26: 
#             y[-1].append(y[-1][-1])
#     y = np.array(y)
#     # calculate the 95% confidence interval
#     y_mean = np.mean(y, axis=0)
#     yerr = np.std(y, axis=0) * 1.96 / np.sqrt(len(seeds))

#     axes[1][i//7].plot(x, y_mean, label=labels[i%7])
#     axes[1][i//7].fill_between(x, y_mean-yerr, y_mean+yerr, alpha=0.2)

titles = [r"$R_0 = 0.00001$", r"$R_0 = 0.0001$", r"$R_0 = 0.001$", r"$R_0 = 0.01$"]
xlabel = "Number of States Visited(x10\N{SUPERSCRIPT FOUR})"
for i in range(4):
    axes.set_ylabel("Number of Modes")
    # axes[1][0].set_ylabel("Number of Modes")
    axes.set_xlabel("Number of Training Steps")
    # axes[i][1].set_title(titles[i])

# Add a legend
# for axe1d in axes:
#     for ax in axe1d:
#         # ax.legend(loc='upper right')
axes.set_xlim([0, 2500])

# DRAW A DOTTED RED VERTICAL LINE AT X = 100 TO INDICATE THE END OF PRE-TRAINING
# axes.axvline(x=100, color='r', linestyle='--', label='Beginning of reward-estimation')
# Adjust layout for better appearance
        
# # add a single legend box for all the subplots
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1),fancybox=False, shadow=False, ncol=7)
# make sure the legend box is not cut off
plt.subplots_adjust(bottom=0.15)
# Common legend for all subplots

# Adjust layout to prevent clipping of legends
# plt.tight_layout(rect=[0, 0, 1, 0.95])

# plt.tight_layout()
plt.savefig("fig.png")
