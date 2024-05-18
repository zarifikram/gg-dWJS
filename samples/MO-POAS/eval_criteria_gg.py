import pandas as pd
import argparse
import numpy as np
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns

# use argparse to get the choise of ggdwjs or dwjs
# parser = argparse.ArgumentParser()
# parser.add_argument("--dir", type=str, default="some data", help="link to the data")

# data_dirs = ['ab_mo-gg-dWJS_arbeta_0_1.csv', 'ab_mo-gg-dWJS_arbeta_1_0.csv', 'ab_mo-gg-dWJS_arbeta_05_05.csv']
data_dirs = ['gg500_0_1.csv', 'gg500_1_0.csv', 'gg500_05_05.csv']

# get the data
dfs = [pd.read_csv(data_dir) for data_dir in data_dirs]

beta_sheets_list = []
aromaticity_list = []
instability_indices_list = []
for i, df in enumerate(dfs):
    # make a list of the sequences combined
    sequences = [heavy+light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)]

    # remove '-' from the sequences
    sequences = [seq.replace("-", "") for seq in sequences]

    # get protein analysis object for each sequence
    from Bio.SeqUtils.ProtParam import ProteinAnalysis

    sequences = [ProteinAnalysis(str(seq)) for seq in sequences]

    # get a list of the beta sheet percentages
    beta_sheets = np.array([seq.secondary_structure_fraction()[2] for seq in sequences])
    instability_indices = np.array([seq.instability_index() for seq in sequences])
    aromaticity = np.array([seq.aromaticity() for seq in sequences])
    
    beta_sheets_list.append(beta_sheets)
    aromaticity_list.append(aromaticity)
    instability_indices_list.append(instability_indices)

beta_sheets_list = np.array(beta_sheets_list)
aromaticity_list = np.array(aromaticity_list)
instability_indices_list = np.array(instability_indices_list)



# print the shape
print(f"beta sheet percentages: {beta_sheets_list.shape}")
print(f"instability indices: {instability_indices_list.shape}")
print(f"aromaticity: {aromaticity_list.shape}")

# calculate their friedman test
from scipy.stats import friedmanchisquare
f_beta, p_beta = friedmanchisquare(*beta_sheets_list)
f_ii, p_ii = friedmanchisquare(*instability_indices_list)
f_ar, p_ar = friedmanchisquare(*aromaticity_list)
print("                 f  |    p    |    w1=1,w2=0   | w1=0,w2=1  | w1=0.5,w2=0.5  ")
print(f"Beta sheet percentage | {f_beta} | {p_beta} | {beta_sheets_list[0].mean()}+= {beta_sheets_list[0].std()} | {beta_sheets_list[1].mean()}+= {beta_sheets_list[1].std()} | {beta_sheets_list[2].mean()}+= {beta_sheets_list[2].std()} ")
print(f"Aromaticity | {f_ar} | {p_ar} | {aromaticity_list[0].mean()}+= {aromaticity_list[0].std()} | {aromaticity_list[1].mean()}+= {aromaticity_list[1].std()} | {aromaticity_list[2].mean()}+= {aromaticity_list[2].std()} ")
print(f"Instability index | {f_ii} | {p_ii} | {instability_indices_list[0].mean()}+= {instability_indices_list[0].std()} | {instability_indices_list[1].mean()}+= {instability_indices_list[1].std()} | {instability_indices_list[2].mean()}+= {instability_indices_list[2].std()} ")

# check if they are stati
# print(f"beta sheet percentages: {beta_sheets.mean()} +- {beta_sheets.std()}")
# print(f"instability indices: {instability_indices.mean()} +- {instability_indices.std()}")
# print(f"aromaticity: {aromaticity.mean()} +- {aromaticity.std()}")

# plot the a scatter plot of the beta sheet percentages (x-axis) and the aromaticity (y-axis)
import matplotlib.pyplot as plt
import seaborn as sns

# plot the beta sheet percentages
# fig, ax = plt.subplots()
# sns.scatterplot(x=beta_sheets_list[0], y=instability_indices_list[0], label="w1=1,w2=0")
# sns.scatterplot(x=beta_sheets_list[1], y=instability_indices_list[1], label="w1=0,w2=1")
# sns.scatterplot(x=beta_sheets_list[2], y=instability_indices_list[2], label="w1=0.5,w2=0.5")
# plt.xlabel("Beta sheet percentage")
# plt.ylabel("Instability index")
# plt.legend()
# plt.show()

# next attempt
# fix, axes = plt.subplots(1, 3, figsize=(15, 3), sharey=True)
# # plt.figure(figsize=(8, 6))
# for i in range(3):
#     df = pd.DataFrame({'Beta_sheet_percentage': beta_sheets_list[i], 'Instability': instability_indices_list[i]})
#     sns.kdeplot(
#     x=df['Beta_sheet_percentage'], 
#     y=df['Instability'], 
#     fill=True, 
#     cmap="viridis", 
#     levels=100, 
#     thresh=0,
#     cbar=True,
#     ax=axes[i]
#     )
# # plt.colorbar(hb, label='Counts')
# axes[1].set_xlabel('Beta sheet percentage')
# axes[0].set_ylabel('Instability Index')
# plt.title('Hexbin Heatmap')
# plt.show()


fig, ax = plt.subplots(1, 1, figsize=(5, 5), sharey=True)


# Plot each KDE plot
for i in range(3):
    if i == 2:
        inds = [k for k in range(len(instability_indices_list[i])) if instability_indices_list[i][k] > 50]
        df = pd.DataFrame({'beta': beta_sheets_list[i][inds], 'instability': instability_indices_list[i][inds]})
    else:
        df = pd.DataFrame({'beta': beta_sheets_list[i], 'instability': instability_indices_list[i]})
    sns.kdeplot(ax=ax, data=df, x='beta', y='instability', fill=True, alpha=0.6)

# Set consistent limits for X and Y axes
x_limits = (0.35, 0.48)
y_limits = (25, 75)
ax.set_xlim(x_limits)
ax.set_ylim(y_limits)

# Set labels for X and Y axes
ax.set_xlabel('Beta Sheet Percentage', fontsize=12)
ax.set_ylabel('Instability Index', fontsize=12)

# Create legend with labels and colors
labels = ["[1, 0]", "[0, 1]", "[0.5, 0.5]"]
colors = ["#FF7F50", "#47B9D2", "#78EE84"]
legend_elements = [Patch(facecolor=color, edgecolor=color, label=label) for label, color in zip(labels, colors)]
ax.legend(handles=legend_elements, title='', loc='upper right', fontsize=10)

plt.tight_layout()  # Adjust layout to prevent overlap of labels
plt.show()