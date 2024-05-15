import pandas as pd
import argparse
import numpy as np

# use argparse to get the choise of ggdwjs or dwjs
# parser = argparse.ArgumentParser()
# parser.add_argument("--dir", type=str, default="some data", help="link to the data")

data_dirs = ['ab_mo-gg-dWJS_arbeta_0_1.csv', 'ab_mo-gg-dWJS_arbeta_1_0.csv', 'ab_mo-gg-dWJS_arbeta_05_05.csv']

# get the data
dfs = [pd.read_csv(data_dir) for data_dir in data_dirs]

beta_sheets_list = []
aromaticity_list = []
instability_indices_list = []
for df in dfs:
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
fig, ax = plt.subplots()
sns.scatterplot(x=beta_sheets_list[0], y=-1*aromaticity_list[0], label="w1=1,w2=0")
sns.scatterplot(x=beta_sheets_list[1], y=-1*aromaticity_list[1], label="w1=0,w2=1")
sns.scatterplot(x=beta_sheets_list[2], y=-1*aromaticity_list[2], label="w1=0.5,w2=0.5")
plt.xlabel("Beta sheet percentage")
plt.ylabel("Aromaticity")
plt.legend()
plt.show()
# fix, axes = plt.subplots(1, 3, figsize=(10, 3))
# # plt.figure(figsize=(8, 6))
# for i in range(3):
#     df = pd.DataFrame({'Beta_sheet_percentage': beta_sheets_list[i], 'Aromaticity': -1*aromaticity_list[i]})
#     sns.kdeplot(
#     x=df['Beta_sheet_percentage'], 
#     y=df['Aromaticity'], 
#     fill=True, 
#     cmap="viridis", 
#     levels=100, 
#     thresh=0,
#     cbar=True,
#     ax=axes[i]
#     )
# # plt.colorbar(hb, label='Counts')
# plt.xlabel('Beta sheet percentage')
# plt.ylabel('Aromaticity')
# plt.title('Hexbin Heatmap')
# plt.show()