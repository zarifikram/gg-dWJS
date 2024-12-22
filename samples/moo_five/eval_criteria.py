import pandas as pd
import argparse
import numpy as np

# use argparse to get the choise of ggdwjs or dwjs

# get the data
dirs = ["10000.csv", "01000.csv", "00100.csv", "00010.csv", "00001.csv"]

dfs = [pd.read_csv(dir_) for dir_ in dirs]


# make a list of the sequences combined
sequences = [[heavy + light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)] for df in dfs]

# remove '-' from the sequences
sequences = [[seq.replace("-", "") for seq in seqs] for seqs in sequences]

lengths = [np.array([len(seq) for seq in seqs]) for seqs in sequences]
# get protein analysis object for each sequence
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sequences = [[ProteinAnalysis(str(seq)) for seq in seqs] for seqs in sequences]

# get a list of the beta sheet percentages
# r_mol_w', 'r_inv_length', 'r_neg_gravy', 'r_beta_sheet', 'r_instability_index'

for i, seqs in enumerate(sequences):
    molecular_weights = np.array([seq.molecular_weight() for seq in seqs])
    
    gravies = np.array([seq.gravy() for seq in seqs])
    beta_sheets = np.array([seq.secondary_structure_fraction()[2] for seq in seqs])
    instability_indices = np.array([seq.instability_index() for seq in seqs])
    aromaticity = np.array([seq.aromaticity() for seq in seqs])
    
    # print(dirs[i])
    
    
    # print(f"molecular weight: {molecular_weights.mean()} +- {molecular_weights.std()}")
    # print(f"sequence length: {lengths[i].mean()} +- {lengths[i].std()}")
    # print(f"gravy: {gravies.mean()} +- {gravies.std()}")
    # print(f"beta sheet percentages: {beta_sheets.mean()} +- {beta_sheets.std()}")
    # print(f"instability indices: {instability_indices.mean()} +- {instability_indices.std()}")
    # print(f"aromaticity: {aromaticity.mean()} +- {aromaticity.std()}")

    # print the results in latex table format 2 decimal places
    print(f" [0, 0, 0, 0, 0] & {molecular_weights.mean():.2f} $\pm$ {molecular_weights.std():.2f} & {lengths[i].mean():.2f} $\pm$ {lengths[i].std():.2f} & {gravies.mean():.2f} $\pm$ {gravies.std():.2f} & {beta_sheets.mean():.2f} $\pm$ {beta_sheets.std():.2f} & {instability_indices.mean():.2f} $\pm$ {instability_indices.std():.2f}  \\\\")