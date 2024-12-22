import pandas as pd
import argparse
import numpy as np

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir1", type=str, default="ggdwjs_beta", help="ggdwjs_beta, ggdwjs_ii, ggdwjs_beta_ii, dwjs, train")
parser.add_argument("--dir2", type=str, default="ggdwjs_beta", help="ggdwjs_beta, ggdwjs_ii, ggdwjs_beta_ii, dwjs, train")

directory1 = parser.parse_args().dir1
directory2 = parser.parse_args().dir2

# get the data
df1 = (
    pd.read_csv(directory1)
    if not "gz" in directory1
    else pd.read_csv(directory1, compression="gzip")
)

df2 = (
    pd.read_csv(directory2)
    if not "gz" in directory2
    else pd.read_csv(directory2, compression="gzip")
)

# make a list of the sequences combined
sequences1 = [heavy + light for heavy, light in zip(df1.fv_heavy_aho, df1.fv_light_aho)]
sequences2 = [heavy + light for heavy, light in zip(df2.fv_heavy_aho, df2.fv_light_aho)]

# remove '-' from the sequences
sequences1 = [seq.replace("-", "") for seq in sequences1]
sequences2 = [seq.replace("-", "") for seq in sequences2]

# get protein analysis object for each sequence
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sequences1 = [ProteinAnalysis(str(seq)) for seq in sequences1]
sequences2 = [ProteinAnalysis(str(seq)) for seq in sequences2]

# get a list of the beta sheet percentages
beta_sheets1 = np.array([seq.secondary_structure_fraction()[2] for seq in sequences1])
instability_indices1 = np.array([seq.instability_index() for seq in sequences1])
aromaticity1 = np.array([seq.aromaticity() for seq in sequences1])

beta_sheets2 = np.array([seq.secondary_structure_fraction()[2] for seq in sequences2])
instability_indices2 = np.array([seq.instability_index() for seq in sequences2])
aromaticity2 = np.array([seq.aromaticity() for seq in sequences2])

# do paired t-test and print the resutls
from scipy.stats import ttest_rel

print(f"beta sheet percentages: {ttest_rel(beta_sheets1, beta_sheets2)}")
print(f"instability indices: {ttest_rel(instability_indices1, instability_indices2)}")
print(f"aromaticity: {ttest_rel(aromaticity1, aromaticity2)}")
