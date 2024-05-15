import pandas as pd
import argparse
import numpy as np

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="ggdwjs_beta", help="ggdwjs_beta, ggdwjs_ii, ggdwjs_beta_ii, dwjs, train")

directory = parser.parse_args().dir

# get the data
df = (
    pd.read_csv(directory)
    if not "gz" in directory
    else pd.read_csv(directory, compression="gzip")
)

# make a list of the sequences combined
sequences = [heavy + light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)]

# remove '-' from the sequences
sequences = [seq.replace("-", "") for seq in sequences]

# get protein analysis object for each sequence
from Bio.SeqUtils.ProtParam import ProteinAnalysis

sequences = [ProteinAnalysis(str(seq)) for seq in sequences]

# get a list of the beta sheet percentages
beta_sheets = np.array([seq.secondary_structure_fraction()[2] for seq in sequences])
instability_indices = np.array([seq.instability_index() for seq in sequences])
aromaticity = np.array([seq.aromaticity() for seq in sequences])

print(f"beta sheet percentages: {beta_sheets.mean()} +- {beta_sheets.std()}")
print(f"instability indices: {instability_indices.mean()} +- {instability_indices.std()}")
print(f"aromaticity: {aromaticity.mean()} +- {aromaticity.std()}")
