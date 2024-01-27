import pandas as pd
import argparse
import numpy as np

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="ggdwjs", help="ggdwjs or dwjs")

mode_to_data_dir = {
    "ggdwjs_beta": "samples/ab_gg-dWJS_beta.csv",
    "ggdwjs_ii": "samples/ab_gg-dWJS_ii.csv",
    "ggdwjs_beta_ii": "samples/ab_gg-dWJS_beta_ii.csv",
    "dwjs": "samples/ab_dWJS_samples.csv",
    "train": "data/poas.csv.gz",
}

# get the data
df = (
    pd.read_csv(mode_to_data_dir[parser.parse_args().mode])
    if not "gz" in mode_to_data_dir[parser.parse_args().mode]
    else pd.read_csv(mode_to_data_dir[parser.parse_args().mode], compression="gzip")
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
print(f"beta sheet percentages: {beta_sheets.mean()} +- {beta_sheets.std()}")
print(f"instability indices: {instability_indices.mean()} +- {instability_indices.std()}")
