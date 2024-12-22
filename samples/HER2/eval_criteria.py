import pandas as pd
import argparse
import numpy as np
import editdistance
from sklearn.preprocessing import LabelEncoder
import torch
from walkjump.utils import token_string_to_tensor

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="ggdwjs.csv", help="sample.csv from the benchmark foder")
args = parser.parse_args()
# get the data
df = pd.read_csv(args.dir)

from walkjump.model._her_classifier import HERClassifierModel
model = HERClassifierModel.load_from_checkpoint("../../checkpoints/her_classifier/last.ckpt")

TOKEN_GAP = "-"
TOKENS_AA = list("ARNDCEQGHILKMFPSTWYV")
TOKENS_AHO = sorted([TOKEN_GAP, *TOKENS_AA])

ALPHABET_AHO = LabelEncoder().fit(TOKENS_AHO)

# Function to calculate edit distance between two sequences
def calculate_edit_distance(seq1, seq2):
    return editdistance.eval(seq1, seq2)

# Function to calculate average edit distance
def unique_percentage(sequences):
    unique = set(sequences)
    return len(unique) / len(sequences)

def average_edit_distance(sequences):
    total_distance = 0
    num_pairs = 0
    
    editdistance = []
    # Calculate total distance
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            editdistance.append(calculate_edit_distance(sequences[i], sequences[j]))
         
    
    return np.mean(editdistance), np.std(editdistance)

def average_bind(sequences):
    alphabet = ALPHABET_AHO
    tensor = torch.stack([token_string_to_tensor(seq, alphabet, onehot=True).float() for seq in sequences])
    xhat = model.model(tensor)
    # xhat = torch.where(xhat > 0.5, 1, 0).squeeze(1).float()
    return torch.mean(xhat), torch.std(xhat)

seqs = df.fv_heavy_aho.to_list()
bind_mean, bind_std = average_bind(seqs)
print(f"Average binding: {bind_mean} +/- {bind_std}")

print(f"Unique percentage: {unique_percentage(seqs)}")

ed_mean, ed_std = average_edit_distance(seqs)
print(f"Average edit distance: {ed_mean} +/- {ed_std}")
