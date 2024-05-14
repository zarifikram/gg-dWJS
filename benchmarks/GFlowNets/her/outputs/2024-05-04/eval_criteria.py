import pandas as pd
import argparse
import numpy as np
import editdistance
from Bio.SeqUtils.ProtParam import ProteinAnalysis

import torch
from walkjump.utils._tokenize import token_string_to_tensor
from walkjump.constants import ALPHABET_AHO
from dataclasses import InitVar, dataclass, field
from sklearn.preprocessing import LabelEncoder

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="ggdwjs.csv", help="sample.csv from the benchmark foder")
args = parser.parse_args()
# get the data
df = pd.read_csv(args.dir)

from walkjump.model._her_classifier import HERClassifierModel
model = HERClassifierModel.load_from_checkpoint("../../../../../checkpoints/her_classifier/last.ckpt")
model.eval()
alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
alphabet = (
            alphabet_or_token_list
            if isinstance(alphabet_or_token_list, LabelEncoder)
            else LabelEncoder().fit(alphabet_or_token_list)
        )

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
    
    # Calculate total distance
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            total_distance += calculate_edit_distance(sequences[i], sequences[j])
            num_pairs += 1
    
    # Calculate average
    if num_pairs > 0:
        return total_distance / num_pairs
    else:
        return 0

def avarage_instability_index(sequences):
    seqs = [seq.replace("-", "") for seq in sequences]
    seqs = [ProteinAnalysis(str(seq)) for seq in seqs]
    instability_indices = np.array([seq.instability_index() for seq in seqs])
    return np.mean(instability_indices), np.std(instability_indices)

def average_bind(sequences):
    tensor = torch.stack([token_string_to_tensor(seq, alphabet, onehot=True).float() for seq in sequences])
    xhat = model.model(tensor)
    # xhat = torch.where(xhat > 0.5, 1, 0).squeeze(1).float()
    return torch.mean(xhat), torch.std(xhat)

def p_bind(sequences):  
    tensor = torch.stack([token_string_to_tensor(seq, alphabet, onehot=True).float() for seq in sequences])
    xhat = model.model(tensor)
    # xhat = torch.where(xhat > 0.5, 1, 0).squeeze(1).float()
    return xhat.tolist()

df = df[~(df.AASeq.str.contains('-'))]
seqs = df.AASeq.to_list()
# seqs = [seq for seq in seqs if "-" not in seq]
print(f"Unique percentage: {unique_percentage(seqs)}")
print(f"Average edit distance: {average_edit_distance(seqs)}")
print(f"P bind: {average_bind(seqs)[0]} +- {average_bind(seqs)[1]}")


# update the df with each sequences' instability index
df['p_bind'] = p_bind(seqs)

# take the top 100 sequences based on the instability index
top_100_seqs = df.sort_values(by='p_bind', ascending=False).head(500).AASeq.to_list()

print(f"Top 100 unique percentage: {unique_percentage(top_100_seqs)}")
print(f"Top 100 average edit distance: {average_edit_distance(top_100_seqs)}")
print(f"Top 100 p bind: {average_bind(top_100_seqs)[0]} +- {average_bind(top_100_seqs)[1]}")
