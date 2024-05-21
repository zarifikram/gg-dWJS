import pandas as pd
import argparse
import numpy as np
import editdistance

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="ggdwjs.csv", help="sample.csv from the benchmark foder")
args = parser.parse_args()
# get the data
df = pd.read_csv(args.dir)

# from walkjump.model._her_classifier import HERClassifierModel
# classifier_model = HERClassifierModel.load_from_checkpoint("../../../checkpoints/her_classifier/last.ckpt")


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

seqs = df.fv_heavy_aho.to_list()
print(f"Unique percentage: {unique_percentage(seqs)}")
ed_mean, ed_std = average_edit_distance(seqs)
print(f"Average edit distance: {ed_mean} +/- {ed_std}")

