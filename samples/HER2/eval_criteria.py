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

seqs = df.fv_heavy_aho.to_list()
print(f"Unique percentage: {unique_percentage(seqs)}")
print(f"Average edit distance: {average_edit_distance(seqs)}")

