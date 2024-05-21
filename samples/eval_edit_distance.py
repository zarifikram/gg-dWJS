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
# classifier_model = HERClassifierModel.load_from_checkpoint("../checkpoints/her_classifier/last.ckpt")


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
    
    edit_distances = []
    # Calculate total distance
    for i in range(len(sequences)):
        for j in range(i+1, len(sequences)):
            edit_distances.append(calculate_edit_distance(sequences[i], sequences[j]))
    
    edit_distances = np.array(edit_distances)
    return edit_distances.mean(), edit_distances.std()

seqs = [heavy + light for heavy, light in zip(df.fv_heavy_aho, df.fv_light_aho)]
seqs = [seq.replace("-", "") for seq in seqs]
print(f"Unique percentage: {unique_percentage(seqs)}")
ed = average_edit_distance(seqs)
print(f"Average edit distance: {ed[0]} +- {ed[1]}")

