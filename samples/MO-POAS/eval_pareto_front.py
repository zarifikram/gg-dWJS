import pandas as pd
import argparse
import numpy as np

data_dirs = ['gg500_0_1.csv', 'gg500_1_0.csv', 'gg500_05_05.csv']

# get the data
dfs = [pd.read_csv(data_dir) for data_dir in data_dirs]


combined_list = []
for i, df in enumerate(dfs):
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
    
    combined_list.extend([(beta_sheet, instability_index) for beta_sheet, instability_index in zip(beta_sheets, instability_indices)])

def is_dominated(point, other_point):
    """Returns True if other_point dominates point."""
    return (other_point[0] <= point[0] and other_point[1] <= point[1]) and other_point != point


def pareto_front(points):
    """Returns the Pareto front of a given list of points."""
    pareto_points = []
    for point in points:
        if not any(is_dominated(point, other_point) for other_point in points):
            pareto_points.append(point)
    return pareto_points

# Find Pareto front
pareto_points = pareto_front(combined_list)

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plotting the results
# Convert lists of tuples into separate lists for plotting
beta_sheets, instability_indices = zip(*combined_list)
pareto_beta_sheets, pareto_instability_indices = zip(*pareto_points)

plt.figure(figsize=(4, 4))
plt.scatter(beta_sheets, instability_indices, s=30, alpha=0.5, color='blue', label='All Points', edgecolors='none')
plt.scatter(pareto_beta_sheets, pareto_instability_indices, s=60, color='red', label='Pareto Front', edgecolors='black')
plt.xlabel('Beta Sheet Percentage', fontsize=14)
plt.ylabel('Instability Index', fontsize=14)
# plt.title('Pareto Front of Beta Sheet Percentage vs. Instability Index', fontsize=16)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

pareto_points_sorted = np.array(sorted(pareto_points))  # Sort points for correct line drawing

for i in range(len(pareto_points_sorted) - 1):
    plt.plot([pareto_points_sorted[i][0], pareto_points_sorted[i + 1][0]],
             [pareto_points_sorted[i][1], pareto_points_sorted[i + 1][1]],
             color='red', linestyle='-', linewidth=2)

plt.tight_layout()
plt.savefig('pareto_front.png', dpi=300)
plt.show()