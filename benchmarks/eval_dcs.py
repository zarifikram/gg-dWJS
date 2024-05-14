# script to perform distributional conformity score (DCS) evaluation
from tqdm import tqdm
import pandas as pd
import argparse
import numpy as np
from sklearn.neighbors import KernelDensity
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# use argparse to get the choise of ggdwjs or dwjs
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, default="outputs/IgLM.csv", help="sample.csv from the benchmark foder")
args = parser.parse_args()


# get training data
dataset = pd.read_csv("../data/poas.csv.gz", compression="gzip")
train_df = dataset[dataset.partition == "train"]
test_df = dataset[dataset.partition == "test"]

ref_seqs = [heavy+light for heavy, light in zip(train_df.fv_heavy_aho, train_df.fv_light_aho)]
ref_seqs = [seq.replace("-", "") for seq in ref_seqs]

val_seqs = [heavy+light for heavy, light in zip(test_df.fv_heavy_aho, test_df.fv_light_aho)]
val_seqs = [seq.replace("-", "") for seq in val_seqs]

# get the generated data
df = pd.read_csv(args.dir)
# make a list of the sequences combined
sequences = [heavy+light for heavy,light in zip(df.fv_heavy_aho, df.fv_light_aho)]
sample_seqs = [seq.replace("-", "") for seq in sequences]

k = 100  # number of samples to draw from the validation set
n = 1000  # number of samples to draw from the reference set


vals = val_seqs[: k - 1]
refs = ref_seqs[:n]


def get_kde(refs):
    # we calculate the conformity score as the KDE log probability of the sample. (The KDE is fit on the reference sequences.)
    # We estimate the global bandwidth of the kernel using Silverman’s method, set the adaptive local kernel bandwidth to 0.15, and employed a diagonal covariance matrix.
    # get the hydrophilicity and molecular weight of the reference sequences
    ref_hydros = [ProteinAnalysis(str(ref)).gravy() for ref in refs]
    ref_mol_weights = [ProteinAnalysis(str(ref)).molecular_weight() for ref in refs]

    # now we fit the KDE on the hydrophilicity and molecular weight of the reference sequences
    # use bandwidth of 0.15
    bandwidth = 0.15
    # fit the KDE on the hydrophilicity and molecular weight of the reference sequences
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
        np.array(list(zip(ref_hydros, ref_mol_weights)))
    )
    return kde


def conformity_score(sample, ref_kde):
    """Computes the conformity score of a sample with respect to a set of reference sequences.

    Parameters
    ----------
    sample : str
        The sample sequence.
    refs : list[str]
        The reference sequences.

    Returns
    -------
    float
        The conformity score.
    """
    # get the hydrophilicity and molecular weight of the sample
    sample_hydro = ProteinAnalysis(str(sample)).gravy()
    sample_mol_weight = ProteinAnalysis(str(sample)).molecular_weight()

    # get the log probability of the sample
    log_prob = ref_kde.score_samples(np.array([sample_hydro, sample_mol_weight]).reshape(-1, 2))
    # get the conformity score
    # print(log_prob)
    conformity_score = log_prob[
        0
    ]  # the log probability of the sample is the first element of the log_prob array (high log probability means high conformity)
    return conformity_score


ref_kde = get_kde(refs)
scores = []
for sample_id, sample in tqdm(enumerate(sample_seqs)):
    # add sample to the validation set (get a new list)
    target_vals = vals + [sample]
    conformity_scores = []
    for i, val in enumerate(target_vals):
        conformity_scores.append(conformity_score(val, ref_kde))

    # find how many of the conformity scores are greater than the conformity score of the sample
    conformity_scores = np.array(conformity_scores)
    num_less_than_sample = (conformity_scores < conformity_scores[-1]).sum()
    # print(f"DCS: {num_less_than_sample / k}")
    scores.append(num_less_than_sample / k)

print(f"DCS: {np.mean(scores)} +- {np.std(scores)}")
