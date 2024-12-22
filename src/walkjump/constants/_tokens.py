from sklearn.preprocessing import LabelEncoder

TOKEN_GAP = "-"
TOKEN_GAP_AMP = "X"
TOKENS_AA = list("ARNDCEQGHILKMFPSTWYV")
TOKENS_AHO = sorted([TOKEN_GAP, *TOKENS_AA])
TOKENS_AMP = sorted([TOKEN_GAP_AMP, *TOKENS_AA])

ALPHABET_AHO = LabelEncoder().fit(TOKENS_AHO)
ALPHABET_AMP = LabelEncoder().fit(TOKENS_AMP)