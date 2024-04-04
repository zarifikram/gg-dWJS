import pandas as pd
# from src.walkjump.utils._tokenize import token_string_to_tensor
# from src.walkjump.constants import ALPHABET_AHO
# from dataclasses import InitVar, dataclass, field
# from sklearn.preprocessing import LabelEncoder
# from src.walkjump.data._dataset import AbWithLabelDataset, AbDataset
# from src.walkjump.data._batch import AbBatch, AbWithLabelBatch
# from src.walkjump.data._batch import MNISTBatch


# dataset = pd.read_csv("data/poas.csv.gz", compression="gzip")
dataset2 = pd.read_csv("data/poas_train.csv.gz", compression="gzip")

print(dataset2.columns)
# print(ALPHABET_AHO)
# alphabet_or_token_list: InitVar[LabelEncoder | list[str]] = ALPHABET_AHO
# alphabet = (
#             alphabet_or_token_list
#             if isinstance(alphabet_or_token_list, LabelEncoder)
#             else LabelEncoder().fit(alphabet_or_token_list)
#         )

# ds = AbWithLabelDataset(dataset2, alphabet)
# # ds = AbDataset(dataset, alphabet)
# entry = AbWithLabelBatch.from_tensor_pylist([ds.__getitem__(0)])
# # print(entry)
# print(entry.x.shape)

# # for i in range(5):
# #     x, y = ds.__getitem__(i)
# #     print(x.shape, y)

# from src.walkjump.model.arch import ByteNetMLPArch
# bytenet_mlp = ByteNetMLPArch(
#     n_tokens= 21,
#     d_model= 128,
#     n_layers= 35,
#     kernel_size= 3,
#     max_dilation= 128,
# )
# print(bytenet_mlp(entry.x))
# print(entry.y.shape)
# for i in range(5):
#     print(token_string_to_tensor(dataset.iloc[i].fv_heavy_aho, alphabet), " ", token_string_to_tensor(dataset.iloc[i].fv_light_aho, alphabet))

# for i in range(5):
#     print(token_string_to_tensor(dataset2.iloc[i].HeavyAA_aligned, alphabet), " ", token_string_to_tensor(dataset2.iloc[i].LightAA_aligned, alphabet), f" sasa {dataset2.iloc[i].sasa} beta percent {dataset2.iloc[i].ss_perc_sheet}")    

# print(dataset2.ss_perc_sheet.max(), " ", dataset2.ss_perc_sheet.min())
# row = dataset.iloc[100]
# k = token_string_to_tensor(row.fv_heavy_aho, alphabet)
# print(k)
# print(k.shape)
# data_module = MNISTDataset(dataset, alphabet)
# print(data_module.__len__())
# item = data_module.__getitem__(5)
# print(item.shape)
# batch = MNISTBatch.from_tensor_pylist([item])
# print(f"batch: {batch}")