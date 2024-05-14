import pandas as pd

df = pd.read_csv("data/poas.csv.gz", compression="gzip")

prefix = "You are an expert antibody engineer. I am going to give you examples of antibody heavy chain and light chain variable regions from the paired observed antibody space database. You will generate 10 new antibody heavy chain and light chain that are not in the database. Output the 10 samples as a python list. Here are the examples: "

heavy_plus_light_in_a_tuple = []

for i in range(10):
   heavy_plus_light_in_a_tuple.append((df.loc[i].fv_heavy_aho, df.loc[i].fv_light_aho))

print(prefix)
print(heavy_plus_light_in_a_tuple)