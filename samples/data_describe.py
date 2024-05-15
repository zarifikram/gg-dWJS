import pandas as pd   
                
                
dataset = pd.read_csv("data/poas_train_2.csv.gz", compression="gzip")
print(f"aromaticity {dataset.aromaticity.mean()}+-{dataset.aromaticity.std()} max: {dataset.aromaticity.max()} min: {dataset.aromaticity.min()} median: {dataset.aromaticity.median()}")
print(f"beta sheet perc {dataset.ss_perc_sheet.mean()}+-{dataset.ss_perc_sheet.std()} max: {dataset.ss_perc_sheet.max()} min: {dataset.ss_perc_sheet.min()} median: {dataset.ss_perc_sheet.median()}")
print(f"instability index {dataset.instability_index.mean()}+-{dataset.instability_index.std()} max: {dataset.instability_index.max()} min: {dataset.instability_index.min()} median: {dataset.instability_index.median()}")