import pandas as pd
METHOD_NAME = "GPT3_5"
# the results obtained from GPT 3.5
generated_seqs = [
    'WSNKGFYVFD', 'WGSNGFYVFS', 'WHKSGFYVFD', 'WHLNGFYVFD', 'WRLPGFYVFS', 
    'WGATGFYVFD', 'WLPQGFYVFS', 'WQLPGFYVFD', 'WLPLGFYVFD', 'WLPHGFYVFD', 
    'WQKGGFYVFS', 'WSPAGFYVFD', 'WSKGGFYVFD', 'WSQPGFYVFD', 'WGRPGFYVFS', 
    'WGKNGFYVFD', 'WRLNGFYVFS', 'WSPNGFYVFS', 'WQPTGFYVFD', 'WSLNGFYVFS', 
    'WGQNGFYVFD', 'WGPAGFYVFS', 'WLLNGFYVFS', 'WSLPGFYVFS', 'WGHNGFYVFD', 
    'WQPAGFYVFS', 'WGKPGFYVFD', 'WSRPGFYVFS', 'WQLNGFYVFS', 'WSPRGFYVFD', 
    'WSPNGFYVFD', 'WSPQGFYVFS', 'WGPQGFYVFD', 'WGPNGFYVFS', 'WQLPGFYVFS', 
    'WSLQGFYVFD', 'WGPPGFYVFS', 'WRLPGFYVFD', 'WSPQGFYVFD', 'WGHQGFYVFD', 
    'WQKPGFYVFS', 'WRLQGFYVFD', 'WQKQGFYVFS', 'WGHQGFYVFS', 'WSKQGFYVFD', 
    'WRLQGFYVFS', 'WQLQGFYVFS', 'WGHQGFYVFD', 'WQKQGFYVFD', 'WGKQGFYVFS', 
    'WGKQGFYVFD', 'WSKQGFYVFS', 'WSLQGFYVFS', 'WQLQGFYVFD', 'WLLQGFYVFS', 
    'WGPQGFYVFS', 'WSKPGFYVFS', 'WSRQGFYVFS', 'WGHQGFYVFS', 'WLLQGFYVFD', 
    'WQKPGFYVFD', 'WGHQGFYVFD', 'WSPQGFYVFS', 'WLLQGFYVFD', 'WSKQGFYVFS', 
    'WLLQGFYVFD', 'WSKQGFYVFD', 'WSPQGFYVFD', 'WGKQGFYVFD', 'WSLQGFYVFD', 
    'WQLQGFYVFD', 'WRLQGFYVFD', 'WSKPGFYVFD', 'WSRQGFYVFD', 'WGKPGFYVFD', 
    'WGHQGFYVFD', 'WQLQGFYVFS', 'WGHQGFYVFS', 'WGKQGFYVFS', 'WSKQGFYVFS', 
    'WSLQGFYVFS', 'WRLQGFYVFS', 'WLLQGFYVFS', 'WSPQGFYVFS', 'WGKPGFYVFS', 
    'WSRQGFYVFS', 'WSPQGFYVFD', 'WSRQGFYVFD', 'WGKPGFYVFD', 'WQLQGFYVFD', 
    'WSRPGFYVFS', 'WSPRGFYVFS', 'WSPNGFYVFS', 'WSPAGFYVFS', 'WRLNGFYVFS', 
    'WQPTGFYVFS', 'WSLNGFYVFS', 'WSKGGFYVFS', 'WSQPGFYVFS', 'WSKGGFYVFD', 
    'WGRPGFYVFD', 'WQPGGFYVFS', 'WRLPGFYVFD', 'WHLNGFYVFD', 'WHKSGFYVFD'
]



data = {"fv_heavy_aho": generated_seqs} 

df = pd.DataFrame(
       data
)

df.to_csv(f"samples/HER2/{METHOD_NAME}.csv", index=False)

