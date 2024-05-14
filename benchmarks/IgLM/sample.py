from iglm import IgLM
import pandas as pd
iglm = IgLM()
METHOD_NAME = "IgLM"
prompt_sequence = ""
chain_token = "[HEAVY]"
species_token = "[HUMAN]"
num_seqs = 1000

generated_seqs_heavy = iglm.generate(
    chain_token,
    species_token,  
    prompt_sequence=prompt_sequence,
    num_to_generate=num_seqs,
)

chain_token = "[LIGHT]"
generated_seqs_light = iglm.generate(
    chain_token,
    species_token,  
    prompt_sequence=prompt_sequence,
    num_to_generate=num_seqs,
)

data = {"fv_heavy_aho": generated_seqs_heavy, "fv_light_aho": generated_seqs_light} 

df = pd.DataFrame(
       data
)

df.to_csv(f"benchmarks/outputs/{METHOD_NAME}.csv", index=False)

