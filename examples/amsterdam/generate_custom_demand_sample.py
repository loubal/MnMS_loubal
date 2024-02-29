import numpy as np
import pandas as pd
import json
from mnms.tools.geometry import points_in_polygon

### Parameters
## Parameters file
f = open('params.json')
params = json.load(f)

fname_in = 'inputs/custom_demand_city_full.csv'
fname_out = 'inputs/custom_demand_city.csv'

RATIO = 0.1
RANDOM_STATE = 79678

### Take a sample from full demand

df_agents_full = pd.read_csv(fname_in, sep=';')

nb_sample = int(len(df_agents_full)*RATIO)
df_agents = df_agents_full.sample(nb_sample, random_state=RANDOM_STATE)

df_agents.sort_values(by = 'DEPARTURE', inplace=True)
df_agents.reset_index(drop=True, inplace=True)

print('%i agents created' %(len(df_agents)))

### Save data

df_agents.to_csv(fname_out, sep = ';', index = False)