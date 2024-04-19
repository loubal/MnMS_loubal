import numpy as np
import pandas as pd
import json
from mnms.io.graph import load_graph

### Parameters
## Parameters file
f = open('params.json')
params = json.load(f)

fname_in = 'inputs/custom_demand_city_full.csv'
fname_out = 'inputs/custom_demand_city.csv'

RATIO = 0.1
RANDOM_STATE = 79678

# Retrieve PT stations positions
mmgraph_pt = load_graph(params["indir"] + params["fn_network"])
position_stations = []
for key in mmgraph_pt.roads.stops.keys():
    if 'METRO' in key or 'TRAM' in key:
        position_stations.append(mmgraph_pt.roads.stops[key].absolute_position)

### Take a sample from full demand

df_agents_full = pd.read_csv(fname_in, sep=';')

weights = np.ones(len(df_agents_full))

origin_points = np.asarray([[float(x) for x in s.split(' ')] for s in df_agents_full.ORIGIN.values])
destination_points = np.asarray([[float(x) for x in s.split(' ')] for s in df_agents_full.DESTINATION.values])

dist_o = []
for o in origin_points:
    distances = [(o[0]-s[0])**2+(o[1]-s[1])**2 for s in position_stations]
    dist_o.append(np.sqrt(min(distances)))
dist_d = []
for d in destination_points:
    distances = [(d[0]-s[0])**2+(d[1]-s[1])**2 for s in position_stations]
    dist_d.append(np.sqrt(min(distances)))

length_eucl = [((o[0]-d[0])**2+(o[1]-d[1])**2)**0.5 for o,d in zip(origin_points, destination_points)]

for i in range(len(df_agents_full)):
    if length_eucl[i] > 7e3:
        weights[i] = 0.7
    if length_eucl[i] > 1e4:
        weights[i] = 0.5
    if dist_o[i]>1000 or dist_d[i]>1000:
        weights[i] = 3

nb_sample = int(len(df_agents_full)*RATIO)
df_agents = df_agents_full.sample(nb_sample, random_state=RANDOM_STATE, weights=weights)

df_agents.sort_values(by = 'DEPARTURE', inplace=True)
df_agents.reset_index(drop=True, inplace=True)

print('%i agents created' %(len(df_agents)))

### Save data

df_agents.to_csv(fname_out, sep = ';', index = False)