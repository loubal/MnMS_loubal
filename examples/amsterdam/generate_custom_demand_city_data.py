import numpy as np
import pandas as pd
import json
from mnms.tools.geometry import points_in_polygon

### Parameters
## Parameters file
f = open('params.json')
params = json.load(f)

fname_in = 'inputs/Data_city/demand_city_hex_9.csv'
fname_out = 'inputs/custom_demand_city.csv'
t_start = 7*3600
t_end = 9*3600
peak = 'OS' # OS, AS, RD -> 7-9/16-18/off
modes = ['OV', 'FI'] # PA, OV, FI -> car, PT, bike

labels = [mode+peak+'_trips' for mode in modes]

#ratio = 0.3
polygon = np.asarray(params['polygon_demand'])
dist_min = 250 # m

df_dmd_all = pd.read_csv(fname_in, sep=',')

### Filter demand

# select O/D in polygon
pts_o = list(df_dmd_all.O_hex_xy.apply(lambda x: [float(y) for y in x[1:-1].split(', ')]))
pts_d = list(df_dmd_all.D_hex_xy.apply(lambda x: [float(y) for y in x[1:-1].split(', ')]))
mask_o = points_in_polygon(polygon, pts_o)
mask_d = points_in_polygon(polygon, pts_d)
df_dmd = df_dmd_all[mask_o & mask_d]

# remove small dist
pts_o = list(df_dmd.O_hex_xy.apply(lambda x: [float(y) for y in x[1:-1].split(', ')]))
pts_d = list(df_dmd.D_hex_xy.apply(lambda x: [float(y) for y in x[1:-1].split(', ')]))
dist = [np.sqrt((o[0]-d[0])**2 + (o[1]-d[1])**2) for o,d in zip(pts_o, pts_d)]
mask_dist = [d>=dist_min for d in dist]
df_dmd = df_dmd[mask_dist]

# remove empty od
mask = df_dmd.apply(lambda row: np.asarray([row[label]==0 for label in labels]).any(), axis=1)
df_dmd = df_dmd[mask]
df_dmd.reset_index(drop=True, inplace=True)

### Generate agents

agents_id = []
agents_o = []
agents_d = []
agents_dept_time = []
i_agent=1
for i, row in df_dmd.iterrows():
    o = row['O_hex_xy'][1:-1].replace(',','')
    d = row['D_hex_xy'][1:-1].replace(',','')
    nb_dmd = int(row[labels].sum()+0.5)
    for _ in range(nb_dmd):
        td = t_start + np.random.random()*(t_end-t_start)
        td_str = '%02i:%02i:%02i' %(td/3600, np.remainder(td,3600)/60, np.remainder(td,60))

        agents_id.append(i_agent)
        agents_o.append(o)
        agents_d.append(d)
        agents_dept_time.append(td_str)
        i_agent += 1

df_agents = pd.DataFrame({'ID': agents_id, 'DEPARTURE':agents_dept_time,
                          'ORIGIN':agents_o, 'DESTINATION':agents_d})
df_agents.sort_values(by = 'DEPARTURE', inplace=True)
df_agents.reset_index(drop=True, inplace=True)

print('%i agents created from a demand of %.2f' %(i_agent, sum([df_dmd[label].sum() for label in labels])))

### Save data

df_agents.to_csv(fname_out, sep = ';', index = False)