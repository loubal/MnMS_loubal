import numpy as np
import pandas as pd
from mnms.tools.geometry import points_in_polygon

### Parameters
fname_in = 'inputs/Data_MATSim/demand_all.csv'
fname_out = 'inputs/custom_demand.csv'
t_start = 7*3600
t_end = 9*3600
modes = ['car', 'walk', 'bike', 'taxi', 'pt']
ratio = 0.5
polygon = np.asarray([
    [628000, 5.81e6],
    [632500, 5.808e6],
    [634000, 5.8055e6],
    [634000, 5.803e6],
    [633000, 5.801e6],
    [630500, 5.7995e6],
    [629000, 5.8e6],
    [625500, 5.8e6],
    [625500, 5.803e6],
    [625500, 5.806e6]
])
min_dist = 500 # m

df_dmd_all = pd.read_csv(fname_in, sep=';')

### Filter demand
df_dmd_all['dept_float'] = df_dmd_all.apply( lambda x: sum([int(y)*60**(2-i) for i,y in enumerate(x.DEPARTURE.split(':'))]), axis=1 )
df_dmd_time = df_dmd_all[(df_dmd_all['dept_float'] >= t_start) & (df_dmd_all['dept_float'] < t_end)]
df_dmd_mode = df_dmd_time[df_dmd_time['SERVICE'].isin(modes)]

# select users based on O/D locations
pts_o = list(df_dmd_mode.ORIGIN.apply(lambda x: [float(y) for y in x.split(' ')]))
pts_d = list(df_dmd_mode.DESTINATION.apply(lambda x: [float(y) for y in x.split(' ')]))
mask_o = points_in_polygon(polygon, pts_o)
mask_d = points_in_polygon(polygon, pts_d)

df_dmd = df_dmd_mode[mask_o & mask_d]

print('original %i; after time %i; after modes %i; after OD %i' %(len(df_dmd_all),
      len(df_dmd_time), len(df_dmd_mode), len(df_dmd)))

nb_sample = int(len(df_dmd)*ratio)
df_sample = df_dmd.sample(nb_sample)
df_sample.sort_values(by='dept_float', inplace=True)
df_sample.drop(columns=['SERVICE', 'dept_float'], inplace=True)


df_sample.to_csv(fname_out, sep = ';', index = False)