import json

# Define parameters for the Amsterdam emoped simulation

params = {
# Filenames
'indir' : "INPUTS/",
'outdir' : "OUTPUTS/",
'figdir' : 'figures/',
'fn_network' : "network_pt_short.json",
'fn_odlayer' : "od_layer_od.json",
'fn_transit' : "",
'fn_demand' : "custom_demand_city.csv",
#'fn_emoped_ff_init' : "init_pos_emoped.csv",
'fn_emoped_st_init' : "emoped_stations600.csv",

# Vehicles speeds
'V_EMOPED' : 7,
'V_BUS' : 5.5,
'V_TRAM' : 8.3,
'V_METRO' : 13,

# Transit connection (m)
'DIST_MAX' : 500,
'DIST_CONNECTION_OD' : 300,
'DIST_CONNECTION_PT' : 300,
'DIST_CONNECTION_MIX' : 300,

# Travel costs
'VOT' : 20 / 3600, # EUR/s
'FEE_EMOPED_TIME' : 0.33 / 60,  # 33cts/min
'FEE_EMOPED_BASE' : 1, # EUR

# Demand area
'polygon_demand' : [
    [620000, 5.811e6],
    [637000, 5.811e6],
    [637000, 5.794e6],
    [620000, 5.794e6]
]
}

with open('params.json', 'w') as f:
    json.dump(params, f, indent=4)