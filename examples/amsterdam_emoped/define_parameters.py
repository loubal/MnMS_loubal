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
#'fn_demand' : "custom_demand_city.csv",
'fn_demand' : "simplistic_demand_city_full.csv",
#'fn_emoped_ff_init' : "init_pos_emoped.csv",
'fn_emoped_st_init' : "emoped_stations30.csv",

# Vehicles speeds
'V_EMOPED' : 7,
'V_BUS' : 3,
'V_TRAM' : 8,
'V_METRO' : 10,

# Transit connection (m)
'DIST_MAX' : 500,
'DIST_CONNECTION_OD' : 300,
'DIST_CONNECTION_PT' : 300,
'DIST_CONNECTION_MIX' : 300,

# Travel costs
'VOT' : 30 / 3600, # EUR/s
'FEE_EMOPED_TIME' : 0.33 / 60,  # 33cts/min
'FEE_EMOPED_BASE' : 1, # EUR

# Demand area
'polygon_demand' : [
    [615000, 5.813e6],
    [638000, 5.813e6],
    [638000, 5.793e6],
    [615000, 5.793e6]
]
}

with open('params.json', 'w') as f:
    json.dump(params, f, indent=4)