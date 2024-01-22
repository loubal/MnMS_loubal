import json

# Define parameters for the Amsterdam emoped simulation

params = {
# Filenames
'indir' : "INPUTS/",
'outdir' : "OUTPUTS/",
'fn_network' : "network_pt_short.json",
'fn_odlayer' : "od_layer_clustered_200.json",
'fn_transit' : "",
'fn_demand' : "custom_demand.csv",
'fn_emoped_ff_init' : "init_pos_emoped.csv",
'fn_emoped_st_init' : "emoped_stations.csv",

# Vehicles speeds
'V_EMOPED' : 7,
'V_BUS' : 9,
'V_TRAM' : 11,
'V_METRO' : 13,

# Transit connection (m)
'DIST_MAX' : 500,
'DIST_CONNECTION_OD' : 200,
'DIST_CONNECTION_PT' : 100,
'DIST_CONNECTION_MIX' : 100,

# Travel costs
'VOT' : 20 / 3600,
'FEE_EMOPED_TIME' : 0.33 / 60,  # 33cts/min
'FEE_EMOPED_BASE' : 1
}

with open('params.json', 'w') as f:
    json.dump(params, f, indent=4)