import json

# Define parameters for the Amsterdam emoped simulation

params = {
# Filenames
'indir' : "INPUTS/",
'outdir' : "OUTPUTS/",
'figdir' : 'figures/',
'fn_network' : "network_pt_short.json",
'fn_odlayer' : "od_layer_od.json",
#'fn_demand' : "demand_city_cars.csv",
'fn_demand' : "custom_demand_city_tcs.csv",

# Vehicles speeds
'V_BUS' : 5.5,
'V_TRAM' : 8.3,
'V_METRO' : 13,
'V_BIKE' : 4,

# Transit connection (m)
'DIST_MAX' : 500,
'DIST_CONNECTION_OD' : 300,
'DIST_CONNECTION_PT' : 300,

# Travel costs
'VOT' : 20 / 3600, # EUR/s

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