import os
import matplotlib.pyplot as plt
from mnms.graph.layers import SharedVehicleLayer

from mnms.mobility_service.on_demand import OnDemandDepotMobilityService
from mnms.mobility_service.vehicle_sharing import OnVehicleSharingMobilityService
from mnms.simulation import Supervisor
from mnms.demand import CSVDemandManager
from mnms.flow.MFD import Reservoir, MFDFlowMotor
from mnms.log import attach_log_file, LOGLEVEL, get_logger, set_all_mnms_logger_level, set_mnms_logger_level
from mnms.time import Time, Dt
from mnms.io.graph import load_graph, load_odlayer, save_odlayer
from mnms.travel_decision.logit import LogitDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_bbox_origin_destination_layer, generate_matching_origin_destination_layer
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.io.graph import save_transit_link_odlayer, load_transit_links
from mnms.vehicles.veh_type import Bike

from mnms.tools.render import draw_roads

import pandas as pd
import numpy as np
import random

# Run Amsterdam pilot for WP5, emoped competition with PT

indir = "INPUTS"
outdir = "OUTPUTS"

# set_all_mnms_logger_level(LOGLEVEL.WARNING)
set_mnms_logger_level(LOGLEVEL.INFO, ["mnms.simulation"])

# get_logger("mnms.graph.shortest_path").setLevel(LOGLEVEL.WARNING)
attach_log_file(outdir + '/simulation.log')

# 'DESTINATION_R_82604106' 'ORIGIN_E_83202447'

def calculate_V_MFD(acc):
    N = acc["CAR"]

    a = 18.323199544221808
    b = 4813.38024078608
    # V = a * np.exp(-N / (2 * b))
    # V = 11.5*(1-N/60000)
    # V = max(V, 0.001)  # min speed to avoid gridlock
    # V_TRAM_BUS = 0.7 * V
    V_EMOPED = 7
    V_BUS = 9
    V_TRAM = 11
    V_METRO = 13
    # return {"CAR": V, "METRO": 17, "BUS": V_TRAM_BUS, "TRAM": V_TRAM_BUS}
    return {"CAR": V_EMOPED, "BUS": V_BUS, "TRAM": V_TRAM, "METRO": V_METRO}

if __name__ == '__main__':
    NX = 14
    NY = 16
    DIST_CONNECTION = 1000

    mmgraph = load_graph(indir + "/amsterdam_tc.json")

    odlayer = generate_bbox_origin_destination_layer(mmgraph.roads, NX, NY)
    # odlayer = generate_matching_origin_destination_layer(mmgraph.roads)
    # save_odlayer(odlayer, outdir + "/odlayer.json")

    # odlayer = load_odlayer(indir + "/odlayer.json")

    mmgraph.add_origin_destination_layer(odlayer)
    # mmgraph.connect_origin_destination_layer(1e3)

    if not os.path.exists(indir + f"/transit_link_{NX}_{NY}_{DIST_CONNECTION}_grid.json"):
        mmgraph.connect_origin_destination_layer(DIST_CONNECTION)
        save_transit_link_odlayer(mmgraph, indir + f"/transit_link_{NX}_{NY}_{DIST_CONNECTION}_grid.json")
    else:
        load_transit_links(mmgraph, indir + f"/transit_link_{NX}_{NY}_{DIST_CONNECTION}_grid.json")

    #personal_car = PersonalMobilityService("PersonalVehicle")
    #personal_car.attach_vehicle_observer(CSVVehicleObserver(outdir + "/veh.csv"))
    #mmgraph.layers["CAR"].add_mobility_service(personal_car)

    # Vehicle sharing mobility service
    emoped1 = OnVehicleSharingMobilityService("emoped", dt_matching=0)
    emoped2 = OnVehicleSharingMobilityService("emoped", dt_matching=0)

    emoped_layer = SharedVehicleLayer(mmgraph.roads, 'emoped_layer', Bike, 7, services=[emoped1, emoped2],
                                      observer=CSVVehicleObserver("emoped.csv"))

    # Add stations
    emoped1.create_station('ES1', '0', 20, 5)
    emoped1.create_station('ES2', '4', 20, 5)

    tram_service = PublicTransportMobilityService("TRAM")
    tram_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "/veh.csv"))
    mmgraph.layers["TRAMLayer"].add_mobility_service(tram_service)

    metro_service = PublicTransportMobilityService("METRO")
    metro_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "/veh.csv"))
    mmgraph.layers["METROLayer"].add_mobility_service(metro_service)

    mmgraph.connect_inter_layers(["CAR", "TRAMLayer", "METROLayer"], 100)

    print("Input file \n")
    # --------------- Input file with a percenatge of uber -------------------
    # demand_file = indir + "/test_all_in_highway_7h_9h.csv"          # Input demand csv
    demand_file = indir + "/demand.csv"

    demand = CSVDemandManager(demand_file)
    demand.add_user_observer(CSVUserObserver(outdir + "/user.csv"), user_ids="all")

    flow_motor = MFDFlowMotor(outfile=outdir + "/flow.csv")
    flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["CAR", "TRAM", "METRO"], calculate_V_MFD))
    #flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["CAR"], calculate_V_MFD))

    travel_decision = LogitDecisionModel(mmgraph, outfile=outdir + "/path.csv")

    supervisor = Supervisor(graph=mmgraph,
                            flow_motor=flow_motor,
                            demand=demand,
                            decision_model=travel_decision,
                            outfile=outdir + "/travel_time_link.csv")

    def gc_emoped(mlgraph, link, costs):
        gc = (0.33/60 + 20/3600) * link.length / costs['EMOPED']['speed']
        return gc

    def gc_metro(mlgraph, link, costs):
        gc = 20/3600 * link.length / costs['METRO']['speed']
        return gc

    def gc_bus(mlgraph, link, costs):
        gc = 20/3600 * link.length / costs['BUS']['speed']
        return gc

    def gc_tram(mlgraph, link, costs):
        gc = 20/3600 * link.length / costs['TRAM']['speed']
        return gc

    supervisor._mlgraph.add_cost_function(layer_id = 'EMOPEDLayer', cost_name = 'gen_cost',
                                          cost_function = gc_emoped)
    supervisor._mlgraph.add_cost_function(layer_id='METROLayer', cost_name='gen_cost',
                                          cost_function=gc_metro)
    supervisor._mlgraph.add_cost_function(layer_id='TRAMLayer', cost_name='gen_cost',
                                          cost_function=gc_tram)
    supervisor._mlgraph.add_cost_function(layer_id='BUSLayer', cost_name='gen_cost',
                                          cost_function=gc_bus)

    # supervisor.run(Time('07:00:00'), Time('07:10:00'), Dt(minutes=1), 10)
