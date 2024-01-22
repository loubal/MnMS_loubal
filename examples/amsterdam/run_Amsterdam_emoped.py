import os
import matplotlib.pyplot as plt
from mnms.graph.layers import SharedVehicleLayer

from mnms.mobility_service.vehicle_sharing import VehicleSharingMobilityService
from mnms.simulation import Supervisor
from mnms.demand import CSVDemandManager
from mnms.flow.MFD import Reservoir, MFDFlowMotor
from mnms.log import attach_log_file, LOGLEVEL, get_logger, set_all_mnms_logger_level, set_mnms_logger_level
from mnms.time import Time, Dt
from mnms.io.graph import load_graph, load_odlayer, save_odlayer
from mnms.travel_decision.logit import LogitDecisionModel
from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_layer_from_roads
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.io.graph import save_transit_link_odlayer, load_transit_links
from mnms.vehicles.veh_type import Bike
from mnms.graph.layers import MultiLayerGraph

from mnms.tools.render import draw_roads

import pandas as pd
import numpy as np
import random
import time
import json

# Run Amsterdam pilot for WP5, emoped competition with PT
###############


# Load parmaeters
f = open('params.json')
params = json.load(f)

indir = "INPUTS/"
outdir = "OUTPUTS/"

# set_all_mnms_logger_level(LOGLEVEL.WARNING)
set_mnms_logger_level(LOGLEVEL.INFO, ["mnms.simulation"])

# get_logger("mnms.graph.shortest_path").setLevel(LOGLEVEL.WARNING)
attach_log_file(outdir + 'simulation.log')

# 'DESTINATION_R_82604106' 'ORIGIN_E_83202447'

def calculate_V_MFD(acc):
    V_EMOPED = params['V_EMOPED']
    V_BUS = 9
    V_TRAM = 11
    V_METRO = 13
    return {"BIKE": V_EMOPED,"BUS": V_BUS, "TRAM": V_TRAM, "METRO": V_METRO}

if __name__ == '__main__':

    t0_run = time.time()

    USE_BUS = True

    # Transit connection (m)
    DIST_MAX = params['DIST_MAX']
    DIST_CONNECTION_OD = 200
    DIST_CONNECTION_PT = 100
    DIST_CONNECTION_MIX = 100

    VOT = 20/3600
    FEE_MOPED_TIME = 0.33/60 # 33cts/min

    mmgraph = load_graph(indir + "network_pt_short.json")

    odlayer = load_odlayer(indir + "od_layer_clustered_200.json")

    mmgraph.add_origin_destination_layer(odlayer)

    # Vehicle sharing mobility service
    #mmgraph_roads = load_graph(indir + "new_network.json")  # to get the road network only
    df_emoped1 = pd.read_csv(indir + 'init_pos_emoped.csv')
    df_emoped2 = pd.read_csv(indir + 'init_pos_emoped.csv')

    emoped1 = VehicleSharingMobilityService("emoped1", free_floating_possible=True, dt_matching=0)
    emoped2 = VehicleSharingMobilityService("emoped2", free_floating_possible=True, dt_matching=0)
    emoped_layer1 = generate_layer_from_roads(mmgraph.roads, 'EMOPEDLayer1', SharedVehicleLayer, Bike, 7,
                                              [emoped1])
    emoped_layer2 = generate_layer_from_roads(mmgraph.roads, 'EMOPEDLayer2', SharedVehicleLayer, Bike, 7,
                                              [emoped2])
    if USE_BUS:
        emoped_layer1.add_connected_layers(["BUSLayer", "TRAMLayer", "METROLayer"])
        emoped_layer2.add_connected_layers(["BUSLayer", "TRAMLayer", "METROLayer"])
    else:
        emoped_layer1.add_connected_layers(["TRAMLayer", "METROLayer"])
        emoped_layer2.add_connected_layers(["TRAMLayer", "METROLayer"])
    emoped1.attach_vehicle_observer(CSVVehicleObserver(outdir + "emoped1.csv"))
    emoped2.attach_vehicle_observer(CSVVehicleObserver(outdir + "emoped2.csv"))

    # Add stations
    init_emoped1 = df_emoped1.value_counts('closest_node')
    for node, nb in init_emoped1.items():
        emoped1.init_free_floating_vehicles(node, nb)
    init_emoped2 = df_emoped2.value_counts('closest_node')
    for node, nb in init_emoped2.items():
        emoped2.init_free_floating_vehicles(node, nb)

    # PT
    if USE_BUS:
        bus_service = PublicTransportMobilityService("BUS")
        bus_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "bus.csv"))
        mmgraph.layers["BUSLayer"].add_mobility_service(bus_service)

    tram_service = PublicTransportMobilityService("TRAM")
    tram_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "tram.csv"))
    mmgraph.layers["TRAMLayer"].add_mobility_service(tram_service)

    metro_service = PublicTransportMobilityService("METRO")
    metro_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "metro.csv"))
    mmgraph.layers["METROLayer"].add_mobility_service(metro_service)


    # Create graph

    if USE_BUS:
        mmgraph = MultiLayerGraph([mmgraph.layers["BUSLayer"], mmgraph.layers["TRAMLayer"], mmgraph.layers["METROLayer"],
                                   emoped_layer1, emoped_layer2], odlayer, DIST_CONNECTION_OD)
    else:
        mmgraph = MultiLayerGraph(
            [mmgraph.layers["TRAMLayer"], mmgraph.layers["METROLayer"],
             emoped_layer1, emoped_layer2], odlayer, DIST_CONNECTION_OD)
        mmgraph = MultiLayerGraph(
            [mmgraph.layers["TRAMLayer"], mmgraph.layers["METROLayer"],
             emoped_layer1], odlayer, DIST_CONNECTION_OD)

    # Connect PT layers
    print('Connect PT layers')
    if USE_BUS:
        mmgraph.custom_connect_inter_layers(["BUSLayer", "TRAMLayer", "METROLayer"], DIST_CONNECTION_PT,
                                        ensure_connect=True, max_connect_dist=DIST_MAX)
        mmgraph.custom_connect_intra_layer("BUSLayer", DIST_CONNECTION_PT, same_line=False)
    else:
        mmgraph.custom_connect_inter_layers(["TRAMLayer", "METROLayer"], DIST_CONNECTION_PT,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
    mmgraph.custom_connect_intra_layer("TRAMLayer", DIST_CONNECTION_PT, same_line=False)
    mmgraph.custom_connect_intra_layer("METROLayer", DIST_CONNECTION_PT, same_line=False)

    # Connect PT network to emoped #
    """print('Connect PT layers with EMOPED')
    if USE_BUS:
        mmgraph.custom_connect_inter_layers(["EMOPEDLayer1", "BUSLayer"], DIST_CONNECTION_MIX,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
        mmgraph.custom_connect_inter_layers(["EMOPEDLayer2", "BUSLayer"], DIST_CONNECTION_MIX,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
    mmgraph.custom_connect_inter_layers(["EMOPEDLayer1", "TRAMLayer"], DIST_CONNECTION_MIX,
                                        ensure_connect=True, max_connect_dist=DIST_MAX)
    mmgraph.custom_connect_inter_layers(["EMOPEDLayer2", "TRAMLayer"], DIST_CONNECTION_MIX,
                                        ensure_connect=True, max_connect_dist=DIST_MAX)
    mmgraph.custom_connect_inter_layers(["EMOPEDLayer1", "METROLayer"], DIST_CONNECTION_MIX,
                                        ensure_connect=True, max_connect_dist=DIST_MAX)
    mmgraph.custom_connect_inter_layers(["EMOPEDLayer2", "METROLayer"], DIST_CONNECTION_MIX,
                                        ensure_connect=True, max_connect_dist=DIST_MAX)"""

    # Connect odlayer
    #print('Connect OD layer')
    #mmgraph.connect_origindestination_layers(DIST_CONNECTION_OD)

    # Demand
    demand_file = indir + "custom_demand.csv"          # Input demand csv
    # demand_file = indir + "demand.csv"

    demand = CSVDemandManager(demand_file)
    demand.add_user_observer(CSVUserObserver(outdir + "user.csv"), user_ids="all")

    # Flow
    flow_motor = MFDFlowMotor(outfile=outdir + "flow.csv")
    if USE_BUS:
        flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["BIKE", "BUS", "TRAM", "METRO"], calculate_V_MFD))
    else:
        flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["BIKE", "TRAM", "METRO"], calculate_V_MFD))

    travel_decision = DummyDecisionModel(mmgraph, outfile=outdir + "path.csv", cost='gen_cost')

    supervisor = Supervisor(graph=mmgraph,
                            flow_motor=flow_motor,
                            demand=demand,
                            decision_model=travel_decision)#outfile=outdir + "travel_time_link.csv")

    def gc_emoped1(mlgraph, link, costs):
        gc = (FEE_MOPED_TIME + VOT) * link.length / costs['emoped1']['speed']
        return gc

    def gc_emoped2(mlgraph, link, costs):
        gc = (FEE_MOPED_TIME + VOT) * link.length / costs['emoped2']['speed']
        return gc

    def gc_metro(mlgraph, link, costs):
        gc = VOT * link.length / costs['METRO']['speed']
        return gc

    def gc_bus(mlgraph, link, costs):
        gc = VOT * link.length / costs['BUS']['speed']
        return gc

    def gc_tram(mlgraph, link, costs):
        gc = VOT * link.length / costs['TRAM']['speed']
        return gc

    def gc_transit(mlgraph, link, costs):
        gc = VOT * link.length / 1
        return gc

    supervisor._mlgraph.add_cost_function(layer_id = 'EMOPEDLayer1', cost_name = 'gen_cost',
                                          cost_function = gc_emoped1)
    #supervisor._mlgraph.add_cost_function(layer_id = 'EMOPEDLayer2', cost_name = 'gen_cost',
    #                                      cost_function = gc_emoped2)
    supervisor._mlgraph.add_cost_function(layer_id='METROLayer', cost_name='gen_cost',
                                          cost_function=gc_metro)
    supervisor._mlgraph.add_cost_function(layer_id='TRAMLayer', cost_name='gen_cost',
                                          cost_function=gc_tram)
    if USE_BUS:
        supervisor._mlgraph.add_cost_function(layer_id='BUSLayer', cost_name='gen_cost',
                                              cost_function=gc_bus)
    supervisor._mlgraph.add_cost_function(layer_id='TRANSIT', cost_name='gen_cost',
                                          cost_function=gc_transit)
    t1_run = time.time()
    print(t1_run-t0_run, 's loading time')
    print("Start simulation")
    supervisor.run(Time('07:00:00'), Time('07:05:00'), Dt(minutes=1), 1)

    t2_run = time.time()
    print(t2_run-t1_run, 's sim time')
