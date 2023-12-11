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
from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_bbox_origin_destination_layer, generate_matching_origin_destination_layer
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

    t0 = time.time()

    DIST_CONNECTION = 200

    mmgraph_pt = load_graph(indir + "/network_pt.json")

    odlayer = load_odlayer(indir + "/od_layer_clustered_200.json")

    #mmgraph.add_origin_destination_layer(odlayer)

    #personal_car = PersonalMobilityService("PersonalVehicle")
    #personal_car.attach_vehicle_observer(CSVVehicleObserver(outdir + "/veh.csv"))
    #mmgraph.layers["CAR"].add_mobility_service(personal_car)

    # Vehicle sharing mobility service
    emoped1 = OnVehicleSharingMobilityService("emoped1", free_floating_possible=True, dt_matching=0)
    emoped2 = OnVehicleSharingMobilityService("emoped2", free_floating_possible=True, dt_matching=0)

    emoped_layer1 = SharedVehicleLayer(mmgraph_pt.roads, 'EMOPEDLayer1', Bike, 7, services=[emoped1],
                                      observer=CSVVehicleObserver(outdir+"/emoped1.csv"), prefix='em1_')
    emoped_layer2 = SharedVehicleLayer(mmgraph_pt.roads, 'EMOPEDLayer2', Bike, 7, services=[emoped2],
                                      observer=CSVVehicleObserver(outdir+"/emoped2.csv"), prefix='em2_')
    # Add stations
    #emoped1.init_free_floating_vehicles('em1_1', 1)
    #emoped2.init_free_floating_vehicles('em2_0', 1) #TODO: connect ff emopeds

    # PT
    bus_service = PublicTransportMobilityService("BUS")
    bus_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "/bus.csv"))
    mmgraph_pt.layers["BUSLayer"].add_mobility_service(bus_service)

    tram_service = PublicTransportMobilityService("TRAM")
    tram_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "/tram.csv"))
    mmgraph_pt.layers["TRAMLayer"].add_mobility_service(tram_service)

    metro_service = PublicTransportMobilityService("METRO")
    metro_service.attach_vehicle_observer(CSVVehicleObserver(outdir + "/metro.csv"))
    mmgraph_pt.layers["METROLayer"].add_mobility_service(metro_service)


    # Create graph

    mmgraph = MultiLayerGraph([mmgraph_pt.layers["BUSLayer"], mmgraph_pt.layers["TRAMLayer"], mmgraph_pt.layers["METROLayer"],
                               emoped_layer1, emoped_layer2], odlayer, DIST_CONNECTION)

    mmgraph.connect_inter_layers(["BUSLayer", "TRAMLayer", "METROLayer"], DIST_CONNECTION)
    mmgraph.connect_intra_layer("BUSLayer", DIST_CONNECTION)
    mmgraph.connect_intra_layer("TRAMLayer", DIST_CONNECTION)
    mmgraph.connect_intra_layer("METROLayer", DIST_CONNECTION)

    # Connect PT network to emoped # TODO

    # Connect odlayer
    # mmgraph.connect_origindestination_layers(DIST_CONNECTION)

    # Demand
    demand_file = indir + "/test_all_in_highway_7h_9h.csv"          # Input demand csv
    # demand_file = indir + "/demand.csv"

    demand = CSVDemandManager(demand_file)
    demand.add_user_observer(CSVUserObserver(outdir + "/user.csv"), user_ids="all")

    # Flow
    flow_motor = MFDFlowMotor(outfile=outdir + "/flow.csv")
    flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["BUS", "TRAM", "METRO"], calculate_V_MFD))
    #flow_motor.add_reservoir(Reservoir(mmgraph.roads.zones["RES"], ["CAR"], calculate_V_MFD))

    travel_decision = DummyDecisionModel(mmgraph, outfile=outdir + "/path.csv")

    supervisor = Supervisor(graph=mmgraph,
                            flow_motor=flow_motor,
                            demand=demand,
                            decision_model=travel_decision)#outfile=outdir + "/travel_time_link.csv")

    def gc_emoped1(mlgraph, link, costs):
        gc = (0.33/60 + 20/3600) * link.length / costs['emoped1']['speed']
        return gc

    def gc_emoped2(mlgraph, link, costs):
        gc = (0.33/60 + 20/3600) * link.length / costs['emoped2']['speed']
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

    supervisor._mlgraph.add_cost_function(layer_id = 'EMOPEDLayer1', cost_name = 'gen_cost',
                                          cost_function = gc_emoped1)
    supervisor._mlgraph.add_cost_function(layer_id = 'EMOPEDLayer2', cost_name = 'gen_cost',
                                          cost_function = gc_emoped2)
    supervisor._mlgraph.add_cost_function(layer_id='METROLayer', cost_name='gen_cost',
                                          cost_function=gc_metro)
    supervisor._mlgraph.add_cost_function(layer_id='TRAMLayer', cost_name='gen_cost',
                                          cost_function=gc_tram)
    supervisor._mlgraph.add_cost_function(layer_id='BUSLayer', cost_name='gen_cost',
                                          cost_function=gc_bus)
    t1 = time.time()
    print(t1-t0, 's loading time')
    print("Start simulation")
    supervisor.run(Time('07:00:00'), Time('07:10:00'), Dt(minutes=1), 10)

    t2 = time.time()
    print(t2-t1, 's sim time')
