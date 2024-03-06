###############
### Imports ###
###############
## Casuals
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import time
import json
import sys

## MnMS & HiPOP
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
from mnms.travel_decision.custom_decision import CustomDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_layer_from_roads
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.io.graph import save_transit_link_odlayer, load_transit_links
from mnms.vehicles.veh_type import Bike
from mnms.graph.layers import MultiLayerGraph
from mnms.tools.render import draw_roads

##################
### Parameters ###
##################
## Parameters file
f = open('params.json')
params = json.load(f)

## Mobility service parameters
USE_EMOPED = True

## Directories and files
CURRENT_DIR = str(os.path.dirname(os.path.abspath(__file__)))
INDIR = CURRENT_DIR + '/inputs/'
OUTDIR =  CURRENT_DIR + '/outputs/'
LOG_FILE = OUTDIR + 'sim.log'
SERIALIZED_MLGRAPH = INDIR + params['fn_network']
SERIALIZED_ODLAYER = INDIR + params['fn_odlayer']
STATION_EMOPED = INDIR + params['fn_emoped_st_init']
DEMAND_FILE = INDIR + params['fn_demand']
METROVEH_OUTFILE = OUTDIR + 'metro_veh.csv'
TRAMVEH_OUTFILE = OUTDIR + 'tram_veh.csv'
BUSVEH_OUTFILE = OUTDIR + 'bus_veh.csv'
EMOPED1VEH_OUTFILE = OUTDIR + 'emoped1_veh.csv'
USERS_OUTFILE = OUTDIR + 'users.csv'
PATHS_OUTFILE = OUTDIR + "path.csv"
FLOW_OUTFILE = OUTDIR + "flow.csv"

## Outputs writing
LOG_LEVEL = LOGLEVEL.INFO
OBSERVERS = True

## Flow dynamics parameters
V_EMOPED = params['V_EMOPED'] # m/s
V_BUS = params['V_BUS'] # m/s
V_TRAM = params['V_TRAM'] # m/s
V_METRO = params['V_METRO'] # m/s
WALK_SPEED = 1.42 # m/s

## MultiLayerGraph creation
DIST_MAX = params['DIST_MAX'] # m
DIST_CONNECTION_OD = params['DIST_CONNECTION_OD'] # m
DIST_CONNECTION_PT = params['DIST_CONNECTION_PT'] # m
DIST_CONNECTION_MIX = params['DIST_CONNECTION_MIX'] # m

## EMOPED operation
EMOPED_DT_MATCHING = 0
EMOPED_DT_REBALANCING = 10
STATION_CAPACITY = 100
PENALTY_NO_EMOPED = 1800 #s

## Costs
COST_NAME = 'gen_cost'
VOT = params['VOT'] # euro/s
FEE_MOPED_TIME = params['FEE_EMOPED_TIME'] # euro/s

## Paths choices - # considered modes, packages, nb_paths
if USE_EMOPED:
    considered_modes = [({'BUSLayer', 'TRAMLayer', 'METROLayer'}, None, 1),
                        ({'EMOPEDLayer1'},None,1),
                        ({'BUSLayer', 'TRAMLayer', 'METROLayer', 'EMOPEDLayer1'}, ({'EMOPEDLayer1'}, {'BUSLayer', 'TRAMLayer', 'METROLayer'}), 1)]
else:
    considered_modes = None
#considered_modes = None

## Simulation parameters
START_TIME = Time('15:59:00')
END_TIME = Time('19:00:00')
DT_FLOW = Dt(minutes=1)
AFFECTION_FACTOR = 1

#################
### Functions ###
#################
def timed(func):
    """Decorator to measure the execution time of a function.
    """
    def decorator(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'Execution of {func.__name__} function took {t2-t1:.2f} seconds')
        return result
    return decorator

def gc_emoped1(mlgraph, link, costs, FEE_MOPED_TIME=FEE_MOPED_TIME, VOT=VOT):
    gc = (FEE_MOPED_TIME + VOT) * link.length / costs['emoped1']['speed']
    return gc

def gc_metro(mlgraph, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['METRO']['speed']
    return gc

def gc_bus(mlgraph, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['BUS']['speed']
    return gc

def gc_tram(mlgraph, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['TRAM']['speed']
    return gc

def gc_transit(mlgraph, link, costs, VOT=VOT, WALK_SPEED=WALK_SPEED):
    gc = VOT * link.length / WALK_SPEED
    d_node = link.downstream
    stations = mlgraph.layers['EMOPEDLayer1'].stations
    station = None
    for s in stations:
        if s['node'] == d_node:
            station = s['id']
            break
    if station:
        mob_service = mlgraph.layers['EMOPEDLayer1'].mobility_services['emoped1']
        nb_veh = len(mob_service.available_vehicles(station))
        if nb_veh == 0:
            gc += VOT*PENALTY_NO_EMOPED
    return gc

@timed
def load_mlgraph_from_serialized_data():
    mlgraph = load_graph(SERIALIZED_MLGRAPH)
    odlayer = load_odlayer(SERIALIZED_ODLAYER)
    mlgraph.add_origin_destination_layer(odlayer)
    return mlgraph

@timed
def connect_intra_and_inter_pt_layers(mlgraph):
    mlgraph.custom_connect_inter_layers(["BUSLayer", "TRAMLayer", "METROLayer"], DIST_CONNECTION_PT,
                                    ensure_connect=True, max_connect_dist=DIST_MAX)
    mlgraph.custom_connect_intra_layer("BUSLayer", DIST_CONNECTION_PT, same_line=False)
    mlgraph.custom_connect_intra_layer("TRAMLayer", DIST_CONNECTION_PT, same_line=False)
    mlgraph.custom_connect_intra_layer("METROLayer", DIST_CONNECTION_PT, same_line=False)

@timed
def connect_pt_with_emoped_layers(mlgraph):
    if USE_EMOPED:
        mlgraph.custom_connect_inter_layers(["EMOPEDLayer1", "BUSLayer"], DIST_CONNECTION_MIX,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
        mlgraph.custom_connect_inter_layers(["EMOPEDLayer1", "TRAMLayer"], DIST_CONNECTION_MIX,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
        mlgraph.custom_connect_inter_layers(["EMOPEDLayer1", "METROLayer"], DIST_CONNECTION_MIX,
                                            ensure_connect=True, max_connect_dist=DIST_MAX)
    else:
        pass

def add_new_cost(mlgraph):
    if USE_EMOPED:
        mlgraph.add_cost_function('EMOPEDLayer1', COST_NAME, gc_emoped1)
    mlgraph.add_cost_function('METROLayer', COST_NAME, gc_metro)
    mlgraph.add_cost_function('TRAMLayer', COST_NAME, gc_tram)
    mlgraph.add_cost_function('BUSLayer', COST_NAME, gc_bus)
    mlgraph.add_cost_function('TRANSIT', COST_NAME, gc_transit)

def calculate_V_MFD(acc, V_EMOPED=V_EMOPED, V_BUS=V_BUS, V_TRAM=V_TRAM, V_METRO=V_METRO):
    return {"BIKE": V_EMOPED,"BUS": V_BUS, "TRAM": V_TRAM, "METRO": V_METRO}

@timed
def create_supervisor():
    #### MlGraph ####
    #################
    ## Load mlgraph from serialized data, it contains roads, and PT layers and mob services
    loaded_mlgraph = load_mlgraph_from_serialized_data()
    roads = loaded_mlgraph.roads
    odlayer = loaded_mlgraph.odlayer

    ## Define OBSERVERS
    metro_veh_observer = CSVVehicleObserver(METROVEH_OUTFILE) if OBSERVERS else None
    tram_veh_observer = CSVVehicleObserver(TRAMVEH_OUTFILE) if OBSERVERS else None
    bus_veh_observer = CSVVehicleObserver(BUSVEH_OUTFILE) if OBSERVERS else None
    if USE_EMOPED:
        emoped1_veh_observer = CSVVehicleObserver(EMOPED1VEH_OUTFILE) if OBSERVERS else None

    ## Metro
    metro_layer = loaded_mlgraph.layers['METROLayer']
    metro_service = metro_layer.mobility_services['METRO']
    metro_service.attach_vehicle_observer(metro_veh_observer)

    ## Tram
    tram_layer = loaded_mlgraph.layers['TRAMLayer']
    tram_service = tram_layer.mobility_services['TRAM']
    tram_service.attach_vehicle_observer(tram_veh_observer)

    ## Bus
    bus_layer = loaded_mlgraph.layers['BUSLayer']
    bus_service = bus_layer.mobility_services['BUS']
    bus_service.attach_vehicle_observer(bus_veh_observer)

    ## EMOPED company 1
    if USE_EMOPED:
        emoped1 = VehicleSharingMobilityService("emoped1", free_floating_possible=False, dt_matching=EMOPED_DT_MATCHING,
                                                dt_periodic_maintenance=EMOPED_DT_REBALANCING)

        banned_nodes = [k for k in roads.nodes.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
        banned_sections = [k for k in roads.sections.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
        emoped_layer1 = generate_layer_from_roads(roads, 'EMOPEDLayer1', SharedVehicleLayer, Bike, V_EMOPED,
                                                  [emoped1], banned_nodes=banned_nodes, banned_sections=banned_sections)
        emoped1.attach_vehicle_observer(emoped1_veh_observer)
        emoped_layer1.add_connected_layers(["BUSLayer", "TRAMLayer", "METROLayer"])

    ## Add emoped stations
    if USE_EMOPED:
        df_sta_emoped = pd.read_csv(STATION_EMOPED)
        for i, row in df_sta_emoped.iterrows():
            node = row['closest_node']
            nb_emoped = row['nb_emoped']
            emoped1.create_station(id_station='station'+str(i), dbroads_node=node,
                                   nb_initial_veh=nb_emoped, free_floating=False, capacity=STATION_CAPACITY)

    ## MLGraph with all layers, including odlayer, do the connections between ODLayer and other layers directly
    if USE_EMOPED:
        mlgraph = MultiLayerGraph([bus_layer, tram_layer, metro_layer, emoped_layer1],
            odlayer, DIST_CONNECTION_OD)
    else:
        mlgraph = MultiLayerGraph([bus_layer, tram_layer, metro_layer],
            odlayer, DIST_CONNECTION_OD)

    # Add the transit links intra and inter layers
    connect_intra_and_inter_pt_layers(mlgraph)
    connect_pt_with_emoped_layers(mlgraph)

    # gather transit links ending in emoped stations
    names_st_no = ['EMOPEDLayer1_'+cl_no for cl_no in df_sta_emoped['closest_node'].values]
    transit_per_end_node_station = dict()
    for cl_no in names_st_no:
        transit_per_end_node_station[cl_no] = []
    for layer_id in ['ODLayer','BUSLayer','TRAMLayer','METROLayer']:
        for l_id in mlgraph.transitlayer.links[layer_id]['EMOPEDLayer1']:
            #if l_id in mlgraph.graph.links.keys(): # some links removed after PT creation
            link = mlgraph.graph.links[l_id]
            if link.downstream in names_st_no:
                transit_per_end_node_station[link.downstream].append(l_id)
    emoped1.transit_per_end_node_station = transit_per_end_node_station

    # Add new cost
    add_new_cost(mlgraph)

    #### Demand ####
    ################
    demand = CSVDemandManager(DEMAND_FILE)
    demand.add_user_observer(CSVUserObserver(USERS_OUTFILE))

    #### Decison Model ####
    #######################
    travel_decision = CustomDecisionModel(mlgraph, considered_modes=considered_modes, outfile=PATHS_OUTFILE, cost=COST_NAME)
    travel_decision.add_waiting_cost_function(COST_NAME, lambda wt: VOT*wt)

    #### Flow motor ####
    ####################
    flow_motor = MFDFlowMotor(outfile=FLOW_OUTFILE)
    if USE_EMOPED:
        flow_motor.add_reservoir(Reservoir(mlgraph.roads.zones["RES"], ["BIKE", "BUS", "TRAM", "METRO"], calculate_V_MFD))
    else:
        flow_motor.add_reservoir(Reservoir(mlgraph.roads.zones["RES"], ["BUS", "TRAM", "METRO"], calculate_V_MFD))

    #### Supervisor ####
    ####################
    supervisor = Supervisor(mlgraph,
                            demand,
                            flow_motor,
                            travel_decision,
                            logfile=LOG_FILE,
                            loglevel=LOG_LEVEL)

    return supervisor

@timed
def run_simulation(supervisor):
    set_all_mnms_logger_level(LOG_LEVEL)
    supervisor.run(START_TIME, END_TIME, DT_FLOW, AFFECTION_FACTOR)

############
### Main ###
############
if __name__ == '__main__':
    supervisor = create_supervisor()
    run_simulation(supervisor)
