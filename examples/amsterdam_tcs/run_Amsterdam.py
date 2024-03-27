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
from mnms.simulation import Supervisor
from mnms.demand import CSVDemandManager
from mnms.flow.MFD import Reservoir, MFDFlowMotor
from mnms.log import attach_log_file, LOGLEVEL, get_logger, set_all_mnms_logger_level, set_mnms_logger_level
from mnms.time import Time, Dt
from mnms.io.graph import load_graph, load_odlayer, save_odlayer
from mnms.travel_decision.logit import LogitDecisionModel
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.generation.layers import generate_layer_from_roads
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.vehicles.veh_type import Car, Bike
from mnms.graph.layers import MultiLayerGraph

##################
### Parameters ###
##################
## Parameters file
f = open('params.json')
params = json.load(f)

## Directories and files
CURRENT_DIR = str(os.path.dirname(os.path.abspath(__file__)))
INDIR = CURRENT_DIR + '/inputs/'
OUTDIR = CURRENT_DIR + '/outputs/'
LOG_FILE = OUTDIR + 'sim.log'
SERIALIZED_MLGRAPH = INDIR + params['fn_network']
SERIALIZED_ODLAYER = INDIR + params['fn_odlayer']
DEMAND_FILE = INDIR + params['fn_demand']
METROVEH_OUTFILE = OUTDIR + 'metro_veh.csv'
TRAMVEH_OUTFILE = OUTDIR + 'tram_veh.csv'
BUSVEH_OUTFILE = OUTDIR + 'bus_veh.csv'
CARVEH_OUTFILE = OUTDIR + 'car_veh.csv'
BIKEVEH_OUTFILE = OUTDIR + 'bike_veh.csv'
USERS_OUTFILE = OUTDIR + 'users.csv'
PATHS_OUTFILE = OUTDIR + "path.csv"
FLOW_OUTFILE = OUTDIR + "flow.csv"

## Outputs writing
LOG_LEVEL = LOGLEVEL.INFO
OBSERVERS = True

## Flow dynamics parameters
V_BUS = params['V_BUS'] # m/s
V_TRAM = params['V_TRAM'] # m/s
V_METRO = params['V_METRO'] # m/s
V_BIKE = params['V_BIKE'] # m/s
WALK_SPEED = 1.42 # m/s

## MultiLayerGraph creation
DIST_MAX = params['DIST_MAX'] # m
DIST_CONNECTION_OD = params['DIST_CONNECTION_OD'] # m
DIST_CONNECTION_PT = params['DIST_CONNECTION_PT'] # m

## Costs
COST_NAME = 'gen_cost'
VOT = params['VOT'] # euro/s


## Paths choices - # considered modes, packages, nb_paths
considered_modes = None
USE_BIKE = False
USE_PT = True

## Simulation parameters
START_TIME = Time('15:59:00')
END_TIME = Time('19:00:00')
DT_FLOW = Dt(seconds=60)
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

def gc_car(gnodes, layer, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['CAR']['speed']
    return gc

def gc_metro(gnodes, layer, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['METRO']['speed']
    return gc

def gc_bus(gnodes, layer, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['BUS']['speed']
    return gc

def gc_tram(gnodes, layer, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['TRAM']['speed']
    return gc

def gc_bike(gnodes, layer, link, costs, VOT=VOT):
    gc = VOT * link.length / costs['BIKE']['speed']
    return gc

def gc_transit(gnodes, layer, link, costs, p_car=0, VOT=VOT, WALK_SPEED=WALK_SPEED):
    gc = VOT * link.length / WALK_SPEED
    # add toll equivalent if taking a car
    d_node = link.downstream
    u_node = link.upstream
    if 'CAR' in d_node:
        gc += p_car
    return gc

@timed
def load_mlgraph_from_serialized_data():
    mlgraph = load_graph(SERIALIZED_MLGRAPH)
    odlayer = load_odlayer(SERIALIZED_ODLAYER)
    mlgraph.add_origin_destination_layer(odlayer)
    return mlgraph

@timed
def connect_intra_and_inter_pt_layers(mlgraph):
    mlgraph.connect_inter_layers(["BUSLayer", "TRAMLayer", "METROLayer"], DIST_CONNECTION_PT,
                                    extend_connect=True, max_connect_dist=DIST_MAX)
    mlgraph.custom_connect_intra_layer("BUSLayer", DIST_CONNECTION_PT, same_line=False)
    mlgraph.custom_connect_intra_layer("TRAMLayer", DIST_CONNECTION_PT, same_line=False)
    mlgraph.custom_connect_intra_layer("METROLayer", DIST_CONNECTION_PT, same_line=False)


def add_new_cost(mlgraph, p_car):
    mlgraph.add_cost_function('CARLayer', COST_NAME, gc_car)
    if USE_PT:
        mlgraph.add_cost_function('METROLayer', COST_NAME, gc_metro)
        mlgraph.add_cost_function('TRAMLayer', COST_NAME, gc_tram)
        mlgraph.add_cost_function('BUSLayer', COST_NAME, gc_bus)
    if USE_BIKE:
        mlgraph.add_cost_function('BIKELayer', COST_NAME, gc_bike)
    mlgraph.add_cost_function('TRANSIT', COST_NAME, lambda gnodes, layer, link, costs: gc_transit(gnodes, layer, link, costs, p_car=p_car))

def calculate_V_MFD(acc, V_BUS=V_BUS, V_TRAM=V_TRAM, V_METRO=V_METRO):
    #V_CAR = 13.9*(1-acc['CAR']/5e4)
    V_CAR = 13.9 * (1 - acc['CAR'] / 9e4)**2
    V_CAR = max(1, V_CAR)
    return {"CAR": V_CAR,"BUS": V_BUS, "TRAM": V_TRAM, "METRO": V_METRO, "BIKE": V_BIKE}

@timed
def create_supervisor(p_car):
    #### MlGraph ####
    #################
    ## Load mlgraph from serialized data, it contains roads, and PT layers and mob services
    loaded_mlgraph = load_mlgraph_from_serialized_data()
    roads = loaded_mlgraph.roads
    odlayer = loaded_mlgraph.odlayer

    ## Define OBSERVERS
    car_veh_observer = CSVVehicleObserver(CARVEH_OUTFILE) if OBSERVERS else None
    if USE_PT:
        metro_veh_observer = CSVVehicleObserver(METROVEH_OUTFILE) if OBSERVERS else None
        tram_veh_observer = CSVVehicleObserver(TRAMVEH_OUTFILE) if OBSERVERS else None
        bus_veh_observer = CSVVehicleObserver(BUSVEH_OUTFILE) if OBSERVERS else None
    if USE_BIKE:
        bike_veh_observer = CSVVehicleObserver(BIKEVEH_OUTFILE) if OBSERVERS else None

    if USE_PT:
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

    ## Car
    car = PersonalMobilityService("CAR")
    banned_nodes = [k for k in roads.nodes.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
    banned_sections = [k for k in roads.sections.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
    car_layer = generate_layer_from_roads(roads, 'CARLayer',  veh_type = Car, default_speed = 11, mobility_services = [car],
                                          banned_nodes=banned_nodes, banned_sections=banned_sections)
    car.attach_vehicle_observer(car_veh_observer)

    if USE_BIKE:
        ## Bike
        bike = PersonalMobilityService("BIKE")
        banned_nodes = [k for k in roads.nodes.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
        banned_sections = [k for k in roads.sections.keys() if ('TRAM' in k or 'BUS' in k or 'METRO' in k)]
        bike_layer = generate_layer_from_roads(roads, 'BIKELayer', veh_type = Bike, default_speed = V_BIKE, mobility_services = [bike],
                                              banned_nodes=banned_nodes, banned_sections=banned_sections)
        bike.attach_vehicle_observer(bike_veh_observer)

    ## MLGraph with all layers, including odlayer, do the connections between ODLayer and other layers directly
    if USE_PT and USE_BIKE:
        mlgraph = MultiLayerGraph([bus_layer, tram_layer, metro_layer, car_layer, bike_layer],
            odlayer, DIST_CONNECTION_OD)
    elif USE_PT:
        mlgraph = MultiLayerGraph([bus_layer, tram_layer, metro_layer, car_layer], odlayer, DIST_CONNECTION_OD)
    else:
        mlgraph = MultiLayerGraph([car_layer], odlayer, DIST_CONNECTION_OD)

    if USE_PT:
        # Add the transit links intra PT layers
        connect_intra_and_inter_pt_layers(mlgraph)

    # Add new costs
    add_new_cost(mlgraph, p_car)

    #### Demand ####
    ################
    demand = CSVDemandManager(DEMAND_FILE)
    demand.add_user_observer(CSVUserObserver(USERS_OUTFILE))

    #### Decison Model ####
    #######################
    travel_decision = LogitDecisionModel(mlgraph, considered_modes=considered_modes, outfile=PATHS_OUTFILE, cost=COST_NAME)
    travel_decision.add_waiting_cost_function(COST_NAME, lambda wt: VOT*wt)

    #### Flow motor ####
    ####################
    flow_motor = MFDFlowMotor(outfile=FLOW_OUTFILE)
    if USE_PT and USE_BIKE:
        flow_motor.add_reservoir(Reservoir(mlgraph.roads.zones["RES"], ["CAR", "BUS", "TRAM", "METRO", "BIKE"], calculate_V_MFD))
    elif USE_PT:
        flow_motor.add_reservoir(
            Reservoir(mlgraph.roads.zones["RES"], ["CAR", "BUS", "TRAM", "METRO"], calculate_V_MFD))
    else:
        flow_motor.add_reservoir(
            Reservoir(mlgraph.roads.zones["RES"], ["CAR"], calculate_V_MFD))

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
    p_car = 0
    supervisor = create_supervisor(p_car)
    run_simulation(supervisor)
