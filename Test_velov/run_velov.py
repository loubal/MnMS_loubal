import pathlib
from mnms.generation.roads import generate_manhattan_road
from mnms.mobility_service.vehicle_sharing import OnVehicleSharingMobilityService
from mnms.tools.observer import CSVVehicleObserver, CSVUserObserver
from mnms.generation.layers import generate_layer_from_roads, generate_grid_origin_destination_layer
from mnms.graph.layers import MultiLayerGraph, SharedVehicleLayer
from mnms.vehicles.veh_type import Bike
from mnms.generation.demand import generate_random_demand
from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.flow.MFD import MFDFlowMotor, Reservoir
from mnms.simulation import Supervisor
from mnms.demand import CSVDemandManager, BaseDemandManager, User
from mnms.time import TimeTable, Time, Dt
from mnms.log import set_mnms_logger_level, LOGLEVEL, attach_log_file
from mnms.io.graph import save_graph, load_graph

# set_all_mnms_logger_level(LOGLEVEL.WARNING)
set_mnms_logger_level(LOGLEVEL.INFO, ["mnms.simulation"])

# get_logger("mnms.graph.shortest_path").setLevel(LOGLEVEL.WARNING)
attach_log_file('simulation.log')

# Graph
road_db = generate_manhattan_road(3, 1000)

# Vehicle sharing mobility service
velov = OnVehicleSharingMobilityService("velov", 0)

velov_layer = SharedVehicleLayer(road_db, 'velov_layer', Bike, 3, services=[velov], observer=CSVVehicleObserver("velov.csv"))

# OD layer
odlayer = generate_grid_origin_destination_layer(0, 0, 3000, 3000, 3, 3)

# Multilayer graph
mlgraph = MultiLayerGraph([velov_layer],odlayer)

# Add stations
velov.create_station('S1','0',20,2)
velov.create_station('S2','8',20,1)

# Connect od layer and velov layer
mlgraph.connect_origindestination_layers(10)

# Desicion model
decision_model = DummyDecisionModel(mlgraph, outfile="path.csv")

# Flow Motor
def mfdspeed(dacc):
    dspeed = {'BIKE': 10}
    return dspeed

flow_motor = MFDFlowMotor(outfile="flow.csv")
flow_motor.add_reservoir(Reservoir(road_db.zones["RES"], ['BIKE'], mfdspeed))

#cwd = pathlib.Path(__file__).parent.joinpath('demand.csv').resolve()
#demand = CSVDemandManager(cwd)
demand = BaseDemandManager([User("U0", [0, 0], [2000, 2000], Time("07:00:00"), ['velov'])])
demand.add_user_observer(CSVUserObserver('user.csv'))

supervisor = Supervisor(mlgraph,
                         demand,
                         flow_motor,
                         decision_model)

supervisor.run(Time("07:00:00"),
                Time("08:00:00"),
                Dt(minutes=1),
                10)

from matplotlib import pyplot as plt

roads = road_db
plt.figure()
for id in roads.nodes:
    node = roads.nodes[id]
    plt.plot(node.position[0], node.position[1], '+b')

for id in roads.stops:
    stop = roads.stops[id]
    plt.plot(stop.absolute_position[0], stop.absolute_position[1], 'ro')

for id in velov.stations:
    station = velov.stations[id]
    node = roads.nodes[station.node]
    plt.plot(node.position[0], node.position[1], 'sg')

plt.show()
