import pathlib
from mnms.generation.roads import generate_manhattan_road
from mnms.mobility_service.vehicle_sharing import OnVehicleSharingMobilityService
from mnms.tools.observer import CSVVehicleObserver, CSVUserObserver
from mnms.generation.layers import generate_layer_from_roads, generate_grid_origin_destination_layer
from mnms.graph.layers import SharedVehicleLayer, MultiLayerGraph
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
road_db = generate_manhattan_road(5, 1000, prefix='I_')

# Vehicle sharing mobility service
ff_velov1 = OnVehicleSharingMobilityService("ff_velov1", 1, 0)
ff_velov2 = OnVehicleSharingMobilityService("ff_velov2", 1, 0)

print('Generate layers')

velov_layer1 = SharedVehicleLayer(road_db, 'velov_layer1', Bike, 3, services=[ff_velov1], observer=CSVVehicleObserver("velov1.csv"), prefix='vl1')
velov_layer2 = SharedVehicleLayer(road_db, 'velov_layer2', Bike, 3, services=[ff_velov2], observer=CSVVehicleObserver("velov2.csv"), prefix='vl2')



# OD layer
#odlayer = generate_grid_origin_destination_layer(-1000, -1000, 6000, 6000, 10, 10)
odlayer = generate_grid_origin_destination_layer(-1000, -1000, 3000, 3000, 5, 5)

# Multilayer graph
mlgraph = MultiLayerGraph([velov_layer1, velov_layer2],odlayer)

# Add free-floating vehicle
ff_velov1.init_free_floating_vehicles('vl1I_2',1)
ff_velov2.init_free_floating_vehicles('vl2I_2',1)

# Connect od layer and velov layer
mlgraph.connect_origindestination_layers(500)

#save_graph(mlgraph, 'free_floating_example.json')

# Desicion model
decision_model = DummyDecisionModel(mlgraph, outfile="path.csv")

# Flow Motor
def mfdspeed(dacc):
    dspeed = {'BIKE': 3}
    return dspeed

flow_motor = MFDFlowMotor(outfile="flow.csv")
flow_motor.add_reservoir(Reservoir(road_db.zones["RES"], ['BIKE'], mfdspeed))

#cwd = pathlib.Path(__file__).parent.joinpath('demand2.csv').resolve()
#demand = CSVDemandManager(cwd)
demand = BaseDemandManager([User("U0", "ORIGIN_21", "DESTINATION_4", Time("07:00:00"), ['ff_velov1']),
                            User("U1", "ORIGIN_21", "DESTINATION_4", Time("07:15:00"), ['ff_velov2']),
                            User("U2", "ORIGIN_4", "DESTINATION_21", Time("07:40:00"), ['ff_velov1'])])

demand.add_user_observer(CSVUserObserver('user.csv'))

supervisor = Supervisor(mlgraph,
                         demand,
                         flow_motor,
                         decision_model)

supervisor.run(Time("07:00:00"),
                Time("09:00:00"),
                Dt(minutes=1),
                1)


