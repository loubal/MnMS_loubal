from mnms import LOGLEVEL
from mnms.demand import BaseDemandManager, User
from mnms.generation.roads import generate_line_road
from mnms.generation.layers import generate_layer_from_roads, generate_grid_origin_destination_layer, \
    generate_matching_origin_destination_layer
from mnms.graph.layers import MultiLayerGraph, PublicTransportLayer, CarLayer, SharedVehicleLayer
from mnms.log import set_mnms_logger_level, attach_log_file
from mnms.mobility_service.public_transport import PublicTransportMobilityService
from mnms.travel_decision import LogitDecisionModel

from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.flow.MFD import MFDFlowMotor, Reservoir
from mnms.simulation import Supervisor
from mnms.time import Time, Dt, TimeTable
from mnms.tools.observer import CSVUserObserver, CSVVehicleObserver
from mnms.vehicles.veh_type import Bus, Bike
from mnms.mobility_service.vehicle_sharing import OnVehicleSharingMobilityService

set_mnms_logger_level(LOGLEVEL.INFO, ['mnms.simulation',
                                      'mnms.vehicles.veh_type',
                                      'mnms.flow.user_flow',
                                      'mnms.flow.MFD',
                                      'mnms.layer.public_transport',
                                      'mnms.travel_decision.model',
                                      'mnms.tools.observer'])
attach_log_file('simulation.log')

roads = generate_line_road([0, 0], [0, 5000], 6)
roads.register_stop('S0', '0_1', 0.1)
roads.register_stop('S1', '3_4', 0.9)
roads.register_stop('S2', '4_5', 0.9)

# Vehicle sharing mobility service
emoped = OnVehicleSharingMobilityService("emoped", dt_matching=0)

emoped_layer = SharedVehicleLayer(roads, 'emoped_layer', Bike, 7, services=[emoped],
                                  observer=CSVVehicleObserver("emoped.csv"))

# Add stations
emoped.create_station('ES1', '0', 20, 5)
emoped.create_station('ES2', '4', 20, 5)

bus_service = PublicTransportMobilityService('Bus')
pblayer = PublicTransportLayer(roads, 'BUS', Bus, 5, services=[bus_service],
                               observer=CSVVehicleObserver("veh_bus.csv"))
pblayer.create_line("L0",
                    ["S0", "S1", "S2"],
                    [["0_1", "1_2", "2_3", "3_4"], ["3_4", "4_5"]],
                    timetable=TimeTable.create_table_freq('07:00:00', '08:00:00', Dt(minutes=10)))

odlayer = generate_matching_origin_destination_layer(roads)
#
mlgraph = MultiLayerGraph([emoped_layer, pblayer],
                          odlayer,
                          200)
print('connect layers')
mlgraph.connect_inter_layers(["emoped_layer", "BUS"], 200)
# mlgraph.connect_layers("TRANSIT_LINK", "emoped_layer", "pblayer", 100, {})

# Connect od layer and layers
mlgraph.connect_origindestination_layers(200)

# Demand

print('init demand')
demand = BaseDemandManager([User("U0", [0, 0], [0, 5000], Time("07:00:30"), ['Bus','emoped']),
                            User("U1", [0, 0], [0, 5000], Time("07:00:30"), ['Bus']),
                            User("U2", [0, 0], [0, 5000], Time("07:00:30"))])
demand.add_user_observer(CSVUserObserver('user.csv'))

# Decison Model

# decision_model = LogitDecisionModel(mlgraph, outfile="path.csv")
decision_model = DummyDecisionModel(mlgraph, outfile="path.csv")

# Flow Motor

def mfdspeed(dacc):
    dspeed = {'BUS': 5,
              'BIKE': 7}
    return dspeed

flow_motor = MFDFlowMotor('flow.csv')
flow_motor.add_reservoir(Reservoir(roads.zones["RES"], ['BUS', 'BIKE'], mfdspeed))

supervisor = Supervisor(mlgraph,
                        demand,
                        flow_motor,
                        decision_model)

supervisor.run(Time("07:00:00"),
               Time("07:50:00"),
               Dt(seconds=10),
               10)

from matplotlib import pyplot as plt

plt.figure()
for id in roads.nodes:
    node = roads.nodes[id]
    plt.plot(node.position[0], node.position[1], '+b')

for id in roads.stops:
    stop = roads.stops[id]
    plt.plot(stop.absolute_position[0], stop.absolute_position[1], 'ro')

for id in emoped.stations:
    station = emoped.stations[id]
    node = roads.nodes[station.node]
    plt.plot(node.position[0], node.position[1], 'sg')

plt.show()
