from mnms import LOGLEVEL
from mnms.demand import BaseDemandManager, User
from mnms.generation.roads import generate_line_road
from mnms.generation.layers import generate_layer_from_roads, generate_grid_origin_destination_layer, \
    generate_matching_origin_destination_layer
from mnms.graph.layers import PublicTransportLayer, CarLayer, SharedVehicleLayer, MultiLayerGraph
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
from mnms.mobility_service.vehicle_sharing import VehicleSharingMobilityService

set_mnms_logger_level(LOGLEVEL.INFO, ['mnms.simulation',
                                      'mnms.vehicles.veh_type',
                                      'mnms.flow.user_flow',
                                      'mnms.flow.MFD',
                                      'mnms.layer.public_transport',
                                      'mnms.travel_decision.model',
                                      'mnms.tools.observer'])
attach_log_file('simulation.log')

roads = generate_line_road([0, 0], [0, 2000], 3)
roads.register_stop('S0', '0_1', 0.1)
roads.register_stop('S1', '1_2', 0.01)
roads.register_stop('S2', '1_2', 0.9)

roads.register_stop('S0R', '1_0', 0.9)
roads.register_stop('S1R', '2_1', 0.99)
roads.register_stop('S2R', '2_1', 0.1)

# Vehicle sharing mobility service
emoped1 = VehicleSharingMobilityService("emoped1", free_floating_possible=True, dt_matching=0)
emoped2 = VehicleSharingMobilityService("emoped2", free_floating_possible=True, dt_matching=0)
emoped1.attach_vehicle_observer(CSVVehicleObserver("emoped1.csv"))
emoped2.attach_vehicle_observer(CSVVehicleObserver("emoped2.csv"))

# one layer per emoped service
emoped_layer1 = generate_layer_from_roads(roads, 'emoped_layer1', SharedVehicleLayer, Bike, 7, [emoped1])
emoped_layer2 = generate_layer_from_roads(roads, 'emoped_layer2', SharedVehicleLayer, Bike, 7, [emoped2])

bus_service = PublicTransportMobilityService('Bus')
pblayer = PublicTransportLayer(roads, 'BUS', Bus, 5, services=[bus_service],
                               observer=CSVVehicleObserver("veh_bus.csv"))
pblayer.create_line("L0",
                    ["S0", "S1", "S2"],
                    [["0_1", "1_2"], ["1_2"]],
                    timetable=TimeTable.create_table_freq('07:00:00', '08:00:00', Dt(minutes=10)))
pblayer.create_line("L0R",
                    ["S2R", "S1R", "S0R"],
                    [["2_1"], ["2_1", "1_0"]],
                    timetable=TimeTable.create_table_freq('07:00:00', '08:00:00', Dt(minutes=10)))

odlayer = generate_matching_origin_destination_layer(roads, with_stops=False)

# Add free-floating vehicle
emoped1.init_free_floating_vehicles('2',1)
emoped2.init_free_floating_vehicles('0',1)
emoped2.init_free_floating_vehicles('1',1)

mlgraph = MultiLayerGraph([emoped_layer1, emoped_layer2, pblayer],
                          odlayer,
                          200)
# Connect layers
mlgraph.connect_inter_layers(["emoped_layer1", "BUS"], 200)
mlgraph.connect_inter_layers(["emoped_layer2", "BUS"], 200)
emoped_layer1.add_connected_layers(["BUS"])
emoped_layer2.add_connected_layers(["BUS"])

# Demand

print('init demand')
demand = BaseDemandManager([User("U0", [0, 0], [0, 2000], Time("07:00:30"), ['Bus', 'emoped1', 'emoped2']),
 #User("U0bis", [0, 0], [0, 2000], Time("07:01:00"), ['Bus', 'emoped1', 'emoped2']),
 User("U1", [0, 2000], [0, 1000], Time("07:00:30"), ['emoped1', 'Bus']),
 User("U2", [0, 0], [0, 2000], Time("07:10:00"), ['emoped2', 'Bus']),
 User("U3", [0, 2000], [0, 0], Time("07:30:00"), ['emoped1', 'Bus'])])
demand.add_user_observer(CSVUserObserver('user.csv'))

# Decison Model

layers_groups = [({"emoped_layer1", "emoped_layer2", 'BUS'}, ({"emoped_layer1", "emoped_layer2"}, {'BUS'})),
                 ({"emoped_layer1", "emoped_layer2"}, None),
                 ({'BUS'}, None)]

#decision_model = DummyDecisionModel(mlgraph,
#                                    outfile="path.csv", cost='gen_cost',
#                                    layers_groups = layers_groups)
decision_model = DummyDecisionModel(mlgraph,
                                    outfile="path.csv")#, cost='gen_cost')

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


# Define link based costs
def gc_emoped1(mlgraph, link, costs):
    gc = (0*0.33 / 60 + 20 / 3600) * link.length / costs['emoped1']['speed']
    return gc

def gc_emoped2(mlgraph, link, costs):
    gc = (0*0.33 / 60 + 20 / 3600) * link.length / costs['emoped2']['speed']
    return gc

def gc_bus(mlgraph, link, costs):
    gc = 0*4 + 20 / 3600 * link.length / costs['Bus']['speed']
    return gc


supervisor._mlgraph.add_cost_function(layer_id='emoped_layer1', cost_name='gen_cost',
                                      cost_function=gc_emoped1)
supervisor._mlgraph.add_cost_function(layer_id='emoped_layer2', cost_name='gen_cost',
                                      cost_function=gc_emoped2)
supervisor._mlgraph.add_cost_function(layer_id='BUS', cost_name='gen_cost',
                                      cost_function=gc_bus)


# Add link independent service cost

def gc_additional(path):
    final_cost = path.path_cost
    if 'emoped1' in path.mobility_services:
        final_cost += 1
    if 'emoped2' in path.mobility_services:
        final_cost += 1
    if 'Bus' in path.mobility_services:
        final_cost += 2
    return final_cost

# supervisor._mlgraph.add_additional_cost_function('gen_cost', gc_additional) # TODO: add additional cost

supervisor.run(Time("07:00:00"),
               Time("08:00:00"),
               Dt(seconds=30),
               3)

#print(mlgraph.transitlayer.links['ODLAYER']['emoped_layer2'])
'''for key in mlgraph.graph.links.keys():
    if 'ORIGIN' in key and 'emoped_layer1' in key:
        print(key)
print('='*10)
for key in mlgraph.graph.links.keys():
    if 'DESTINATION' in key and 'emoped_layer1' in key:
        print(key)
print('='*10)
for key in mlgraph.graph.links.keys():
    if 'ORIGIN' in key and 'emoped_layer2' in key:
        print(key)
print('='*10)
for key in mlgraph.graph.links.keys():
    if 'DESTINATION' in key and 'emoped_layer2' in key:
        print(key)'''

for lid in mlgraph.transitlayer.links['ODLAYER']['emoped_layer1']:
    if 'ORIGIN' in lid:
        print(lid, mlgraph.graph.links[lid].length)
print('='*10)
for lid in mlgraph.transitlayer.links['BUS']['emoped_layer1']:
    print(lid, mlgraph.graph.links[lid].length)
print('='*10)
for lid in mlgraph.transitlayer.links['ODLAYER']['emoped_layer2']:
    if 'ORIGIN' in lid:
        print(lid, mlgraph.graph.links[lid].length)
print('='*10)
for lid in mlgraph.transitlayer.links['BUS']['emoped_layer2']:
    print(lid, mlgraph.graph.links[lid].length)


###  Plot
if False:
    from matplotlib import pyplot as plt

    plt.figure()
    for id in roads.nodes:
        node = roads.nodes[id]
        plt.plot(node.position[0], node.position[1], '+b')

    for id in roads.stops:
        stop = roads.stops[id]
        plt.plot(stop.absolute_position[0], stop.absolute_position[1], 'ro')

    for id in emoped1.stations:
        station = emoped1.stations[id]
        node = roads.nodes[station.node]
        plt.plot(node.position[0], node.position[1], 'xg')

    for id in emoped2.stations:
        station = emoped2.stations[id]
        node = roads.nodes[station.node]
        plt.plot(node.position[0], node.position[1], 'xm')

    plt.show()
