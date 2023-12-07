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
roads.register_stop('S1', '1_2', 0.1)
roads.register_stop('S2', '2_3', 0.1)
roads.register_stop('S3', '4_5', 0.9)

# Vehicle sharing mobility service
emoped1 = OnVehicleSharingMobilityService("emoped1", free_floating_possible=True, dt_matching=0)
emoped2 = OnVehicleSharingMobilityService("emoped2", free_floating_possible=True, dt_matching=0)

# one layer per emoped service
emoped_layer1 = SharedVehicleLayer(roads, 'emoped_layer1', Bike, 7, services=[emoped1],
                                  observer=CSVVehicleObserver("emoped1.csv"), prefix='em1_')
emoped_layer2 = SharedVehicleLayer(roads, 'emoped_layer2', Bike, 7, services=[emoped2],
                                  observer=CSVVehicleObserver("emoped2.csv"), prefix='em2_')

# Add stations
# emoped1.create_station('ES1_1', '0', 20, 5)
# emoped1.create_station('ES1_2', '4', 20, 5)
# emoped2.create_station('ES2_1', '1', 20, 5)
# emoped2.create_station('ES2_2', '3', 20, 5)

# Add free-floating vehicle
emoped1.init_free_floating_vehicles('em1_1',1)
emoped2.init_free_floating_vehicles('em2_0',1)

bus_service = PublicTransportMobilityService('Bus')
pblayer = PublicTransportLayer(roads, 'BUS', Bus, 5, services=[bus_service],
                               observer=CSVVehicleObserver("veh_bus.csv"))
pblayer.create_line("L0",
                    ["S0", "S1", "S2", "S3"],
                    [["0_1", "1_2"], ["1_2", "2_3"], ["2_3", "3_4", "4_5"]],
                    timetable=TimeTable.create_table_freq('07:00:00', '08:00:00', Dt(minutes=10)))

odlayer = generate_matching_origin_destination_layer(roads)
#
mlgraph = MultiLayerGraph([emoped_layer1, emoped_layer2, pblayer],
                          odlayer,
                          200)
print('connect layers')
mlgraph.connect_inter_layers(["emoped_layer1", "BUS"], 200)
mlgraph.connect_inter_layers(["emoped_layer2", "BUS"], 200)
# mlgraph.connect_layers("TRANSIT_LINK", "emoped_layer", "pblayer", 100, {})

# Connect od layer and layers
# mlgraph.connect_origindestination_layers(200) # already connected with mlgraph init

# Demand

print('init demand')
demand = BaseDemandManager([User("U0", [0, 0], [0, 5000], Time("07:00:30"), ['emoped1', 'emoped2']),
 User("U1", [0, 0], [0, 5000], Time("07:00:30"), ['emoped1', 'Bus']),
 User("U2", [0, 0], [0, 5000], Time("07:00:30"), ['emoped2', 'Bus'])])
demand.add_user_observer(CSVUserObserver('user.csv'))

# Decison Model

layers_groups = [({"emoped_layer1", "emoped_layer2", 'BUS'}, ({"emoped_layer1", "emoped_layer2"}, {'BUS'})),
                 ({"emoped_layer1", "emoped_layer2"}, None),
                 ({'BUS'}, None)]

#decision_model = DummyDecisionModel(mlgraph,
#                                    outfile="path.csv", cost='gen_cost',
#                                    layers_groups = layers_groups)
decision_model = DummyDecisionModel(mlgraph,
                                    outfile="path.csv", cost='gen_cost')

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
               Dt(seconds=10),
               10)

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
