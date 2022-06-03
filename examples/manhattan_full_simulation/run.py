import pathlib

from mnms.graph.generation import (generate_manhattan_road, generate_layer_from_roads,
                                   generate_grid_origin_destination_layer)
from mnms.graph.layers import MultiLayerGraph
from mnms.demand.manager import CSVDemandManager
from mnms.log import set_mnms_logger_level, LOGLEVEL
from mnms.travel_decision.dummy import DummyDecisionModel
from mnms.mobility_service.car import PersonalCarMobilityService
from mnms.flow.MFD import MFDFlow, Reservoir
from mnms.simulation import Supervisor
from mnms.time import Time, Dt
from mnms.tools.observer import CSVUserObserver


set_mnms_logger_level(LOGLEVEL.INFO, ['mnms.simulation',
                                      'mnms.vehicles.veh_type',
                                      'mnms.flow.user_flow',
                                      'mnms.flow.MFD',
                                      'mnms.layer.public_transport',
                                      'mnms.travel_decision.model',
                                      'mnms.tools.observer'])

cwd = pathlib.Path(__file__).parent.joinpath('demand.csv').resolve()

# Graph

road_db = generate_manhattan_road(10, 100)
car_layer = generate_layer_from_roads(road_db,
                                      'CAR',
                                      mobility_services=[PersonalCarMobilityService()])

odlayer = generate_grid_origin_destination_layer(0, 0, 1000, 1000, 10, 10)

mlgraph = MultiLayerGraph([car_layer],
                          odlayer,
                          1e-3)

# Demand

demand = CSVDemandManager(cwd, demand_type='coordinate')
demand.add_user_observer(CSVUserObserver('user.csv'))

# Decison Model

decision_model = DummyDecisionModel(mlgraph)

# Flow Motor

def mfdspeed(dacc):
    dacc['CAR'] = 3
    return dacc

flow_motor = MFDFlow()
flow_motor.add_reservoir(Reservoir('RES', 'CAR', mfdspeed))

supervisor = Supervisor(mlgraph,
                        demand,
                        flow_motor,
                        decision_model)

supervisor.run(Time("07:00:00"),
               Time("08:00:00"),
               Dt(seconds=10),
               10)