import sys
from abc import ABC, abstractmethod
from typing import List, Set, Dict
import csv
import multiprocessing

import numpy as np
from numpy.linalg import norm as _norm

from mnms.demand.user import User, Path
from mnms.graph.multilayer_graph import MultiLayerGraph
from mnms.mobility_service.personal_vehicle import PersonalMobilityService
from mnms.log import create_logger
from mnms.time import Time
from mnms.tools.dict_tools import sum_dict
from mnms.tools.exceptions import PathNotFound

from hipop.shortest_path import parallel_k_shortest_path, dijkstra, compute_path_length, parallel_intermodal_dijkstra

log = create_logger(__name__)


def _process_shortest_path_inputs(mlgraph: MultiLayerGraph, users):
    odlayer = mlgraph.odlayer
    origins = [None] * len(users)
    destinations = [None] * len(users)
    available_layers = [None] * len(users)
    chosen_mservice = [None] * len(users)

    origins_id = list(odlayer.origins.keys())
    origins_pos = np.array([position for position in odlayer.origins.values()])
    destinations_id = list(odlayer.destinations.keys())
    destinations_pos = np.array([position for position in odlayer.destinations.values()])

    for i, u in enumerate(users):
        if isinstance(u.origin, np.ndarray):
            origins[i] = origins_id[np.argmin(_norm(origins_pos - u.origin, axis=1))]
            destinations[i] = destinations_id[np.argmin(_norm(destinations_pos - u.destination, axis=1))]
        else:
            origins[i] = u.origin
            destinations[i] = u.destination

        available_layers[i] = set() if u.available_mobility_service is None else {layer.id for mservice in u.available_mobility_service for layer in mlgraph.layers.values()  if mservice in layer.mobility_services}

        # if available_layers[i]:
        #     available_layers[i].add("TRANSIT")

        services = {}
        if available_layers[i]:
            for layer in available_layers[i]:
                services[layer] = [ms for ms in u.available_mobility_service \
                    if ms in list(mlgraph.layers[layer].mobility_services.keys())][0]
            available_layers[i].add("TRANSIT")
        else:
            for layer in mlgraph.layers.values():
                # Take the first registered mobility service
                if len(layer.mobility_services)>0:
                    services[layer.id] = list(layer.mobility_services.keys())[0]

        services["TRANSIT"] = "WALK"
        chosen_mservice[i] = services

    return origins, destinations, available_layers, chosen_mservice


def _process_shortest_path_inputs_layer_group(mlgraph: MultiLayerGraph, users: list, layer_group: set):
    odlayer = mlgraph.odlayer
    user_ids = []
    origins = []
    destinations = []
    available_layers = []
    chosen_mservices = []

    origins_id = list(odlayer.origins.keys())
    origins_pos = np.array([position for position in odlayer.origins.values()])
    destinations_id = list(odlayer.destinations.keys())
    destinations_pos = np.array([position for position in odlayer.destinations.values()])

    for u in users:
        # Check that the intersection between this user's available layers and layer_group
        # is not empty
        u_available_layers = set(list(mlgraph.layers.keys())) if u.available_mobility_service is None \
            else {layer.id for mservice in u.available_mobility_service for layer in mlgraph.layers.values() if mservice in layer.mobility_services}
        considered_available_layers = u_available_layers.intersection(layer_group)

        if len(considered_available_layers) > 0:
            user_ids.append(u.id)
            # Get origin and destination of this user
            if isinstance(u.origin, np.ndarray):
                origins.append(origins_id[np.argmin(_norm(origins_pos - u.origin, axis=1))])
                destinations.append(destinations_id[np.argmin(_norm(destinations_pos - u.destination, axis=1))])
            else:
                origins.append(u.origin)
                destinations.append(u.destination)
            # Get chosen mobility services of this user
            mservices = {'TRANSIT': 'WALK'}
            if u.available_mobility_service is None:
                # Take the first registered mobility service of each layer
                for layer in considered_available_layers:
                    mservices[layer] = list(mlgraph.layers[layer].mobility_services.keys())[0]
            else:
                # Take the first defined mobility service of each layer
                for layer in considered_available_layers:
                    mservices[layer] = [ms for ms in u.available_mobility_service \
                        if ms in list(mlgraph.layers[layer].mobility_services.keys())][0]
            chosen_mservices.append(mservices)
            # Add TRANSIT layer in the set
            considered_available_layers.add('TRANSIT')
            available_layers.append(considered_available_layers)
    return user_ids, origins, destinations, available_layers, chosen_mservices


class AbstractDecisionModel(ABC):

    def __init__(self,
                 mlgraph: MultiLayerGraph,
                 n_shortest_path: int = 3,
                 min_diff_dist: float = -100,
                 max_diff_dist: float = 100,
                 personal_mob_service_park_radius: float = 100,
                 outfile: str = None,
                 verbose_file: bool = False,
                 cost: str = 'travel_time',
                 thread_number: int = multiprocessing.cpu_count()):

        """
        Base class for a travel decision model

        Args:
            mlgraph: The multi layer graph on which the model compute the path
            n_shortest_path: The number of shortest path computed for one user
            min_diff_dist: The min distance between the n computed shortest path and the first one that is required to accept the n shortest path
            max_diff_dist: The max distance between the n computed shortest path and the first one that is required to accept the n shortest path
            personal_mob_service_park_radius: radius around refused user origin in which she can still have access to her personal mobility services such as personal car
            outfile: If specified the file in which compute path are written
            verbose_file: If true write all the computed shortest path, not only the one that is selected
            cost: The name of the cost to consider for the shortest path
            thread_number: The number of thread to user fot parallel shortest path computation
        """

        self._n_shortest_path = n_shortest_path
        self._min_diff_dist = min_diff_dist
        self._max_diff_dist = max_diff_dist
        self.personal_mob_service_park_radius = personal_mob_service_park_radius
        self._thread_number = thread_number

        self._mlgraph = mlgraph
        self._cost = cost
        self._verbose_file = verbose_file
        self._mandatory_mobility_services = []

        self._refused_user: List[User] = list()

        if outfile is None:
            self._write = False
            self._verbose_file = False
        else:
            self._write = True
            self._outfile = open(outfile, 'w')
            self._csvhandler = csv.writer(self._outfile, delimiter=';', quotechar='|')
            self._csvhandler.writerow(['ID', 'COST', 'PATH', 'LENGTH', 'SERVICE'])

    @abstractmethod
    def path_choice(self, paths: List[Path]) -> Path:
        pass

    def set_mandatory_mobility_services(self, services:List[str]):
        self._mandatory_mobility_services = services

    def set_refused_users(self, users: List[User]):
        self._refused_user.extend(users)

    def _check_refused_users(self, tcurrent) -> List[User]:
        new_users = []
        gnodes = self._mlgraph.graph.nodes
        for u in self._refused_user:
            upath = u.path.nodes
            ind_node_start = upath.index(u._current_node)
            for ilayer, (layer, slice_nodes) in enumerate(u.path.layers):
                if slice_nodes.start <= ind_node_start < slice_nodes.stop:
                    refused_mservice = u.path.mobility_services[ilayer]
                    break
            else:
                print("Mobility service not found in User path")
                sys.exit(-1)

            # Get all available mobility services
            all_mob_services = [list(v.mobility_services.values()) for v in self._mlgraph.layers.values()]
            all_mob_services = [item for sublist in all_mob_services for item in sublist]
            all_mob_services_names = set([ms.id for ms in all_mob_services])
            personal_mob_services_names = set([ms.id for ms in all_mob_services if isinstance(ms,PersonalMobilityService)])
            # If user had no defined available mobility services, it means she had
            # access to all
            if u.available_mobility_service is None:
                u.available_mobility_service = all_mob_services_names
            # Remove the mobility service traveler has (been) refused (on)
            if refused_mservice in u.available_mobility_service:
                u.available_mobility_service.remove(refused_mservice)
            # Remove personal mobility services if traveler is not near home/origin
            # anymore
            current_pos = gnodes[u._current_node].position
            origin_pos = u.origin if isinstance(u.origin, np.ndarray) else gnodes[u.origin].position
            for personal_mob_service in personal_mob_services_names:
                if personal_mob_service in u.available_mobility_service and _norm(np.array(origin_pos) - np.array(current_pos)) > self.personal_mob_service_park_radius:
                    u.available_mobility_service.remove(personal_mob_service)
            # Check if user has no remaining available mobility_service
            if len(u.available_mobility_service) == 0:
                log.warning(f'{u.id} has no more available mobility service to continue her path')
                u.set_state_deadend() # user is now blocked
                continue

            # Create a new user representing refused user legacy
            u._continuous_journey = u.id
            u.id = f"{u.id}_CONTINUOUS"
            u.path = None
            new_users.append(u)
            u.origin = np.array(current_pos)
            u.departure_time = tcurrent.copy()

        self._refused_user = list()
        return new_users

    # TODO: restrict combination of paths (ex: we dont want Uber->Bus)
    """def __call__(self, new_users: List[User], tcurrent: Time):
        legacy_users = self._check_refused_users(tcurrent)
        log.info(f'There are {len(new_users)} new users and {len(legacy_users)} legacy users')
        all_users = legacy_users+new_users

        if len(all_users)>0:
            origins, destinations, available_layers, chosen_services = _process_shortest_path_inputs(self._mlgraph, all_users)
            paths = parallel_k_shortest_path(self._mlgraph.graph,
                                             origins,
                                             destinations,
                                             self._cost,
                                             chosen_services,
                                             available_layers,
                                             self._min_diff_dist,
                                             self._max_diff_dist,
                                             self._n_shortest_path,
                                             self._thread_number)
            gnodes = self._mlgraph.graph.nodes
            path_not_found = []

            for i, kpath in enumerate(paths):
                user_paths = []
                user = all_users[i]
                path_index = 0
                for p in kpath:
                    if p[0]:
                        p = Path(path_index, p[1], p[0])
                        if len(p.nodes)<=3:
                            log.warning(f"Incorrect path {p.layers} ignored for user {user.id} with available_layers {available_layers[i]}")
                            break
                        else:
                            p.construct_layers(gnodes)
                            # Check that path layers are all available for this user
                            # (this check is useful for path of type ...-> CARnode - TRANSIT -> BUSnode - TRANSIT -> DESTINATIONnode for e.g.
                            # where BUS is not an available layer, path do not pass through BUS links but only one BUSnode
                            if len(available_layers[i]) == 0 or set([lid for lid,slice in p.layers]).issubset(available_layers[i]):
                                path_index += 1
                                user_chosen_services = chosen_services[i]
                                user_mobility_services = [user_chosen_services[layer_id] for layer_id, slice_nodes in p.layers]
                                p.mobility_services = user_mobility_services
                                user_paths.append(p)
                                service_costs = sum_dict(*(self._mlgraph.layers[layer].mobility_services[service].service_level_costs(p.nodes[node_inds]) for (layer, node_inds), service in zip(p.layers, p.mobility_services)))
                                p.service_costs = service_costs
                            else:
                                log.warning(f"Incorrect path {p.layers} ignored for user {user.id} with available_layers {available_layers[i]}")
                    else:
                        path_not_found.append(user)
                        log.warning(f"Path not found for %s", user.id)

                # Check if every possible User mobility service option has been explored
                # if not compute the mono modal journey with the missed option
                if user.available_mobility_service is not None:
                    used_mobility_services = set(service for p in user_paths for service in p.mobility_services)
                    missed_mobility_services = user.available_mobility_service.difference(used_mobility_services)
                    for mservice in missed_mobility_services:
                        layer_id = self._mlgraph.mapping_layer_services[mservice].id
                        path, cost = dijkstra(self._mlgraph.graph,
                                              origins[i],
                                              destinations[i],
                                              self._cost,
                                              {layer_id: mservice,
                                               "TRANSIT": "WALK"},
                                              set([layer_id, "TRANSIT"]))
                        if path:
                            path_index += 1
                            p = Path(path_index, cost, path)
                            p.construct_layers(gnodes)
                            p.mobility_services = [mservice]
                            p.service_costs = self._mlgraph.layers[layer_id].mobility_services[mservice].service_level_costs(p.nodes[p.layers[0][1]])
                            user_paths.append(p)

                if user_paths:
                    path = self.path_choice(user_paths)
                    if len(path.nodes) > 1:
                        user.set_path(path)
                        user._remaining_link_length = self._mlgraph.graph.nodes[path.nodes[0]].adj[path.nodes[1]].length
                    else:
                        log.warning(f"Path %s is not valid for %s", str(path), user.id)
                        raise PathNotFound(user.origin, user.destination)

                    log.info(f"Computed path for {user.id} is {path}")

                    if self._verbose_file:
                        for p in user_paths:
                            self._csvhandler.writerow([user.id,
                                                       str(path.path_cost),
                                                       ' '.join(p.nodes),
                                                       compute_path_length(self._mlgraph.graph, p.nodes),
                                                       ' '.join(p.mobility_services)])

                    elif self._write:
                        self._csvhandler.writerow([user.id,
                                                   str(user.path.path_cost),
                                                   ' '.join(user.path.nodes),
                                                   compute_path_length(self._mlgraph.graph, user.path.nodes),
                                                   ' '.join(user.path.mobility_services)])

            if path_not_found:
                log.warning("Paths not found: %s", len(path_not_found))
                for u in path_not_found:
                    self._csvhandler.writerow([u.id,
                                               "INF",
                                               u._current_node,
                                               "INF",
                                               ''])

        return all_users"""

    def __call__(self, new_users: List[User], tcurrent: Time):
        #assert len(self._layers_groups) > 0, 'Travel decision launched in per_layers_groups type '\
        #    'without any layers groups defined...'
        all_users = self.call_per_layers_groups(new_users, tcurrent)
        return all_users

    def call_per_layers_groups(self, new_users: List[User], tcurrent: Time):
        legacy_users = self._check_refused_users(tcurrent)
        log.info(f"There are {len(legacy_users)} legacy users and {len(new_users)}"\
            " new users for which path should be computed")
        all_users = legacy_users+new_users
        gnodes = self._mlgraph.graph.nodes

        if len(all_users)>0:
            users_paths = {u.id: {'user': u, 'paths': [], 'layers_groups': []} for u in all_users}
            # mandatory_layers_groups = ({'CAR'}, {'TRAMLayer', 'METROLayer'})
            mandatory_layers_groups = ({"emoped_layer"}, {'BUS'})
            # Compute shortest path per layer group
            for layer_group in ({"emoped_layer"}, {'BUS'}): #({'CAR'}, {'TRAMLayer', 'METROLayer'}):
                # Compute shortest path on the intersection between layers of the
                # layer group and available layers for the user, if this intersection
                # is empty, do not compute shortest path for this user on this layer group
                users_ids, origins, destinations, available_layers, chosen_services = _process_shortest_path_inputs_layer_group(self._mlgraph, all_users, layer_group)
                paths = parallel_k_shortest_path(self._mlgraph.graph,
                                                 origins,
                                                 destinations,
                                                 self._cost,
                                                 chosen_services,
                                                 available_layers,
                                                 self._min_diff_dist,
                                                 self._max_diff_dist,
                                                 1,
                                                 self._thread_number)
                for i, kpath in enumerate(paths):
                    user = users_paths[users_ids[i]]['user']
                    path_index = len(users_paths[users_ids[i]]['paths'])
                    p = kpath[0] # only one path computed so we do not forget any path
                    if p[0]:
                        p = Path(path_index, p[1], p[0])
                        p.construct_layers(gnodes)#, pzones=self._mlgraph.pricing_zones)
                        # Check that path layers are all available for this user
                        # (this check is useful for path of type ...-> CARnode - TRANSIT -> BUSnode - TRANSIT -> DESTINATIONnode for e.g.
                        # where BUS is not an available layer, path do not pass through BUS links but only one BUSnode
                        if set([lid for lid,slice in p.layers]).issubset(available_layers[i]):
                            user_chosen_services = chosen_services[i]
                            user_mobility_services = [user_chosen_services[layer_id] for layer_id, slice_nodes in p.layers]
                            p.mobility_services = user_mobility_services
                            #service_costs = sum_cost_dict(*(self._mlgraph.layers[layer].mobility_services[service].service_level_costs(p.nodes[node_inds]) for (layer, node_inds), service in zip(p.layers, p.mobility_services)))
                            #p.service_costs = service_costs
                            # Add eventual additional costs
                            #if self._cost in list(self._mlgraph._additional_cost_functions.keys()):
                            #    p.update_cost(self._mlgraph._additional_cost_functions[self._cost](p))
                            users_paths[users_ids[i]]['paths'].append(p)
                            users_paths[users_ids[i]]['layers_groups'].append((layer_group,mandatory_layers_groups))
                        else:
                            log.warning(f"Incorrect path ignored for user {user.id} in layers group {available_layers[i]}: {p}")
                    else:
                        log.warning(f"Path not found for {user.id} on layers group: {available_layers[i]}")

                # Compute intermodal shortest path for users that can access at least
                # one of the layer defined in mandatory_layers_groups[0] and and at least
                # one of the layer defined in mandatory_layers_groups[1], if user cannot
                # access any, do not compute shortest path for this user of this layer group
                # with mandatory intermodality
                users_ids, origins, destinations, available_layers, chosen_services = \
                    _process_shortest_path_inputs_layer_group(self._mlgraph, all_users, layer_group)
                paths = parallel_intermodal_dijkstra(self._mlgraph.graph,
                                                 origins,
                                                 destinations,
                                                 chosen_services,
                                                 self._cost,
                                                 self._thread_number,
                                                 mandatory_layers_groups,
                                                 available_layers)
                for i, p in enumerate(paths):
                    user = users_paths[users_ids[i]]['user']
                    path_index = len(users_paths[users_ids[i]]['paths'])
                    if p[0]:
                        p = Path(path_index, p[1], p[0])
                        p.construct_layers(gnodes, pzones=self._mlgraph.pricing_zones)
                        intermodal_path_valid = True
                        # Check 1)
                        for mandatory_layers_group in [mandatory_layers_groups[0],mandatory_layers_groups[1]]:
                            # Find the path computed on this layer group only
                            try:
                                idx = users_paths[users_ids[i]]['layers_groups'].index((mandatory_layers_group, None))
                                control_path = users_paths[users_ids[i]]['paths'][idx]
                                if mandatory_layers_group.isdisjoint(set(control_path.get_links_layers(gnodes))):
                                    log.warning(f'Intermodal path {p} doest not check quality criteria, ignored')
                                    intermodal_path_valid = False
                            except ValueError:
                                # No path computed for this layer group means no check possible
                                # Since it probably means that the path computed on this layer
                                # group was not valid, the intermodal path using this group layer
                                # is ignored
                                log.warning(f'No path computed for layers group {mandatory_layers_group} '\
                                    f"for user {user.id} ({users_paths[users_ids[i]]['layers_groups']}), cannot check validity of intermodal path so ignore it")
                                intermodal_path_valid = False
                        # Check 2)
                        if len(set(p.nodes)) != len(p.nodes):
                            log.warning(f'Intermodal path {p} doest not check quality criteria, ignored')
                            intermodal_path_valid = False
                        if intermodal_path_valid and set([lid for lid,slice in p.layers]).issubset(available_layers[i]):
                            user_chosen_services = chosen_services[i]
                            user_mobility_services = [user_chosen_services[layer_id] for layer_id, slice_nodes in p.layers]
                            p.mobility_services = user_mobility_services
                            service_costs = sum_cost_dict(*(self._mlgraph.layers[layer].mobility_services[service].service_level_costs(p.nodes[node_inds]) for (layer, node_inds), service in zip(p.layers, p.mobility_services)))
                            p.service_costs = service_costs
                            # Add eventual additional costs
                            if self._cost in list(self._mlgraph._additional_cost_functions.keys()):
                                p.update_cost(self._mlgraph._additional_cost_functions[self._cost](p))
                            users_paths[users_ids[i]]['paths'].append(p)
                            users_paths[users_ids[i]]['layers_groups'].append((layer_group,mandatory_layers_groups))
                        else:
                            log.warning(f"Incorrect path {p} ignored for user {user.id} on layers group {available_layers[i]} with mandatory transit layers {mandatory_layers_groups}")
                    else:
                        log.warning(f"Path not found for {user.id} on layers group {available_layers[i]} with mandatory transit layers {mandatory_layers_groups}")


            # Proceed to path selection for each user
            for uid in users_paths.keys():
                user = users_paths[uid]['user']
                possible_paths = users_paths[uid]['paths']
                user.set_possible_paths(possible_paths)
                path = self.path_choice(possible_paths)
                if len(path.nodes) > 1:
                    user.set_path(path)
                    user._remaining_link_length = self._mlgraph.graph.nodes[path.nodes[0]].adj[path.nodes[1]].length
                else:
                    log.warning(f"Path %s is not valid for %s", str(path), user.id)
                    raise PathNotFound(user.origin, user.destination)
                log.info(f"Computed path for {uid} is {path}")
                # Save initial path chosen cost
                # user.set_predicted_travel_cost(path.path_cost)
                # Save path in observer
                if self._verbose_file:
                    for p in possible_paths:
                        self._csvhandler.writerow([user.id,
                                                   str(p.path_cost),
                                                   ' '.join(p.nodes),
                                                   compute_path_length(self._mlgraph.graph, p.nodes),
                                                   ' '.join(p.mobility_services)])

                elif self._write:
                    self._csvhandler.writerow([user.id,
                                               str(user.path.path_cost),
                                               ' '.join(user.path.nodes),
                                               compute_path_length(self._mlgraph.graph, user.path.nodes),
                                               ' '.join(user.path.mobility_services)])

        return all_users

    def compute_path(self, origin: str, destination: str, accessible_layers: Set[str], chosen_services: Dict[str, str]):
        return dijkstra(self._mlgraph.graph,
                        origin,
                        destination,
                        self._cost,
                        chosen_services,
                        accessible_layers)
