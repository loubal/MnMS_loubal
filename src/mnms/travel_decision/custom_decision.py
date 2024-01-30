from typing import List
import numpy as np

from mnms.graph.layers import MultiLayerGraph
from mnms.demand.user import Path
from mnms.travel_decision.abstract import AbstractDecisionModel


class CustomDecisionModel(AbstractDecisionModel):
    def __init__(self, mmgraph: MultiLayerGraph, considered_modes=None, cost='travel_time', outfile:str=None, verbose_file=False, personal_mob_service_park_radius:float=100):
        """
        Deterministic decision model: the path with the lowest cost is chosen.

        Args:
            -mmgraph: The graph on which the model compute the path
            -considered_modes: List of guidelines for the guided paths discovery,
                               if None, the default paths discovery is applied
            -cost: name of the cost to consider
            -outfile: If specified the file in which chosen paths are written
            -verbose_file: If True write all the computed shortest path, not only the one that is selected
            -personal_mob_service_park_radius: radius around user's personal veh parking location in which
                                               she can still have access to her vehicle
        """
        super(CustomDecisionModel, self).__init__(mmgraph, considered_modes=considered_modes,
                                                 n_shortest_path=1, outfile=outfile,
                                                 verbose_file=verbose_file,
                                                 cost=cost, personal_mob_service_park_radius=personal_mob_service_park_radius)

    def path_choice(self, paths:List[Path]) -> Path:
        """Method that proceeds to the selection of the path.

        Args:
            -paths: list of paths to consider for the choice
        """
        costs = [p.path_cost for p in paths]
        for i, path in enumerate(paths):
            modes = path.mobility_services
            use_pt = ('BUS' in modes) or ('TRAM' in modes) or ('METRO' in modes)
            use_emoped = 'emoped1' in modes
            cost_add = 0
            if use_emoped:
                cost_add += 1 # EUR, fixed fee
            if use_pt:
                cost_add += 1 # EUR, like daily abo
            if use_emoped and not use_pt:
                cost_add += 0 # EUR, tax
            costs[i] += cost_add

        # Sort paths by ascending cost before returning the best path
        i_p = np.argmin(costs)
        return paths[i_p]