import multiprocessing
from hipop.shortest_path import parallel_k_shortest_path
import json

def precompute_shortest_paths(graph, chosen_mservice, layer_name, outfile):
    """Fonction that pre-computes the shortest paths for each pair of nodes of the graph
    of a layer.
    By shortest we mean in distance.

    - graph: graph of the layer
    - chosen_mservice: dict with a unique element {'layer_name': 'mob_service_name'}
    - layer_name: name of the layer
    - outfile: file where the shortest paths will be saved
    """

    #amodrh_graph = supervisor._mlgraph.layers['AMODRH'].graph
    # Launch parallel dijkstra on all od pairs
    all_ods = [(n1,n2) for n1 in graph.nodes for n2 in graph.nodes if n1 != n2]
    origins = [t[0] for t in all_ods]
    destinations = [t[1] for t in all_ods]
    paths = parallel_k_shortest_path(graph,
                                 origins,
                                 destinations,
                                 'length',
                                 [chosen_mservice]*len(origins),
                                 [{layer_name}]*len(origins),
                                 -100,
                                 100,
                                 1,
                                 multiprocessing.cpu_count())
    # Build a dict of shortest paths
    sps = {}
    for i,(o,d) in enumerate(all_ods):
        if o in sps.keys():
            sps[o][d] = paths[i][0][0]
        else:
            sps[o] = {d: paths[i][0][0]}

    # Dump dict into json
    with open(outfile, 'w') as f:
        json.dump(sps, f)
