import dgl
import torch
import numpy as np

LARGEST_ATOMIC_NUMBER = 119
SI_POTENTIALS = [
    'erhartalbe',
    'edip',
    'meam_spline',
    'sw',
    'sw_silicene1',
    'Tersoff',
    'Tersoff_T3',
    'CF_MOD'
]
AL_POTENTIALS = [
    'WKG',
    'Zha',
    'SL',
    'ZJW_NIST',
    "ZM",
    'EA',
    'JSN',
    'GW_High',
    'GW_Low',
    'GW_Med'
]

def remove_duplicate_edges(edge_index):
    edge_dict = {}
    edges = []
    for i, j in zip(edge_index[0], edge_index[1]):
        if (i, j) not in edge_dict and (j, i) not in edge_dict:
            edges.append((i, j))
            edge_dict[(i, j)] = 1
    
    edges = np.array(edges).T
    return (edges[0], edges[1])

def neighbor_list_to_molecular_graph(edge_index, pos, atomic_numbers, image, is_contributing):
    bidirectional_edge_index = (
        torch.cat([edge_index[0], edge_index[1]]),
        torch.cat([edge_index[1], edge_index[0]])
    )

    source_pos = pos[bidirectional_edge_index[0]]
    target_pos = pos[bidirectional_edge_index[1]]

    distance = torch.linalg.norm(
        source_pos - target_pos, dim=1).float()  # euclidean distance

    graph = dgl.graph(bidirectional_edge_index, num_nodes=is_contributing.numel())
    graph.ndata['node_feats'] = atomic_numbers.unsqueeze(1)
    graph.ndata['is_contributing'] = is_contributing.unsqueeze(1)
    graph.ndata['image'] = image.unsqueeze(1)
    graph.edata['edge_feats'] = distance.unsqueeze(1)

    return graph

def neighbor_list_to_directed_molecular_graph(edge_index, pos, atomic_numbers, image, is_contributing):
    bidirectional_edge_index = (
        edge_index[1],
        edge_index[0]
    )

    source_pos = pos[bidirectional_edge_index[0]]
    target_pos = pos[bidirectional_edge_index[1]]

    distance = torch.linalg.norm(
        source_pos - target_pos, dim=1).float()  # euclidean distance

    graph = dgl.graph(bidirectional_edge_index, num_nodes=is_contributing.numel())
    graph.ndata['node_feats'] = atomic_numbers.unsqueeze(1)
    graph.ndata['is_contributing'] = is_contributing.unsqueeze(1)
    graph.ndata['image'] = image.unsqueeze(1)
    graph.edata['edge_feats'] = distance.unsqueeze(1)

    return graph