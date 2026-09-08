import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

def build_segment_graph(node_features, label_vector=None, segment_number=7, threshold=0.70):
    edges = []
    for i in range(segment_number - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])

    norm = np.linalg.norm(node_features, axis=1, keepdims=True) + 1e-8
    norm_feats = node_features / norm
    sim_matrix = np.dot(norm_feats, norm_feats.T)

    for i in range(segment_number):
        for j in range(i + 1, segment_number):
            if sim_matrix[i, j] >= threshold:
                edges.append([i, j])
                edges.append([j, i])

    x = torch.tensor(node_features, dtype=torch.float)
    if len(edges) > 0:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_index = to_undirected(edge_index)
    if label_vector is not None:
        y_tensor = torch.tensor(label_vector, dtype=torch.float).unsqueeze(0)
        return Data(x=x, edge_index=edge_index, y=y_tensor)
    return Data(x=x, edge_index=edge_index)
