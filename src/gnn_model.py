import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool

class GNNBackbone(nn.Module):
    def __init__(self, input_channels=12, hidden_channels=64):
        super().__init__()
        self.conv1 = SAGEConv(input_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x, edge_index, batch):
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.2, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        g = global_mean_pool(h, batch)
        return g

class MusicGraphSAGE(nn.Module):
    def __init__(self, input_channels=12, hidden_channels=64, num_classes=50):
        super().__init__()
        self.backbone = GNNBackbone(input_channels, hidden_channels)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, batch):
        g = self.backbone(x, edge_index, batch)
        logits = self.classifier(g)
        return logits, g
