import torch.nn as nn
from torch_geometric.nn import SAGEConv
import torch.nn.functional as F
import torch


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x
    
class ProjectionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128)
        )
    def forward(self, x):
        return self.model(x)
    

class LinkPredictor(nn.Module):
    def __init__(self, dim = 128, hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
    def forward(self, h1, h2):
        x = torch.cat([h1, h2], dim = -1)
        return torch.sigmoid(self.net(x)).squeeze()