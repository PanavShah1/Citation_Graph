
import torch
import torch.nn.functional as F
import pickle
import os
import random
from torch.utils.data import TensorDataset, DataLoader
from models import GraphSAGE, ProjectionMLP, LinkPredictor

def train(model, data, epochs = 2_000, lr = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    edge_index = data.edge_index
    num_nodes = data.num_nodes

    for epoch in range(epochs):

        optimizer.zero_grad()
        z = model(data.x, edge_index) # forward pass

        i = torch.randint(0, edge_index.shape[1], (1024,))
        edge_pos = edge_index[:, i]

        src = torch.randint(0, num_nodes, (1024,))
        dst = torch.randint(0, num_nodes, (1024,))
        edge_neg = torch.stack([src, dst], dim=0)

        pos_score = (z[edge_pos[0]] * z[edge_pos[1]]).sum(dim=1)
        neg_score = (z[edge_neg[0]] * z[edge_neg[1]]).sum(dim=1)

        loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()

        loss.backward()
        optimizer.step()

    return None

def train_GraphSAGE(dataG, paper_embeddings, device, save_path = 'data/graphsage_model.pth'):
    dataG_feats = []
    for node_id in range(dataG.num_nodes):
        emb = paper_embeddings.get(int(node_id))
        if emb is None:
            raise ValueError(f"Node ID {node_id} not found in paper embeddings.")
        else:
            dataG_feats.append(emb)
    
    dataG.x = torch.stack(dataG_feats)

    # print("Embedding dim:", dataG.x.shape)

    model = GraphSAGE(in_channels=dataG.x.shape[1], hidden_channels=256, out_channels=128).to(device)
    train(model, dataG)
    torch.save(model.state_dict(), save_path)

    with torch.no_grad():
        final_embs = model(dataG.x, dataG.edge_index).cpu()
    graphsage_emb_dict = {i: final_embs[i] for i in range(dataG.num_nodes)}
    with open("data/graphsage_embeddings.pkl", "wb") as f:
        pickle.dump(graphsage_emb_dict, f)
    return None

def train_projection_mlp(paper_embeddings, graphsage_embeddings, device):
    xs = []
    ys = []
    for i in paper_embeddings:
        if i in graphsage_embeddings:
            xs.append(paper_embeddings[i])
            ys.append(graphsage_embeddings[i])
    xs = torch.stack(xs)
    ys = torch.stack(ys)

    dataset = torch.utils.data.TensorDataset(xs.to(device), ys.to(device)) 
    loader = torch.utils.data.DataLoader(dataset, batch_size = 32, shuffle = True)

    model = ProjectionMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.MSELoss()

    model.train()
    for epoch in range(100):
        for xb, yb in loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    torch.save(model.state_dict(), "data/projection_mlp.pth")
    return model


def train_link_predictor(graphsage_embeddings, citation_edges, device):
    model = LinkPredictor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    loss_fn = torch.nn.BCELoss()

    pos_pairs = citation_edges

    neg_pairs = []
    all_ids = list(graphsage_embeddings.keys())
    while len(neg_pairs) < len(pos_pairs):
        i, j = random.sample(all_ids, 2)
        if (i, j) not in pos_pairs and (j, i) not in pos_pairs:
            neg_pairs.append((i, j))
    
    def pair_to_tensor(pair_list):
        x1 = torch.stack([graphsage_embeddings[i].to(device) for i, j in pair_list])
        x2 = torch.stack([graphsage_embeddings[j].to(device) for i, j in pair_list])
        return x1, x2
    
    pos_x1, pos_x2 = pair_to_tensor(pos_pairs)
    neg_x1, neg_x2 = pair_to_tensor(neg_pairs)

    x1 = torch.cat([pos_x1, neg_x1])
    x2 = torch.cat([pos_x2, neg_x2])

    y = torch.cat([
        torch.ones(len(pos_pairs), device = device),
        torch.zeros(len(neg_pairs), device = device)
    ])

    perm = torch.randperm(x1.size(0))
    x1, x2, y = x1[perm], x2[perm], y[perm]

    model.train()

    for epoch in range(10_000):
        optimizer.zero_grad()
        preds = model(x1, x2)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
    
    torch.save(model.state_dict(), "data/link_predictor.pth")

    return model