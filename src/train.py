import torch
import torch.nn as nn
from src.gnn_model import MusicGraphSAGE
from src.bert_encoder import DistilBertTagClassifier

def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        optimizer.zero_grad()
        targets = batch['ys'].to(device)
        if hasattr(batch, 'graphs'):
            graphs = batch['graphs'].to(device)
            logits, _ = model(graphs.x, graphs.edge_index, graphs.batch)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)
