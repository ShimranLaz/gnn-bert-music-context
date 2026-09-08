import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MusicDualEncoder(nn.Module):
    def __init__(self, gnn_backbone, bert_backbone, gnn_dim=64, bert_dim=768, embed_dim=128):
        super().__init__()
        self.gnn = gnn_backbone
        self.bert = bert_backbone
        self.audio_proj = nn.Sequential(
            nn.Linear(gnn_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(bert_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, graphs, input_ids, attention_mask):
        g = self.gnn(graphs.x, graphs.edge_index, graphs.batch)
        audio_emb = F.normalize(self.audio_proj(g), p=2, dim=-1)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t_cls = bert_out.last_hidden_state[:, 0, :]
        text_emb = F.normalize(self.text_proj(t_cls), p=2, dim=-1)
        return audio_emb, text_emb

def info_nce_loss(audio_emb, text_emb, logit_scale):
    tau = torch.clamp(logit_scale.exp(), max=100.0)
    sim_matrix = torch.matmul(audio_emb, text_emb.T) * tau
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    loss_a2t = F.cross_entropy(sim_matrix, labels)
    loss_t2a = F.cross_entropy(sim_matrix.T, labels)
    return (loss_a2t + loss_t2a) / 2.0, sim_matrix
