import torch
import torch.nn as nn
import torch.nn.functional as F

class ConcatFusionModel(nn.Module):
    def __init__(self, gnn_backbone, bert_backbone, num_classes=50, gnn_dim=64, bert_dim=768):
        super().__init__()
        self.gnn = gnn_backbone
        self.bert = bert_backbone
        self.classifier = nn.Sequential(
            nn.Linear(gnn_dim + bert_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, graphs, input_ids, attention_mask):
        g = self.gnn(graphs.x, graphs.edge_index, graphs.batch)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        t_cls = bert_out.last_hidden_state[:, 0, :]
        z = torch.cat([g, t_cls], dim=-1)
        logits = self.classifier(z)
        return logits, z

class CrossAttentionFusionModel(nn.Module):
    def __init__(self, gnn_backbone, bert_backbone, num_classes=50, gnn_dim=64, bert_dim=768, proj_dim=128):
        super().__init__()
        self.gnn = gnn_backbone
        self.bert = bert_backbone
        self.w_q = nn.Linear(gnn_dim, proj_dim)
        self.w_k = nn.Linear(bert_dim, proj_dim)
        self.w_v = nn.Linear(bert_dim, proj_dim)
        self.scale = proj_dim ** 0.5
        self.classifier = nn.Sequential(
            nn.Linear(gnn_dim + proj_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, graphs, input_ids, attention_mask, return_attn=False):
        g = self.gnn(graphs.x, graphs.edge_index, graphs.batch)
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h_text = bert_out.last_hidden_state

        q = self.w_q(g).unsqueeze(1)
        k = self.w_k(h_text)
        v = self.w_v(h_text)

        attn_scores = torch.bmm(q, k.transpose(1, 2)) / self.scale
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e4)

        attn_weights = F.softmax(attn_scores, dim=-1)
        context = torch.bmm(attn_weights, v).squeeze(1)
        z = torch.cat([g, context], dim=-1)
        logits = self.classifier(z)

        if return_attn:
            return logits, z, attn_weights
        return logits, z
