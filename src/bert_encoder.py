import torch
import torch.nn as nn
from transformers import AutoModel

class DistilBertTagClassifier(nn.Module):
    def __init__(self, num_classes=50):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_token = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_token)
        return logits, cls_token
