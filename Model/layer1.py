import numpy as np
import torch
import torch.nn as nn
import pandas as pd



def init_emb(NEWS_EMB, NEWS):
    emb = np.load(NEWS_EMB)
    news = pd.read_csv(NEWS)
    title, description = emb["title"], emb["description"]
    E = np.zeros((len(news)+1,1536), dtype=np.float32)
    E[1:, :768] = title
    E[1:, 768:] = description

    return torch.from_numpy(E)



class EmbeddingLayer(nn.Module):
    #meta = title+ description ->1536
    def __init__(self,init_emb,id_dim=128, out_dim=256):
        super().__init__()
        num_rows, meta_dim = init_emb.shape
        self.id_emb = nn.Embedding(num_rows, id_dim, padding_idx=0)
        self.meta_emb = nn.Embedding(num_rows, meta_dim, padding_idx=0)
        with torch.no_grad():
            self.meta_emb.weight.copy_(init_emb)
        self.fc_v = nn.Linear(id_dim+meta_dim, out_dim)

    def forward(self,ids):
        return self.fc_v(torch.cat([self.id_emb(ids), self.meta_emb(ids)],-1))
