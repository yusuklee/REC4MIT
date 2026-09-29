import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from docutils.nodes import title



def init_emb(NEWS_EMB, NEWS):
    emb = np.load(NEWS_EMB)
    n = pd.read_csv(NEWS)
    E = np.zeros((len(n)+1, 1536), dtype=np.float32)
    E[1:,:768] = emb["title"]
    E[1:, 768:] = emb["description"]

    return torch.from_numpy(E)




class EmbeddingLayer(nn.Module):
    def __init__(self,emb, id_dim=128, out_dim=256):
        super().__init__()
        num_news, meta_dim = emb.shape
        self.id_emb = nn.Embedding(num_news,id_dim,padding_idx=0)
        self.meta_emb = nn.Embedding(num_news,meta_dim,padding_idx=0)
        with torch.no_grad():
            self.meta_emb.weight.copy_(emb)
        self.fc_v = nn.Linear(id_dim+meta_dim,out_dim)

    def forward(self,idx):
        return self.fc_v(torch.cat([self.id_emb(idx),self.meta_emb(idx)],-1))

# id , meta 임베딩을 합친걸 압축

