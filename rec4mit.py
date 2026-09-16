import torch.nn as nn
from Model.layer1 import EmbeddingLayer
from Model.layer2 import Encoder,EventDecoder,VeracityDecoder,DisentangleLoss
from Model.layer3 import EventDetector,EventTransitionNet,NextNewsPredictor

#layer1, layer2, layer3 를 전부 연결한 코드



class Rec4Mit(nn.Module):
    def __init__(self, init_emb, num_users, k=20, ctx_len=4,
                 v_dim=256, e_dim=128, user_dim=128):
        super().__init__()
        self.embedding = EmbeddingLayer(init_emb, out_dim=v_dim)
        self.encoder = Encoder(v_dim, v_dim)
        self.event_dec = EventDecoder(v_dim, e_dim)
        self.veracity_dec = VeracityDecoder(v_dim, e_dim)
        self.detector = EventDetector(e_dim, k)
        self.transition = EventTransitionNet(e_dim, k, ctx_len, user_dim=user_dim)
        self.predictor = NextNewsPredictor()
        self.user_emb = nn.Embedding(num_users + 1, user_dim, padding_idx=num_users)
        self.dis_loss = DisentangleLoss(e_dim)

    def disentangle(self, ids):
        v = self.embedding(ids)  # [.., 256]
        h = self.encoder(v)
        e = self.event_dec(h)
        l, logit = self.veracity_dec(h)
        return v, e, l, logit

    def forward(self, seq_ids, cand_ids, u_idx, pad_mask):
        v_s, e_s, l_s, lg_s = self.disentangle(seq_ids)  # 컨텍스트 [B,L,*]
        v_c, e_c, l_c, lg_c = self.disentangle(cand_ids)  # 후보    [B,C,*]

        _, e_split = self.detector(e_s)  # Eq 13~14
        R, _ = self.transition.build_R(e_s, e_split, pad_mask)  # Eq 15~16
        c_u, _ = self.transition.activate(R, e_c, self.user_emb(u_idx))  # Eq 17~19
        logits = self.predictor(c_u, e_c)  # Eq 20
        return logits, (v_s, e_s, l_s, lg_s), (v_c, e_c, l_c, lg_c)


