import torch
import torch.nn as nn
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=256):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, out_dim)
        self.dense2 = nn.Linear(in_dim+ out_dim, out_dim)
        self.dense3 = nn.Linear(in_dim+out_dim, out_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self,in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        h = self.act(self.dense3(torch.cat([in_,z2],-1)))
        return h


class EventDecoder(nn.Module):   #인코더랑 동일함 명칭만 다르고  음 뺼까
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, out_dim)
        self.dense2 = nn.Linear(in_dim + out_dim, out_dim)
        self.dense3 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        h = self.act(self.dense3(torch.cat([in_, z2], -1)))
        return h


class VeracityDecoder(nn.Module):
    def __init__(self, in_dim=256, out_dim=128):
        super().__init__()
        self.dense1 = nn.Linear(in_dim, out_dim)
        self.dense2 = nn.Linear(in_dim + out_dim, out_dim)
        self.dense3 = nn.Linear(in_dim + out_dim, out_dim)
        self.act = nn.LeakyReLU(0.1)

        self.cls_dense = nn.Linear(out_dim,1)

    def forward(self, in_):
        z1 = self.act(self.dense1(in_))
        z2 = self.act(self.dense2(torch.cat([in_, z1], -1)))
        l = self.act(self.dense3(torch.cat([in_, z2], -1)))
        label_logit = self.cls_dense(l).squeeze(-1)     # sigmoid(logit) = ỹ (Eq 8)

        return l, label_logit



class DisentangleLoss(nn.Module):
    def __init__(self, e_dim=128):
        super().__init__()
        # Eq 11의 적대 예측기: ỹ = sigmoid(Dense(e))
        self.dense_e = nn.Linear(e_dim, 1) #e를 y햇으로 만든거

    def forward(self, v, e, l, label_logit, y, mask=None):
        """
        v [..., 256] 원본 뉴스 벡터 / e [...,128] 사건 표현 / l [...,128] 진위 표현
        cls_logit [...] veracity decoder 분류기의 로짓 / y [...] 정답 (1=fake)
        mask [...] 유효 위치=1 (패딩 자리 제외용, 없으면 전체 평균)
        """
        # Eq 9 — 재구성: ½‖[e;l] − v‖²
        loss_r = 0.5 * ((torch.cat([e, l], -1) - v) ** 2).mean(-1)

        # Eq 10 — 라벨 예측 (BCE)
        loss_l = F.binary_cross_entropy_with_logits(label_logit, y, reduction="none")

        # Eq 11 — 적대: e로 진위를 예측한 "오차"가 클수록 손실이 작아짐
        adv_logit = self.dense_e(e).squeeze(-1)
        adv_err = F.binary_cross_entropy_with_logits(adv_logit, y, reduction="none")
        loss_a = 1.0 / adv_err.clamp(min=1e-6)      # 논문 Eq 11 식

        # Eq 12 — 합 (mask 있으면 패딩 자리 빼고 평균)
        losses = loss_l  # +loss_a +loss_r
        if mask is not None:
            total = (losses * mask).sum() / mask.sum().clamp(min=1)
        else:
            total = losses.mean()
        return total, \
               {"L_r": loss_r.mean().item(), "L_l": loss_l.mean().item(), "L_a": loss_a.mean().item()}

