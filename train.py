import argparse
import json, torch, torch.nn.functional as F
import numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from main import Rec4Mit
from Model.layer1 import init_emb
import os


p = argparse.ArgumentParser()
p.add_argument("--data", default="pol", choices=["gossip","pol"])
p.add_argument("--fold_num", default=1, type=int, choices=list(range(1,11)))       # 0~9 까지
ar = p.parse_args()
DATA= ar.data
fold_num = ar.fold_num
dev = "cuda" if torch.cuda.is_available() else "cpu"


news = pd.read_csv(f"DataPreperation/datas/news/{DATA}.csv")

# 내부번호: news.csv 행 순서 = init_emb의 E 행 순서 (0은 pad)
name2idx = {n: i + 1 for i, n in enumerate(news["news_id"].astype(str))}
labels = np.concatenate([[0], news["label"].values])  # labels[내부번호]
real_pool = np.where(labels[1:] == 0)[0] + 1
fake_pool = np.where(labels[1:] == 1)[0] + 1


class FoldSet(Dataset):
    def __init__(self, path, user2idx):
        self.inst = [x for x in json.load(open(path)) if labels[name2idx[x[1]]] == 0]      #정답뉴스가 진짜인 인스턴스만 생존
        self.u2i = user2idx

    def __len__(self): return len(self.inst)

    def __getitem__(self, i):
        ctx, tgt, uid = self.inst[i]
        ids = [name2idx[c] for c in ctx][-4:]
        pad = 4 - len(ids)
        seq = [0] * pad + ids  # 왼쪽 패딩
        mask = [False] * pad + [True] * len(ids)

        neg = np.concatenate([np.random.choice(real_pool, 32),  # 진짜32 + 가짜32
                              np.random.choice(fake_pool, 32)])

        cand = [name2idx[tgt]] + neg.tolist()  # 정답후보 진짜만

        return (torch.tensor(seq), torch.tensor(cand),
                torch.tensor(self.u2i[uid]), torch.tensor(mask),
                torch.tensor(labels[seq], dtype=torch.float),
                torch.tensor(labels[cand], dtype=torch.float))

@torch.no_grad()
def evaluate(model, dl, dev):
    model.eval()
    tot, n = 0.0, 0
    for seq, cand, u, mask, y_s, y_c in dl:
        seq, cand, u, mask = seq.to(dev), cand.to(dev), u.to(dev), mask.to(dev)
        y_s, y_c = y_s.to(dev), y_c.to(dev)
        logits, ctx, cd = model(seq, cand, u, mask)
        target = torch.zeros_like(logits);
        target[:, 0] = 1
        loss = loss = loss = F.binary_cross_entropy_with_logits(logits, target) \
        + model.dis_loss(*ctx, y_s, mask)[0] + model.dis_loss(*cd, y_c)[0]
        tot += loss.item() * seq.size(0);
        n += seq.size(0)
    model.train()
    return tot / n





init_matrix = init_emb(f"DataPreperation/datas/emb/{DATA}.npz",
                         f"DataPreperation/datas/news/{DATA}.csv")

all_users = set()

for s in ("train", "val", "test"):
  for _, _, u in json.load(open(f"DataPreperation/datas/folds/{DATA}/0/{s}.json")):
      all_users.add(u)
u2i = {u: i for i, u in enumerate(sorted(all_users))}



os.makedirs(f"model_dir/{DATA}", exist_ok=True)

best = float("inf")
for fold in range(fold_num):
    

    ds = FoldSet(f"DataPreperation/datas/folds/{DATA}/{fold}/train.json", u2i)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    va = DataLoader(FoldSet(f"DataPreperation/datas/folds/{DATA}/{fold}/val.json", u2i),
                    batch_size=64)

                                                               
    model = Rec4Mit(init_matrix, num_users=len(u2i), ctx_len=4).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)


    for ep in range(15): # 15번 반복 학습
        tot = 0
        for seq, cand, u, mask, y_s, y_c in dl:
            seq, cand, u, mask = seq.to(dev), cand.to(dev), u.to(dev), mask.to(dev)
            y_s, y_c = y_s.to(dev), y_c.to(dev)

            logits, ctx, cd = model(seq, cand, u, mask)
            target = torch.zeros_like(logits);
            target[:, 0] = 1
            loss_p = F.binary_cross_entropy_with_logits(logits, target,reduction="none").mean()



            loss_d = model.dis_loss(*ctx, y_s, mask)[0] + model.dis_loss(*cd, y_c)[0]

            (loss_p  + loss_d).backward()  # Eq 22 선호도 추천을 잘못해서 수정
            opt.step();
            opt.zero_grad()
            tot += (loss_p + loss_d).item()
        vl = evaluate(model, va, dev)

        if vl < best:
            best = vl

            torch.save(model.state_dict(), f"model_dir/{DATA}/fold_{fold}.pth")
        print(f"fold{fold} ep{ep} train {tot / len(dl):.4f} val {vl:.4f}")




    


   
