import argparse, glob, json, re
import numpy as np, pandas as pd, torch
from rec4mit import Rec4Mit
from Model.layer1 import init_emb

p = argparse.ArgumentParser()
p.add_argument("--data", default="pol", choices=["gossip", "pol"])
p.add_argument("--batch", default=32, type=int)      # gossip은 후보가 많아 줄여야 함
ar = p.parse_args()
DATA = ar.data
dev = "cuda" if torch.cuda.is_available() else "cpu"

news = pd.read_csv(f"DataPreperation/datas/news/{DATA}.csv")
name2idx = {n: i + 1 for i, n in enumerate(news["news_id"].astype(str))}
labels = np.concatenate([[0], news["label"].values])
NUM_NEWS = len(news)

# u2i는 train.py와 똑같은 규칙으로 다시 만든다 (정렬돼 있어 결정적)
all_users = set()
for s in ("train", "val", "test"):
  for _, _, u in json.load(open(f"DataPreperation/datas/folds/{DATA}/0/{s}.json")):
      all_users.add(u)
u2i = {u: i for i, u in enumerate(sorted(all_users))}

init_matrix = init_emb(f"DataPreperation/datas/emb/{DATA}.npz",
                     f"DataPreperation/datas/news/{DATA}.csv")
TOPK = [5, 10, 20]

folds = sorted(int(re.search(r"fold_(\d+)\.pth", f).group(1))
             for f in glob.glob(f"model_dir/{DATA}/fold_*.pth"))
print(f"학습된 fold: {folds}")


def load_test(fold):
  inst = [x for x in json.load(open(f"DataPreperation/datas/folds/{DATA}/{fold}/test.json"))
          if labels[name2idx[x[1]]] == 0]
  xs, ys, us = [], [], []
  for ctx, tgt, uid in inst:
      ids = [name2idx[c] for c in ctx][-4:]
      xs.append([0] * (4 - len(ids)) + ids)
      ys.append(name2idx[tgt])
      us.append(u2i[uid])
  return torch.tensor(xs), torch.tensor(ys), torch.tensor(us)


@torch.no_grad()
def evaluate(model, xs, ys, us):
  model.eval()
  cand = torch.arange(1, NUM_NEWS + 1, device=dev)          # 뉴스 풀 전체
  _, e_c, _, lg_c = model.disentangle(cand)                 # 후보는 한 번만 인코딩
  fake = (torch.sigmoid(lg_c) >= 0.5).float()

  hit = [0.] * 3; mrr = [0.] * 3; ndcg = [0.] * 3; rt = [0.] * 3
  n = len(ys)
  for i in range(0, n, ar.batch):
      sx, su = xs[i:i+ar.batch].to(dev), us[i:i+ar.batch].to(dev)
      B = sx.size(0)

      _, e_s, _, _ = model.disentangle(sx)
      _, e_split = model.detector(e_s)
      R, _ = model.transition.build_R(e_s, e_split, sx != 0)
      ec = e_c.unsqueeze(0).expand(B, -1, -1)
      c_u, _ = model.transition.activate(R, ec, model.user_emb(su))
      sc = model.predictor(c_u, ec)                          # [B, NUM_NEWS]


      sc = sc - fake * 1e4

      top = sc.topk(20, -1).indices.cpu() + 1                # 0-based -> 뉴스번호
      for b in range(B):
          row = top[b].tolist()
          tgt = int(ys[i + b])
          for j, K in enumerate(TOPK):
              rt[j] += sum(1 for c in row[:K] if labels[c] == 0) / K
              if tgt in row[:K]:
                  rank = row[:K].index(tgt) + 1
                  hit[j] += 1
                  mrr[j] += 1.0 / rank
                  ndcg[j] += 1.0 / np.log2(rank + 1)
  return [x / n for x in hit], [x / n for x in mrr], [x / n for x in ndcg], [x / n for x in rt]


agg = []
for fold in folds:
  model = Rec4Mit(init_matrix, num_users=len(u2i), ctx_len=4).to(dev)
  model.load_state_dict(torch.load(f"model_dir/{DATA}/fold_{fold}.pth", map_location=dev))

  xs, ys, us = load_test(fold)
  hr, mrr, nd, rt = evaluate(model, xs, ys, us)
  agg.append(hr + mrr + nd + rt)
  print(f"fold{fold}: REC@5 {hr[0]:.4f} REC@20 {hr[2]:.4f} | MRR@5 {mrr[0]:.4f} "
        f"| NDCG@5 {nd[0]:.4f} | RT@5 {rt[0]:.4f}")

A = np.array(agg)
names = ["REC@5", "REC@10", "REC@20", "MRR@5", "MRR@10", "MRR@20",
       "NDCG@5", "NDCG@10", "NDCG@20", "RT@5", "RT@10", "RT@20"]
print(f"\n=== [{DATA}] {len(folds)}-fold 평균 ± 표준편차 ===")
for j, nm in enumerate(names):
  print(f"{nm:8s} {A[:, j].mean():.4f} ± {A[:, j].std():.4f}")
