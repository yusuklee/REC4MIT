"""
Rec4Mit 데이터 전처리 통합 파이프라인
  1) interaction : 뉴스 CSV -> 사용자별 시간순 열람 시퀀스
  2) instance    : 시퀀스 -> 학습 인스턴스 + 10-fold 분할
  3) emb         : 뉴스 제목·본문 -> BERT 임베딩

  python DataPreperation/dataPipeline.py
  python DataPreperation/dataPipeline.py --data pol --steps interaction instance
"""
import argparse
import ast
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
NEWS_DIR = os.path.join(BASE, "datas", "news")
INTER_DIR = os.path.join(BASE, "datas", "user_interaction")
FOLD_DIR = os.path.join(BASE, "datas", "folds")
EMB_DIR = os.path.join(BASE, "datas", "emb")

NUM_INSTANCES = {"gossip": 136004, "pol": 47464}   # 논문과 동일한 인스턴스 수
CTX_LEN = 4
TITLE_LEN = 32
DESC_LEN = 128
BERT_NAME = "bert-base-uncased"


def load_news(data):
    return pd.read_csv(os.path.join(NEWS_DIR, f"{data}.csv"))


def make_interaction(data):
    news = load_news(data)

    user_news = defaultdict(list)
    for _, row in news.iterrows():
        users = ast.literal_eval(row["user_ids"])
        times = ast.literal_eval(row["user_times"])
        for u, t in zip(users, times):
            user_news[u].append((row["news_id"], t))

    seq = {u: [n for n, _ in sorted(v, key=lambda x: x[1])] for u, v in user_news.items()}

    os.makedirs(INTER_DIR, exist_ok=True)
    out = os.path.join(INTER_DIR, f"{data}_user_interaction.json")
    json.dump(seq, open(out, "w"))
    print(f"[1/3 interaction] {data}: 사용자 {len(seq):,}명 -> {out}")


def interaction2instance(data):
    user_interaction = json.load(open(os.path.join(INTER_DIR, f"{data}_user_interaction.json")))

    instances = []
    for uid, news in user_interaction.items():
        for i in range(1, len(news)):
            instances.append((tuple(news[max(0, i - CTX_LEN):i]), news[i], uid))

    instances = list(set(instances))[:NUM_INSTANCES[data]]

    # fold k: 청크 k = test, 청크 k+1 = val, 나머지 8개 = train
    order = np.random.default_rng(3).permutation(len(instances))
    chunks = np.array_split(order, 10)
    for k in range(10):
        test = [instances[i] for i in chunks[k]]
        val = [instances[i] for i in chunks[(k + 1) % 10]]
        rest = set(range(10)) - {k, (k + 1) % 10}
        train = [instances[i] for j in rest for i in chunks[j]]

        fold_dir = os.path.join(FOLD_DIR, data, str(k))
        os.makedirs(fold_dir, exist_ok=True)
        json.dump(train, open(os.path.join(fold_dir, "train.json"), "w"))
        json.dump(val, open(os.path.join(fold_dir, "val.json"), "w"))
        json.dump(test, open(os.path.join(fold_dir, "test.json"), "w"))

    print(f"[2/3 instance] {data}: 인스턴스 {len(instances):,}개 -> {os.path.join(FOLD_DIR, data)}")


def make_emb(data, model):
    news = load_news(data)
    news["title"] = news["title"].fillna("").astype(str)
    news["description"] = news["description"].fillna("").astype(str)

    titles = [t.strip() or "unknown news" for t in news["title"]]
    has_text = news["description"].str.len() > 5

    def embed(texts):
        return model.encode(texts, batch_size=128, normalize_embeddings=True,
                            convert_to_numpy=True).astype(np.float32)

    model.max_seq_length = TITLE_LEN
    T = embed(titles)

    model.max_seq_length = DESC_LEN
    D = np.zeros((len(news), 768), dtype=np.float32)
    text_idx = np.where(has_text.values)[0].tolist()
    if text_idx:
        D[text_idx] = embed([news["description"].iloc[i] for i in text_idx])

    os.makedirs(EMB_DIR, exist_ok=True)
    out = os.path.join(EMB_DIR, f"{data}.npz")
    np.savez(out, news_id=np.array(news["news_id"].astype(str)), title=T, description=D)
    print(f"[3/3 emb] {data}: {T.shape} -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["gossip", "pol"], choices=["gossip", "pol"])
    p.add_argument("--steps", nargs="+", default=["interaction", "instance", "emb"],
                   choices=["interaction", "instance", "emb"])
    ar = p.parse_args()

    if "interaction" in ar.steps:
        for d in ar.data:
            make_interaction(d)

    if "instance" in ar.steps:
        for d in ar.data:
            interaction2instance(d)

    if "emb" in ar.steps:
        from sentence_transformers import SentenceTransformer   # 임베딩 단계에서만 필요
        model = SentenceTransformer(BERT_NAME)
        for d in ar.data:
            make_emb(d, model)
