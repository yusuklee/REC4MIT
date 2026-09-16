import argparse
import json
from collections import defaultdict
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--data", default="gossip", choices=["gossip","pol"])
ar = p.parse_args()


user_interaction = json.load(open(f"datas/user_interaction/{ar.data}_user_interaction.json"))


instances = []


for uid, news in user_interaction.items():
    for i in range(1,len(news)):
        ctx = news[max(0,i-4):i]
        tgt = news[i]
        instances.append((tuple(ctx), tgt,uid))

num_instances =0
if ar.data =="gossip":
    num_instances=136004
else:
    num_instances=47464


instances = list(set(instances))
instances = instances[:num_instances]
#pol 47,464
#gossip 136,004



rng = np.random.default_rng(3)
order = rng.permutation(len(instances))
chunks = np.array_split(order,10)



import os
for cnt in range(10):

    test = [instances[i] for i in chunks[cnt]]
    val = [instances[i] for i in chunks[(cnt+1)%10]]

    rest = set(range(10)) - {cnt,(cnt+1)%10}

    train = [instances[i] for j in rest for i in chunks[j]]



    os.makedirs(f"datas/folds/{ar.data}/{cnt}",exist_ok=True)

    with open(f"datas/folds/{ar.data}/{cnt}/train.json","w") as f:
        json.dump(train,f)

    with open(f"datas/folds/{ar.data}/{cnt}/test.json","w") as f:
        json.dump(test,f)

    with open(f"datas/folds/{ar.data}/{cnt}/val.json","w") as f:
        json.dump(val,f)
















