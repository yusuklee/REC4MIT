# Rec4Mit — 가짜뉴스 확산 완화를 위한 뉴스 추천 모델 재현

사용자의 최근 뉴스 열람 이력을 보고 **다음에 읽을 뉴스를 추천**하되, **가짜뉴스는 추천 목록에서 걸러내는** 모델입니다.
뉴스 벡터를 **사건(event) 표현**과 **진위(veracity) 표현**으로 분리(disentangle)하고, 사건 흐름을 따라가며 진짜 뉴스만 추천합니다.

데이터셋은 FakeNewsNet(PolitiFact, GossipCop)을 사용합니다.

---

## 목차

1. [폴더 구조](#1-폴더-구조)
2. [환경 설정](#2-환경-설정)
3. [실행 순서 (빠른 시작)](#3-실행-순서-빠른-시작)
4. [데이터 파이프라인](#4-데이터-파이프라인)
5. [모델 구조](#5-모델-구조)
6. [학습 (train.py)](#6-학습-trainpy)
7. [평가 (test.py)](#7-평가-testpy)
8. [데이터 통계](#8-데이터-통계)
9. [참고 사항 및 주의점](#9-참고-사항-및-주의점)

---

## 1. 폴더 구조

```
REC4MIT/
├── main.py                       # Rec4Mit 모델 조립 (레이어 1~3 연결)
├── train.py                      # 학습 스크립트 (fold별 체크포인트 저장)
├── test.py                       # 평가 스크립트 (REC / MRR / NDCG / RT)
│
├── Model/
│   ├── layer1.py                 # 뉴스 임베딩 레이어 (ID + BERT 메타 임베딩)
│   ├── layer2.py                 # 인코더 / 사건·진위 디코더 / 분리 손실
│   └── layer3.py                 # 사건 감지 / 사건 전이 / 다음 뉴스 예측
│
├── DataPreperation/
│   ├── dataPipeline.py           # 아래 3개 스크립트를 순서대로 실행하는 통합 파이프라인
│   ├── make_emb.py               # 뉴스 제목·본문 → BERT 임베딩 (.npz)
│   ├── make_interaction.py       # 뉴스 CSV → 사용자별 열람 시퀀스 (.json)
│   ├── interaction2instance.py   # 시퀀스 → 학습 인스턴스 + 10-fold 분할
│   └── datas/
│       ├── news/                 # 원본 뉴스 CSV (gossip.csv, pol.csv)  ※ git 제외
│       ├── emb/                  # BERT 임베딩 (gossip.npz, pol.npz)     ※ git 제외
│       ├── user_interaction/     # 사용자별 열람 시퀀스 JSON
│       └── folds/{data}/{0~9}/   # train.json / val.json / test.json     ※ git 제외
│
└── model_dir/{data}/fold_{n}.pth # 학습된 체크포인트                      ※ git 제외
```

> `.gitignore`에 의해 대용량 파일(CSV, 임베딩, fold, 체크포인트)은 저장소에 포함되지 않습니다.
> 저장소를 새로 받았다면 [4. 데이터 파이프라인](#4-데이터-파이프라인)을 따라 데이터를 직접 생성해야 합니다.

---

## 2. 환경 설정

### 요구 사항

| 패키지 | 검증된 버전 | 용도 |
|---|---|---|
| Python | 3.13 | |
| torch | 2.10.0 (cu128) | 모델 학습·추론 |
| numpy | 2.3.5 | 수치 연산 |
| pandas | 2.3.3 | CSV 처리 |
| sentence-transformers | 5.6.0 | BERT 임베딩 생성 (`make_emb.py`에서만 사용) |

### 설치

```bash
pip install torch numpy pandas sentence-transformers
```

GPU가 있으면 자동으로 CUDA를 사용하고, 없으면 CPU로 동작합니다.

---

## 3. 실행 순서 (빠른 시작)

모든 명령은 **저장소 루트**에서 실행합니다.

```bash
# ① 데이터 전처리 (통합 파이프라인, 3단계 순서대로 실행)
python DataPreperation/dataPipeline.py                       # gossip + pol, 전체 단계
python DataPreperation/dataPipeline.py --data pol            # pol 만
python DataPreperation/dataPipeline.py --steps interaction instance   # 임베딩 생략

# ② 학습
python train.py --data pol --fold_num 1     # fold 0 만 학습
python train.py --data pol --fold_num 10    # fold 0~9 전부 학습

# ③ 평가 (루트에서)
python test.py --data pol
python test.py --data gossip --batch 16     # gossip은 뉴스 풀이 커서 배치를 줄이는 것을 권장
```

---

## 4. 데이터 파이프라인

`dataPipeline.py` 하나로 아래 3단계를 순서대로 실행합니다. 개별 스크립트(`make_interaction.py`, `interaction2instance.py`, `make_emb.py`)도 그대로 남아 있으며, 이 셋은 `DataPreperation/` 안에서 실행해야 합니다.

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data` | `gossip pol` | 처리할 데이터셋 (여러 개 가능) |
| `--steps` | `interaction instance emb` | 실행할 단계 (여러 개 가능) |

### 4-1. 원본 데이터 형식 (`datas/news/{pol,gossip}.csv`)

| 컬럼 | 설명 |
|---|---|
| `news_id` | 뉴스 고유 ID (예: `politifact12418`, `gossipcop-893510`) |
| `title` | 뉴스 제목 |
| `description` | 뉴스 본문 요약 |
| `label` | `0` = 진짜, `1` = 가짜 |
| `user_ids` | 해당 뉴스를 공유한 사용자 ID 리스트 (문자열로 저장된 파이썬 리스트) |
| `user_times` | 각 사용자의 공유 시각 리스트 (`user_ids`와 순서 일치) |

### 4-2. `make_emb.py` — 뉴스 텍스트 임베딩

- 모델: `bert-base-uncased` (sentence-transformers)
- 제목은 최대 32 토큰, 본문은 최대 128 토큰으로 잘라서 인코딩
- 제목이 비어 있으면 `"unknown news"`로 대체
- 본문 길이가 5자 이하이면 0 벡터 처리
- 출력: `datas/emb/{data}.npz`

| 키 | 형태 | 설명 |
|---|---|---|
| `news_id` | `(N,)` | 뉴스 ID (CSV 행 순서와 동일) |
| `title` | `(N, 768)` | 제목 임베딩 (L2 정규화) |
| `description` | `(N, 768)` | 본문 임베딩 (L2 정규화) |

### 4-3. `make_interaction.py` — 사용자별 열람 시퀀스

CSV의 `user_ids`·`user_times`를 풀어서 **사용자 → 시간순 뉴스 ID 리스트**로 재구성합니다.

```json
{ "2363305464": ["politifact674", "politifact12418", ...], ... }
```

출력: `datas/user_interaction/{data}_user_interaction.json`

### 4-4. `interaction2instance.py` — 학습 인스턴스 및 10-fold 분할

각 사용자 시퀀스에서 **슬라이딩 윈도우**로 인스턴스를 만듭니다.

```
시퀀스: [n1, n2, n3, n4, n5, n6]
        → (ctx=[n1],             tgt=n2)
        → (ctx=[n1,n2],          tgt=n3)
        → (ctx=[n1,n2,n3],       tgt=n4)
        → (ctx=[n1,n2,n3,n4],    tgt=n5)
        → (ctx=[n2,n3,n4,n5],    tgt=n6)   # 컨텍스트는 최근 4개까지
```

인스턴스 하나의 형식: `[ctx(리스트), tgt(문자열), uid(문자열)]`

- 중복 인스턴스 제거 후 논문과 같은 개수로 자름 (pol 47,464 / gossip 136,004)
- 시드 3으로 섞은 뒤 10등분
- fold `k`: 청크 `k` = test, 청크 `(k+1)%10` = val, 나머지 8개 = train
- 출력: `datas/folds/{data}/{k}/{train,val,test}.json`

---

## 5. 모델 구조

`main.py`의 `Rec4Mit`이 아래 세 레이어를 순서대로 연결합니다.

```
뉴스 ID ──► [Layer 1] 임베딩 ──► [Layer 2] 인코더 ──┬─► 사건 표현 e (128)
                                                    └─► 진위 표현 l (128) ─► 가짜 확률 ỹ
                                                    
컨텍스트 e ──► [Layer 3] 사건 감지 ─► 사건 전이 R ─► 사용자 결합 c_u ─► 후보 점수
```

### Layer 1 — `Model/layer1.py` 임베딩

| 구성 | 설명 |
|---|---|
| `init_emb()` | `.npz`의 제목·본문 임베딩을 이어 붙여 `(N+1, 1536)` 행렬 생성. 0번 행은 패딩 |
| `EmbeddingLayer` | 학습 가능한 ID 임베딩(128) + BERT 메타 임베딩(1536, 사전값으로 초기화) → Linear → 뉴스 벡터 `v` (256) |

### Layer 2 — `Model/layer2.py` 분리(Disentangle)

| 구성 | 입력 → 출력 | 설명 |
|---|---|---|
| `Encoder` | 256 → 256 | 3단 Dense + skip-connection, LeakyReLU(0.1) |
| `EventDecoder` | 256 → 128 | 사건 표현 `e` |
| `VeracityDecoder` | 256 → 128 (+ 로짓 1) | 진위 표현 `l`과 가짜 여부 로짓 (Eq 8) |
| `DisentangleLoss` | | 아래 세 손실 계산 |

`DisentangleLoss`가 계산하는 항목:

| 손실 | 식 | 의미 |
|---|---|---|
| `L_r` | ½‖[e ; l] − v‖² | 재구성: 두 표현을 합치면 원래 뉴스 벡터가 복원돼야 함 (Eq 9) |
| `L_l` | BCE(로짓, y) | 진위 표현으로 가짜 여부를 맞춰야 함 (Eq 10) |
| `L_a` | 1 / BCE(Dense(e), y) | 적대 손실: 사건 표현 `e`로는 진위를 **못 맞춰야** 함 (Eq 11) |

> **현재 코드는 `L_l`만 최종 손실에 포함합니다.** `L_r`, `L_a`는 계산은 되지만 합산에서 제외되어 있습니다(`layer2.py`의 `losses = loss_l` 줄). 세 항을 모두 쓰려면 해당 줄을 `loss_r + loss_l + loss_a`로 바꾸면 됩니다.

### Layer 3 — `Model/layer3.py` 사건 전이 및 추천

| 구성 | 설명 |
|---|---|
| `EventDetector` | `e`를 K=20개 잠재 사건으로 소프트 분배 (β = softmax(W1·e)) → `e_split` `[B, L, K, 128]` (Eq 13~14) |
| `EventTransitionNet.build_R` | 위치 임베딩을 붙인 뒤 attention 가중치 γ로 컨텍스트를 합쳐 사건 전이 행렬 `R` `[B, K, 128]` 생성. 패딩 위치는 마스킹 (Eq 15~16) |
| `EventTransitionNet.activate` | 후보 뉴스 `e_t`와 `R` 사이 attention δ → 문맥 벡터 `c` → 사용자 임베딩과 결합해 `c_u` (Eq 17~19) |
| `NextNewsPredictor` | 점수 = `c_u · e_t` (내적) (Eq 20) |
| `NextNewsPredictor.recommend` | 가짜 확률이 임계값 이상인 후보를 제외하고 top-k 반환 (test.py에서는 같은 로직을 직접 구현) |

### 주요 하이퍼파라미터 (`Rec4Mit.__init__` 기본값)

| 인자 | 기본값 | 의미 |
|---|---|---|
| `k` | 20 | 잠재 사건 개수 |
| `ctx_len` | 4 | 컨텍스트 길이 (최근 뉴스 개수) |
| `v_dim` | 256 | 뉴스 벡터 차원 |
| `e_dim` | 128 | 사건 / 진위 표현 차원 |
| `user_dim` | 128 | 사용자 임베딩 차원 |

---

## 6. 학습 (`train.py`)

```bash
python train.py --data {pol|gossip} --fold_num N
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data` | `pol` | 데이터셋 선택 |
| `--fold_num` | `1` | **학습할 fold 개수**. `N`이면 fold `0 ~ N-1`을 순서대로 학습 |

### 동작 방식

1. `news.csv` 행 순서대로 뉴스 내부번호를 매김 (0은 패딩, 1부터 시작)
2. fold 0의 train/val/test 전체에서 사용자 집합을 만들어 정렬 후 인덱스 부여 → `u2i`
3. **정답 뉴스가 진짜(label 0)인 인스턴스만 학습에 사용**
4. 컨텍스트는 최근 4개, 부족하면 왼쪽 패딩 + 마스크
5. 후보 구성: 정답 1개 + 네거티브 64개 (진짜 32 + 가짜 32 무작위 추출)
6. 손실 = 추천 BCE (정답 후보만 1) + 컨텍스트 분리 손실 + 후보 분리 손실 (Eq 21~22)
7. Adam, lr 1e-3, batch 64, 15 epoch
8. 매 epoch 검증 손실을 계산해 **가장 낮을 때** `model_dir/{data}/fold_{k}.pth` 저장

> `best`는 fold 루프 바깥에서 한 번만 초기화됩니다. 여러 fold를 연속 학습할 때 이전 fold보다 검증 손실이 낮아져야 다음 fold의 체크포인트가 저장됩니다. fold마다 독립적으로 저장하려면 `best = float("inf")`를 fold 루프 안으로 옮기세요.

---

## 7. 평가 (`test.py`)

```bash
python test.py --data {pol|gossip} --batch 32
```

| 인자 | 기본값 | 설명 |
|---|---|---|
| `--data` | `pol` | 데이터셋 선택 |
| `--batch` | `32` | 평가 배치 크기. gossip은 후보 뉴스가 17,527개라 메모리에 맞춰 줄이는 것을 권장 |

### 동작 방식

1. `model_dir/{data}/fold_*.pth`를 모두 찾아 fold별로 평가
2. 후보 = **전체 뉴스 풀** (훈련 때와 달리 샘플링 없음)
3. 후보는 한 번만 인코딩해 재사용
4. 진위 디코더가 가짜라고 판단한(σ(로짓) ≥ 0.5) 후보는 점수에서 1e4를 빼서 사실상 제외
5. top-20 안에서 아래 지표를 K = 5, 10, 20에 대해 계산
6. fold 전체 평균 ± 표준편차 출력

### 지표

| 지표 | 의미 |
|---|---|
| `REC@K` | 정답 뉴스가 top-K 안에 있으면 1 (Hit Rate) |
| `MRR@K` | 정답 순위의 역수 |
| `NDCG@K` | 1 / log₂(순위 + 1) |
| `RT@K` | top-K 중 **진짜 뉴스 비율** (가짜뉴스 완화 효과 측정) |

출력 예시:

```
학습된 fold: [0]
fold0: REC@5 0.xxxx REC@20 0.xxxx | MRR@5 0.xxxx | NDCG@5 0.xxxx | RT@5 0.xxxx

=== [pol] 1-fold 평균 ± 표준편차 ===
REC@5    0.xxxx ± 0.0000
...
```

---

## 8. 데이터 통계

| 항목 | PolitiFact (`pol`) | GossipCop (`gossip`) |
|---|---|---|
| 뉴스 수 | 599 | 17,527 |
| 진짜 / 가짜 | 280 / 319 | 13,120 / 4,407 |
| 사용자 수 | 162,262 | 251,681 |
| 열람 기록 수 | 256,380 | 1,106,623 |
| 학습 인스턴스 (전체) | 47,464 | 136,004 |
| fold 하나의 train / val / test | 37,970 / 4,747 / 4,747 | 108,802 / 13,601 / 13,601 |

---

## 9. 참고 사항 및 주의점

- **뉴스 내부번호 규칙**: `news.csv` 행 순서 = 임베딩 행 순서 = 내부번호 − 1. 0번은 패딩입니다. CSV 순서를 바꾸면 임베딩과 어긋나므로 주의하세요.
- **사용자 인덱스 규칙**: 학습과 평가 모두 fold 0의 전체 사용자 집합을 정렬해 인덱스를 만듭니다. 결정적이므로 두 스크립트가 같은 `u2i`를 얻습니다.
- **정답이 가짜인 인스턴스 제외**: 학습·평가 모두 정답 뉴스가 진짜인 인스턴스만 사용합니다. 가짜뉴스는 추천 대상이 아니기 때문입니다.
- **전처리 스크립트 실행 위치**: `dataPipeline.py`는 어디서 실행해도 됩니다. 개별 스크립트 3개는 상대경로 `datas/...`를 사용하므로 `DataPreperation/` 안에서 실행해야 합니다.
- **`layer2.py`의 `from mpmath import sigmoid`**: 사용되지 않는 import입니다. `mpmath`가 없는 환경이면 지워도 됩니다.
- **OpenMP 중복 경고** (Windows): `OMP: Error #15`가 뜨면 환경변수 `KMP_DUPLICATE_LIB_OK=TRUE`를 설정하면 넘어갈 수 있습니다.
