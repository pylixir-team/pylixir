
Name | Algorithm Group | sum14↑ (%) | sum16 | sum18 | Registered | Note | insight
-- | -- | -- | -- | -- | -- | -- | --
Elphago | ML | 18.6 | 7.78 | 2.69 | August 26, 2023 12:12 AM | Previous SOTA |  
transformer-L3-H4-Emb128-lrdecay3e-4-lock-long-ffnx4 | DQN | 34.37 | 12.07 | 2.65 | August 1, 2023 2:31 PM | 3 layer, 6000k epoch (total 34h) |  
transformer-L3-H4-Emb128-lrdecay3e-4-lock | DQN | 33.28 | 11.71 | 2.52 | July 25, 2023 11:54 PM | 3-Layer transformer encoder, 4 Head, step 2000k, LR decat(exponential to 3e-5, warm-up 200k), LOCK embedding 추가 | apply lock embedding
transformer-L6-H4-Emb128-lrdecay3e-4 | DQN | 33.44 | 11.58 | 2.44 | July 24, 2023 2:21 AM | 6-layer | 6L>3L
transformer-L3-H4-Emb128-lrdecay3e-4-ffn512 | DQN | 33.42 | 11.86 | 2.42 | July 24, 2023 2:22 AM | 3-Layer transformer encoder, 4 head, step 2000k, LR decat(exponential to 3e-5, warm-up 200k), FFN hiddendim 512 | FFNx4 is better than x2
transformer-L3-H4-Emb128-lrdecay3e-4-simplePolicy | DQN | 31.6 | 10.87 | 2.21 | July 25, 2023 11:55 PM | FeatureExtractor는 Transformer, Policy는 MLP | complex policy is better
transformer-L3-H8-Emb128-lrdecay3e-4 | DQN | 30.9 | 10.8 | 2.19 | July 22, 2023 8:50 PM | 3-Layer transformer encoder, 8 Head, step 2000k, LR decat(exponential to 3e-5, warm-up 200k) | 4 head is better than 8 head
transformer-L3-H4-Emb128-lrdecay3e-4 | DQN | 33.84 | 11.46 | 2.16 | July 22, 2023 8:50 PM | 3-Layer transformer encoder, 4 Head, step 2000k, LR decat(exponential to 3e-5, warm-up 200k) |  
transformer-L2-H4-Emb128-lr3e-4 | DQN | 31.95 | 10.88 | 2.1 | July 22, 2023 8:21 PM | 3-Layer transformer encoder, 4 Head, step 2000k |  
transformer-L2-H4-Emb128-lr3e-4 | DQN | 32.64 | 11 | 2.05 | July 22, 2023 11:03 PM | 3-Layer transformer encoder, 4 Head, step 2600k |  
DQN-[obs-space]-lr3e-4-b128, embedding size=256 | DQN | 27.92 | 8.59 | 1.59 | July 30, 2023 10:45 PM | commit hash #https://github.com/pylixir-team/pylixir/pull/25/commits/a27ac4627bb58fe11718a5c1631b08fb8a34801d |  
DQN-[obs-space]-lr3e-4-b128, embedding size=128 | DQN | 26.95 | 8.29 | 1.57 | July 21, 2023 1:54 AM | commit hash #https://github.com/pylixir-team/pylixir/pull/25/commits/a27ac4627bb58fe11718a5c1631b08fb8a34801d |  
DQN-[obs-space]-lr1e-4-b128, embedding size=128 | DQN | 25.56 | 7.8 | 1.28 | July 21, 2023 1:37 AM | commit hash #https://github.com/pylixir-team/pylixir/pull/25/commits/a27ac4627bb58fe11718a5c1631b08fb8a34801d |  
DDQN-[obs-space]-lr3e-4-b128, embedding size=256 | DDQN, DQN | 25.51 | 6.8 | 1.17 | August 1, 2023 12:43 PM | commit hash #https://github.com/pylixir-team/pylixir/pull/25/commits/a27ac4627bb58fe11718a5c1631b08fb8a34801d에 double DQN만 적용 $\tau = 0.5$ (”DDQN”) | $\tau = 1$, target net update freq no improvement
DQN-dict-exponential-lesspen-large_batch-[128,128]-fixfloatembedding-grouped | DQN | 23.34 | 6.97 | 1.14 | July 21, 2023 12:45 AM | Model: [128, 128] Float embedding을 5개 float에 대해 동일한 연산을 하도록 변경 committee, board도 동일 속성에 대해 동일한 embedding이 적용되도록 변경 |  
DQN-dict-exponential-lesspen-large_batch-[128,128]-fixfloatembedding | DQN | 23.05 | 6.31 | 1.09 | July 19, 2023 7:44 PM | Model: [128, 128] Float embedding을 5개 float에 대해 동일한 연산을 하도록 변경 |  
DQN-dict-exponential-lesspen-large_batch-[128,128] | DQN | 18.32 | 5.08 | 0.69 | July 30, 2023 10:47 PM | Model: [128, 128] |  
DQN-dict-exponential-lesspen-large_batch-3.4M | DQN | 17.41 | 4.81 | 0.549 | July 30, 2023 10:47 PM | Batch size = 512 (very large batch) 3400k steps |  
PPO-baseline | PPO | 15.22 | 4.12 | 0.53 | July 14, 2023 3:03 AM | PPO, lr 3e-4 batch size 128 network 128,128 exp/less penalty reward |  
DQN-dict-exponential-lesspenalty | DQN | 15.1 | 3.65 | 0.52 | July 13, 2023 2:02 AM | exponential reward negative reward는 1/3로 책정 |  
DQN-dict-exponential-lesspen-large_batch | DQN | 16.8 | 4 | 0.5 | July 13, 2023 9:27 PM | Batch size = 512 (very large batch) 1500k steps |  
DQN-dict-exponential | DQN | 14.14 | 2.89 | 0.4 | July 12, 2023 11:59 PM | exponential reward ( 2 **first + 2 ** second) |  
DQN-dict-baseline | DQN | 13.8 | 2.98 | 0.35 | July 13, 2023 12:19 AM | reroll Available No stop when wrong action |  
  |   |   |   |   | July 25, 2023 7:17 PM |   |  
  |   |   |   |   | July 22, 2023 8:49 PM |   |  
DQN | DQN | 10.8 |   |   | July 9, 2023 4:21 PM | DQN_3 (DQN_1:5e5) |  
DQN-baseline | DQN | 8.8 |   |   | July 8, 2023 4:23 PM | [128, 128], $\tau$=0.5 (”DQN_2”) |  
