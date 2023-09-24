Pylixir: python Lost Ark Elixir simulator
============


pylixir는 python environment 내에서 Elixir 게임을 시뮬레이션합니다.

Install
=========

해당 라이브러리는 의존성 관리를 위해 `poetry` 를 사용합니다. 
[poetry home](https://python-poetry.org/docs/) 에서 poetry를 설치해 주세요.

```bash
git clone https://github.com/oleneyl/pylixir
poetry install
```

Run in terminal
==========
```bash
poetry run python cli.py
```

Usage
=======

```python
from pylixir.interface.cli import ClientBuilder

client_builder = ClientBuilder()
client = client_builder.get_client(seed)

print(client.view())
client.pick(sage_index=0, effect_index=3)

print(client.view())
```

CLI
=====
본 프로젝트는 command line interface를 통해 간편하게 동작시킬 수 있습니다.
실행 이후 [대상 현자 인덱스] [대상 옵션 인덱스] 를 0부터 시작하는 인덱스로 입력합니다.
선택이 불가능한 현자는 옵션 인덱스로 0을 입력합니다.

ex. 2번째 현자를 5번째 옵션을 선택하려고 할 경우: 1 4 입력
ex. 3번째 현자를 선택하려고 할 경우: 2 0 입력


```sh
poetry run python cli.py

>>>
0: [0]   20.00% | [10%]
1: [0]   20.00% | [10%]
2: [0]   20.00% | [10%]
3: [0]   20.00% | [10%]
4: [0]   20.00% | [10%]
Turn left: 13 | reroll left: 2

[______] | 이번 연성에서 <{3}> 효과가 연성될 확률을 <35>% 올려주지.
[______] | 이번에 연성되는 효과는 <3>단계 올라갈 걸세. 대신, 기회를 <2>회 소모하겠네.
[______] | 이번에는 <{3}> 효과를 연성해드리죠.
    
>>> 0 2

0: [0] _,_,1,_,_,2,_,3,4,5  20.00% | [10%]
1: [0] _,_,1,_,_,2,_,3,4,5  20.00% | [10%]
2: [0] _,_,1,_,_,2,_,3,4,5  20.00% | [10%]
3: [1] X]_,1,_,_,2,_,3,4,5  20.00% | [10%]
4: [0] _,_,1,_,_,2,_,3,4,5  20.00% | [10%]
Turn left: 12 | reroll left: 2

[O__]    | 남은 모든 연성에서 <네가 고르는> 효과가 연성될 확률을 <5>% 올려주지.
[X_____] | 남은 모든 연성에서 <모든> 효과의 대성공 확률을 <5>% 올리겠네.
[X_____] | 이번 연성에서 <{4}> 효과가 연성될 확률을 <35>% 올려드리죠.
```

Deep Learning
==============

pylixir의 deep 디렉토리에는 pylixir 게임을 Deep RL로 풀이한 모델이 구현되어 있습니다.
본 프로젝트에서는 5개 선택지중 1, 2번째 옵션의 강화를 최대화 하는 알고리즘을 학습했습니다.
최적의 알고리즘은 Transformer architecture에 기반한 DQN 알고리즘으로, 53이상 달성 확률 2.65%를 달성했습니다.

- Model Size는 14M으로, serverless하게 배포 가능한 규모입니다.

[Benchmark](benchmark.md)

Train with best configuration
===========

```sh
poetry run python deep/stable_baselines/train.py deep/conf/dqn_transformer.yaml
```

Evaluate
===========
```sh
poetry run python deep/stable_baselines/evaluate.py $ZIP_FILE_PATH
```


