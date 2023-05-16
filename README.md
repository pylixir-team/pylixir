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
