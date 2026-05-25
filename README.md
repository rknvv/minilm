# MiniLM

Decoder-only языковая модель на ~250M параметров для русского языка, написанная с нуля на PyTorch.

Архитектура вдохновлена LLaMA: RMSNorm, RoPE, SwiGLU FFN, Grouped-Query Attention, weight tying. Поддерживает pretrain на сырых токенах и SFT в чат-формате.

## Архитектура

| Параметр | Значение |
|---|---|
| Размерность (`dim`) | 768 |
| Слои (`n_layers`) | 12 |
| Головы внимания (`n_heads`) | 12 |
| KV-головы (`n_kv_heads`) | 6 |
| Словарь (`vocab_size`) | 32 000 |
| Макс. длина (`max_seq_len`) | 1 024 |
| FFN | SwiGLU |
| Позиционное кодирование | RoPE |
| Нормализация | RMSNorm |

## Датасет

Строго русскоязычный претрейн-корпус, собранный и сбалансированный вручную из открытых источников:

| Источник | Описание | Размер |
|---|---|---|
| CulturaX | Мультиязычный веб-корпус (ru subset) | 25,3 ГБ |
| Wikipedia | Русская Википедия | 4,7 ГБ |
| Librusec | Художественная литература | 8,2 ГБ |
| Proza | Проза (proza.ru) | 8,1 ГБ |
| WanJuan (1–5) | Китайский открытый корпус (ru части) | ~4,3 ГБ |
| Conversations | Диалоговые данные | 1,6 ГБ |
| Lenta-ru | Новости Lenta.ru | 542 МБ |
| Telegram News | Новостные телеграм-каналы | 39 МБ |
| RU News | Русскоязычные новости | 3,9 МБ |
| WanJuan Books | Книги | 1,0 ГБ |
| Math | Математические тексты | 58 МБ |

Итого ~12B токенов (~48 токенов на параметр для 250M модели).

Токенизатор — SentencePiece BPE, 32k словарь, обученный на этом же корпусе.

## Веса модели

> **Pretrained checkpoint:** [`ссылка`](#) *(TODO: добавить ссылку)*

## Результаты обучения

Pretrain ~10k шагов.

- **Train loss:** ~2.7 (финальный)
- **Perplexity:** ~16 (eval)

<p align="center">
  <img src="assets/train_loss.png" width="420"/>
  <img src="assets/train_lr.png" width="420"/>
</p>

LR schedule — cosine с warmup ~500 шагов, пиковый LR = 4e-4.

## Структура проекта

```
├── main.py              # Точка входа: конфиг, DDP, запуск обучения
├── run_ddp.sh           # Скрипт запуска multi-GPU
├── requirements.txt
├── src/
│   ├── model.py         # MiniLM: трансформер, генерация (top-k/top-p)
│   ├── trainer.py       # Тренировочный цикл, eval, чекпоинтинг, wandb
│   ├── dataset.py       # MemmapDataset (pretrain) + SFTDataset (chat)
│   ├── tokenizer.py     # SentencePiece обёртка с чат-темплейтом
│   ├── checkpoint.py    # Утилиты для state_dict (нормализация ключей)
│   └── config.py        # ModelArgs + TrainConfig (dataclasses)
├── configs/             # YAML-конфиги для обучения
└── utils/               # Вспомогательные скрипты
```

## Быстрый старт

### Установка

```bash
pip install -r requirements.txt
```

### Конфиг

Обучение запускается через YAML-конфиг. Пример `configs/pretrain.yaml`:

```yaml
model:
  dim: 768
  n_layers: 12
  n_heads: 12
  n_kv_heads: 6
  vocab_size: 32000
  max_seq_len: 1024
  dropout: 0.1

train:
  task: pretrain
  dataset_dir: ./data/pretrain    # должны лежать train.bin и val.bin
  out_dir: ./out
  train_batch_size: 32
  gradient_accumulation_steps: 4
  learning_rate: 4e-4
  warmup_iters: 500
  max_iters: 10000
  eval_interval: 500
  wandb_log: true
  wandb_project: minilm-pretrain
```

### Pretrain

```bash
# single GPU
python main.py --yaml_path configs/pretrain.yaml

# multi-GPU (DDP)
bash run_ddp.sh
# или напрямую:
torchrun --nproc_per_node=4 main.py --yaml_path configs/pretrain.yaml
```

### SFT

Для файнтюна нужен JSONL с чат-сообщениями и обученный SentencePiece-токенизатор:

```yaml
train:
  task: sft
  train_data_path: ./data/sft/train.jsonl
  eval_data_path: ./data/sft/val.jsonl
  tokenizer_path: ./tokenizer.model
  pretrained_checkpoint: ./out/ckpt_best.pt
```

Формат данных — JSONL, каждая строка:

```json
{"messages": [{"role": "user", "content": "Привет!"}, {"role": "assistant", "content": "Привет! Чем могу помочь?"}]}
```

### Генерация

```python
import torch
from src.model import MiniLM
from src.config import ModelArgs
from src.tokenizer import Tokenizer

args = ModelArgs(vocab_size=32000, max_seq_len=1024)
model = MiniLM(args)

checkpoint = torch.load("out/ckpt_best.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

tokenizer = Tokenizer("tokenizer.model")

messages = [{"role": "user", "content": "Расскажи о Python"}]
input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")

output = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_id=tokenizer.eos_id,
    pad_id=tokenizer.pad_id if tokenizer.pad_id >= 0 else 0,
    temperature=0.7,
    top_p=0.9,
)

print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Фичи

- **DDP** — мультигипушный тренинг из коробки через `torchrun`
- **Gradient accumulation** — эмуляция больших батчей
- **Mixed precision** — AMP с bfloat16/float16 + GradScaler
- **torch.compile** — опциональная компиляция модели
- **Чекпоинтинг** — автосохранение latest + best, resume из любого
- **WandB** — логирование train/eval loss, LR, tokens seen
- **KV-cache** — для быстрого авторегрессивного инференса

## Лицензия

MIT