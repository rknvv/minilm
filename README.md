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
