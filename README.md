# MiniLM

Рукописная реализация Gemma-3-1B на PyTorch для continued pretraining на русском + английском с пруненым-словарём 262k -> 154k. Цель проекта — выжать максимум скорости из H100.

Архитектура (Gemma-3): RMSNorm-сэндвич, QK-norm, RoPE (global 1M / local 10k), GeGLU MLP, GQA, чередование sliding-window (512) и global attention 5:1, weight tying.

## Архитектура

| Параметр | Значение |
|---|---|
| Размерность (`dim`) | 1152 |
| Слои (`n_layers`) | 26 |
| Головы внимания (`n_heads`) | 4 |
| KV-головы (`n_kv_heads`) | 1 |
| `head_dim` | 256 |
| Словарь (`vocab_size`) | 153 856 (601×256) |
| Макс. длина (`max_seq_len`) | 2 048 |
| Sliding window / pattern | 512 / 6 |

## Оптимизации

- **FlexAttention**
- **Liger fused linear cross-entropy**
- **Muon (dion)**
- `torch.compile`
- Метрики в логах/wandb: ток/s, ms/step, MFU