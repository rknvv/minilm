# MiniLM

Рукописная реализация Gemma-3-1B на PyTorch для continued pretraining (CPT) на русском+английском с pruned-словарём (262k → 154k: frequency+merge-closure прунинг, размер кратен 256 под tensor-core GEMM). Цель проекта — пощупать претрейн руками и выжать максимум скорости из 2×H100.

Архитектура (Gemma-3): RMSNorm-сэндвич, QK-norm, RoPE (global 1M / local 10k), GeGLU MLP, GQA, чередование sliding-window (512) и global attention 5:1, weight tying. Совпадение логитов с HF `Gemma3ForCausalLM` проверяется скриптом `scripts/verify_gemma.py`.

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

- **FlexAttention** на local-слоях (block-sparse sliding window), `is_causal` flash-путь на global-слоях, `enable_gqa` без материализации KV-голов
- **Liger fused linear cross-entropy** (логиты 184k-словаря не материализуются); fallback — chunked CE
- **Muon (dion)** для скрытых матриц + AdamW для embeddings/head/скаляров, раздельные lr
- `torch.compile` поверх DDP (DDPOptimizer оверлапит allreduce с backward), bf16 autocast
- Метрики в логах/wandb: tokens/s, ms/step, MFU, peak memory