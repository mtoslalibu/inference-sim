# Configuration: Real Deployment vs BLIS Simulation

## Real Deployment (llm-d / vLLM)

Extracted from real deployment logs:
`experiments/sim2real_admission_evolution/old/real-deployment-after-feedback/adminctrl1/deploy_baseline_log/w1_14b/sim2real-adminctrl1-decode-864b44dfc9-88czq_vllm.log`

```
non-default args: {'block_size': 64, 'gpu_memory_utilization': 0.95, 'max_num_batched_tokens': 2048, 'max_num_seqs': 256}
max_seq_len=40960, dtype=torch.bfloat16, tensor_parallel_size=1, enable_prefix_caching=True, enable_chunked_prefill=True
```

| vLLM Parameter | Value | Notes |
|---|---|---|
| `block_size` | 64 | |
| `gpu_memory_utilization` | 0.95 | |
| `max_num_batched_tokens` | 2048 | |
| `max_num_seqs` | 256 | |
| `max_model_len` | 40960 | |
| `tensor_parallel_size` | 1 | |
| `enable_prefix_caching` | False | set to False for these experiments (round-robin routing, no prefix scorer) |
| `enable_chunked_prefill` | True | |
| Model | Qwen/Qwen3-14B | |
| GPU | H100-SXM-80GB | |
| Instances | 4 | |

### GAIE Admission Thresholds (Production Defaults)

| Parameter | Value | Source |
|---|---|---|
| QD threshold | 5 | `saturationdetector/utilization/config.go:31` (`DefaultQueueDepthThreshold`) |
| KV threshold | 0.8 | `saturationdetector/utilization/config.go:33` (`DefaultKVCacheUtilThreshold`) |

### Load Generator (`blis observe` — use BLIS v0.7.10 image)

| Parameter | Value | Notes |
|---|---|---|
| `--max-concurrency` | 10000 | **Required.** Default 256 is too low — causes client-side queueing that throttles actual server QPS. High value ensures true open-loop arrival. |
| `--warmup-requests` | 200 | Discard initial requests to avoid cold-start artifacts (KV cache empty, scheduler not warmed up). |
| `--timeout` | 900 | Per-request HTTP timeout (15 min). Prevents premature timeouts for long-context requests under overload. |

### Notes

- `enable_prefix_caching` set to False for these experiments — BLIS uses round-robin routing with no prefix scorer, so prefix caching has no effect.
- `enable_chunked_prefill=True` is modeled via `--max-num-scheduled-tokens 2048`.
