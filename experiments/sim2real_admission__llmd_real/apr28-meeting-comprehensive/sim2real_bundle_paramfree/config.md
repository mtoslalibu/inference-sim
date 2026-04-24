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

## BLIS Simulation

BLIS flags calibrated to match the real deployment above. See `run.sh` for full command.

| Real vLLM Parameter | BLIS Flag | Value |
|---|---|---|
| `block_size=64` | `--block-size-in-tokens` | 64 |
| `gpu_memory_utilization=0.95` | `--gpu-memory-utilization` | 0.95 |
| `max_num_batched_tokens=2048` | `--max-num-scheduled-tokens` | 2048 |
| `max_num_seqs=256` | `--max-num-running-reqs` | 256 |
| `max_seq_len=40960` | `--max-model-len` | 40960 |
| `tensor_parallel_size=1` | `--tp` | 1 |
| H100-SXM-80GB | `--hardware` | H100 |
| Qwen/Qwen3-14B | `--model` | Qwen/Qwen3-14B |
| 4 instances | `--num-instances` | 4 |

### BLIS-Only Parameters

These have no real deployment equivalent — they control the simulator itself.

| BLIS Flag | Value | Notes |
|---|---|---|
| `--latency-model` | trained-physics | Roofline basis functions with learned corrections |
| `--routing-policy` | round-robin | |
| `--snapshot-refresh-interval` | 50000 (50ms) | Staleness of routing signals |
| `--total-kv-blocks` | 4719 | Pinned to real vLLM value (302,016 tokens / 64 block_size). BLIS auto-calc gives 4387 — 7% gap due to memory accounting differences. |

### Load Generator (`blis observe`)

| Parameter | Value | Notes |
|---|---|---|
| `--max-concurrency` | 3000 | **Required.** Default 256 is too low — causes client-side queueing that throttles actual server QPS. Sized for worst case: 140 QPS x 20s p99 E2E = 2,800 concurrent. Pod ulimit is ~1M file descriptors, so 3000 is safe. |

### Notes

- `enable_prefix_caching` set to False for these experiments — BLIS uses round-robin routing with no prefix scorer, so prefix caching has no effect.
- `enable_chunked_prefill=True` is modeled via `--max-num-scheduled-tokens 2048`.
