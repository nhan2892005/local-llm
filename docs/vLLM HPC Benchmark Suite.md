# vLLM HPC Benchmark Suite

This suite is designed for the proven deployment:

- Model: `Qwen/Qwen2.5-72B-Instruct`
- Served model: `qwen2.5-72b`
- API: `http://10.1.1.238:50380`
- 2 nodes × 4 Tesla V100-SXM2-32GB
- TP=4, PP=2
- Current server: `max_model_len=2048`, `max_num_seqs=4`

## Files

- `run_vllm_benchmark_suite.sh` — benchmark orchestrator.
- `collect_vllm_metrics.py` — polls `/metrics` every second into CSV.
- `summarize_benchmarks.py` — exports one consolidated `summary.csv` and `summary.md`.
- `gpu_telemetry.sh` — optional `nvidia-smi` sidecar for the next server run.

## Install into the project

```bash
cd ~/llm_serving
mkdir -p benchmarks/suite
cp /path/to/vllm_hpc_benchmark_suite/* benchmarks/suite/
chmod +x benchmarks/suite/*.sh benchmarks/suite/*.py
```

## Preflight

The server must already be running:

```bash
curl -fsS http://10.1.1.238:50380/health
curl -sS http://10.1.1.238:50380/v1/models | python3 -m json.tool
curl -sS http://10.1.1.238:50380/metrics | grep '^vllm:' | head -30
```

## Basic suite

```bash
cd ~/llm_serving
SUITE_LEVEL=basic ./benchmarks/suite/run_vllm_benchmark_suite.sh
```

Runs:

1. Tiny correctness load: c=1, input=32, output=16.
2. Single-request baseline: c=1, input=128, output=64.
3. Normal capacity baseline: c=4, input=512, output=128.

## Full suite

```bash
SUITE_LEVEL=full ./benchmarks/suite/run_vllm_benchmark_suite.sh
```

Adds:

- concurrency 2 / 8 / 16;
- prefill-heavy workload: 1536 input / 64 output;
- decode-heavy workload: 256 input / 512 output.

## Stress suite

```bash
SUITE_LEVEL=stress ./benchmarks/suite/run_vllm_benchmark_suite.sh
```

Adds:

- concurrency 32 saturation;
- long balanced workload: 1024 input / 512 output;
- open-loop RPS sweep at 25%, 50%, 75%, 100%, 125% of the measured c=4 closed-loop request throughput.

Optional 5-minute nominal soak:

```bash
RUN_SOAK=1 SOAK_SECONDS=300 SUITE_LEVEL=stress \
  ./benchmarks/suite/run_vllm_benchmark_suite.sh
```

## Exported client metrics

The vLLM 0.8.5 serving benchmark produces:

- successful requests;
- benchmark duration;
- input tokens;
- generated tokens;
- request throughput;
- output-token throughput;
- total-token throughput;
- TTFT;
- TPOT;
- ITL;
- E2E request latency;
- P50 / P90 / P95 / P99 percentiles;
- detailed per-request arrays in `client.json`.

## Exported server metrics

`collect_vllm_metrics.py` stores all `vllm:*` Prometheus metrics every second.

The consolidated summary extracts:

- peak running requests;
- peak waiting requests;
- peak/mean reported GPU KV-cache usage;
- preemption delta;
- prompt-token counter delta;
- generation-token counter delta.

Raw Prometheus snapshots before and after each test are preserved.

## Important interpretation

The current server was launched with:

```text
max_num_seqs=4
```

Therefore:

- concurrency 1–4 measures operation at or below the configured active-sequence limit;
- concurrency 8/16/32 intentionally measures queueing and saturation;
- a high `vllm:num_requests_waiting` during c=8/16/32 is expected and is useful evidence.

Do not interpret c=32 as “32 sequences execute simultaneously”; with the current server configuration it primarily stresses queueing, scheduling, TTFT and E2E latency.

## GPU utilization / power / VRAM telemetry

The current already-running Slurm jobs cannot be retroactively given a clean per-node `nvidia-smi` sidecar without introducing another execution mechanism.

For the next controlled server run, source `gpu_telemetry.sh` from both head and worker batch scripts after `SESSION` is defined:

```bash
source "$ROOT/benchmarks/suite/gpu_telemetry.sh"
start_gpu_telemetry "$SESSION"
trap 'stop_gpu_telemetry' EXIT
```

This writes one CSV per node:

```text
gpu-gpunode1.csv
gpu-gpunode3.csv
```

with 1-second samples for:

- GPU utilization;
- memory utilization;
- VRAM used/total;
- power draw;
- temperature;
- SM clocks.

## Result layout

```text
benchmarks/results/YYYYMMDD-HHMMSS/
  suite.env
  runtime_versions.txt
  models.json
  metrics_initial.prom
  metrics_final.prom
  summary.csv
  summary.md

  00_smoke_c1_i32_o16/
    client.json
    client.log
    metrics_before.prom
    metrics_after.prom
    metrics_timeseries.csv
    PASS

  ...
```

## Archive/export

```bash
RESULT=~/llm_serving/benchmarks/results/YYYYMMDD-HHMMSS

tar -C "$(dirname "$RESULT")" \
  -czf "$RESULT.tar.gz" \
  "$(basename "$RESULT")"
```

The primary files for a report are:

```text
summary.csv
summary.md
runtime_versions.txt
suite.env
models.json
client.json files
metrics_timeseries.csv files
vllm_server_tail.log
```
