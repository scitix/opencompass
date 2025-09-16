#!/usr/bin/env bash
set -euo pipefail

export AIDER_DOCKER=1
export AIDER_BENCHMARK_DIR="/benchmarks"
export OPENAI_API_BASE=""
export OPENAI_API_KEY=""

./benchmark/benchmark.py aider-polyglot \
  --model "openai/gpt-4o-2024-05-13" \
  --edit-format diff \
  --exercises-dir polyglot-benchmark \
  --num-tests -1 \
  --threads 8 \
  --new
