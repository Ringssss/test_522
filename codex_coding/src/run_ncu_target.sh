#!/bin/bash
# Wrapper for ncu profiling - sets PYTHONPATH internally
export PYTHONPATH=/home/wuhang/wuhang/dllm_wh/lib_cite/dInfer/python:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
exec /home/wuhang/miniconda3/envs/dllm/bin/python /home/wuhang/wuhang/dllm_wh/codex_coding/src/bench_ncu_fused_experts.py "$@"
