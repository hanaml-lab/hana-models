#!/usr/bin/env bash
# Train one stage from a recipe on every GPU of this machine.
#   bash scripts/train.sh recipes/sft-qwen3-8b.yaml
set -euo pipefail
recipe="${1:?usage: train.sh <recipe.yaml>}"
gpus="${GPUS:-$(nvidia-smi --list-gpus | wc -l)}"
python -c "from distill_kit.train import load_recipe; load_recipe('$recipe')"   # fail fast on a bad recipe
exec accelerate launch --num_processes "$gpus" --mixed_precision bf16 -m distill_kit.train "$recipe"
