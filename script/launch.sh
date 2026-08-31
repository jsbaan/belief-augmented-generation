#!/bin/bash
# Submit one run_pipeline_single.job per model, all running in parallel.
# Usage: bash script/launch.sh

MODEL_NAMES=(
  "olmo2-7b-instruct"
  "olmo2-13b-instruct"
  "olmo3-7b-instruct"
  "qwen3-8b"
  "qwen3-14b"
  "gemini-2.5-flash"
  "olmo3-7b-think"
  "qwen3-8b-think"
)

DIRECT_PROMPTS=(
  "vanilla"
  "concise"
  "sentence"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for DIRECT_PROMPT in "${DIRECT_PROMPTS[@]}"; do
    JOB_ID=$(sbatch --parsable \
      --job-name="pipeline_${MODEL_NAME}_${DIRECT_PROMPT}" \
      --export=ALL,MODEL_NAME="${MODEL_NAME}",DIRECT_PROMPT="${DIRECT_PROMPT}" \
      script/single_pipeline_run.job)
    echo "Submitted job ${JOB_ID} for model: ${MODEL_NAME}, direct_prompt: ${DIRECT_PROMPT}"
  done
done
