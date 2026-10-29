#!/usr/bin/env bash
# Run the evaluation with real hosted models from NVIDIA's API catalog (no model downloads).
#
#   export NVIDIA_API_KEY=nvapi-...      # free key from https://build.nvidia.com
#   bash scripts/run_with_nvidia.sh
#
# Detection: nvidia/gliner-pii (hosted) vs. regex, CRM dictionary and Presidio.
# Answers, judging and the attacker: a hosted LLM (default meta/llama-3.1-70b-instruct;
# override with NVIDIA_LLM_MODEL). The free tier is rate limited, so detection uses a random
# 400-document Gretel subset and 200 client notes; API results are cached in data/cache/.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ -z "${NVIDIA_API_KEY:-}" ]; then
  echo "Set NVIDIA_API_KEY first: export NVIDIA_API_KEY=nvapi-..."
  exit 1
fi

python3 -m pip install -q -r requirements.txt presidio-analyzer pyarrow requests
python3 -m spacy download en_core_web_sm -q || true
[ -f data/processed/acs_persons.csv.gz ] || python3 scripts/download_data.py

echo "Step 1 of 2: every metric, with hosted gliner-pii in the detection comparison"
python3 eval/run_real.py --gliner nvidia-api --detect-docs 400 --detect-notes 200

echo "Step 2 of 2: real model answers, LLM-judged quality and the LLM attacker"
python3 eval/llm_eval.py --backend nvidia --n 15

echo "Done. Results are in eval/results_real/ and figures in eval/figures_real/."
