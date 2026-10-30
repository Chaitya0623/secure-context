#!/usr/bin/env bash
# Hosted Groq evaluation on public benchmark data. Rerun to use cached responses.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${GROQ_API_KEY:?Set GROQ_API_KEY in your Terminal first}"
export PYTHONUNBUFFERED=1
python3 -m pip install -q -r requirements.txt presidio-analyzer pyarrow requests
python3 -c 'import spacy; spacy.load("en_core_web_sm")' || python3 -m spacy download en_core_web_sm
[ -f data/processed/acs_persons.csv.gz ] || python3 scripts/download_data.py
# Fail early on unavailable models or credentials, before overwriting results.
python3 - <<'PY'
import os
from secure_context.llm import GroqLLM
for model in {os.environ.get('GROQ_LLM_MODEL', 'openai/gpt-oss-120b'),
              os.environ.get('GROQ_DETECT_MODEL', 'openai/gpt-oss-20b')}:
    llm = GroqLLM(model)
    llm.complete('Reply briefly.', 'Reply with OK.')
    print('Connected:', llm.name)
PY
# Preserve previous results before this run writes new metrics.
python3 - <<'PY'
from pathlib import Path
from datetime import datetime
import shutil
root = Path('data/cache/run_backups') / datetime.now().strftime('%Y%m%dT%H%M%S%f')
for name in ('results_real', 'figures_real'):
    source = Path('eval') / name
    if source.exists():
        shutil.copytree(source, root / name)
print('Previous outputs saved in', root)
PY
echo 'Step 1 of 2: public-data metrics with Groq LLM PII detection'
python3 eval/run_real.py --gliner groq-llm --detect-docs 120 --detect-notes 50
echo 'Step 2 of 2: answers, grading and attacker evaluation'
python3 eval/llm_eval.py --backend groq --n 8
echo 'Done. Results: eval/results_real/. Figures: eval/figures_real/.'
