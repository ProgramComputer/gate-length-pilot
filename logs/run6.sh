#!/bin/bash
cd /root/pilot
/root/venv/bin/pip install -q torch==2.14.1 numpy safetensors tokenizers > pip.log 2>&1
while pgrep -f "^/root/venv/bin/python /root/dl.py" > /dev/null; do sleep 5; done
export HF_HUB_DISABLE_PROGRESS_BARS=1
/root/venv/bin/python pilot6.py --stage tune > p6_tune.log 2>&1 && /root/venv/bin/python pilot6.py --stage feasibility > p6_feasibility.log 2>&1
echo STAGES_DONE
