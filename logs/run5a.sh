#!/bin/bash
cd /root/pilot
export HF_HUB_DISABLE_PROGRESS_BARS=1
/root/venv/bin/python pilot5.py --stage verify > p5_verify.log 2>&1
/root/venv/bin/python pilot5.py --stage feasibility > p5_feasibility.log 2>&1
echo STAGE_A_DONE
