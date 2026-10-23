#!/bin/bash
cd /root/pilot
export HF_HUB_DISABLE_PROGRESS_BARS=1
/root/venv/bin/python pilot6.py --stage full > p6_full.log 2>&1
echo FULL_DONE
