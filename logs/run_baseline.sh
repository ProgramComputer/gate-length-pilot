#!/bin/bash
cd /root/pilot
while pgrep -f "[t]rain.py --seed [012] " > /dev/null; do sleep 10; done
for s in 0 1 2; do
  /root/venv/bin/python evaluate.py --seed $s --baseline-only > eval_baseline_seed$s.log 2>&1 &
done
wait
echo BASELINE_DONE
