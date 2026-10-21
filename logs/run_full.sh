#!/bin/bash
cd /root/pilot
for s in 0 1 2; do
  /root/venv/bin/python evaluate.py --seed $s > eval_full_seed$s.log 2>&1 &
done
wait
echo FULL_DONE
