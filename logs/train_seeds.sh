#!/bin/bash
cd /root/pilot
for s in 0 1 2; do
  /root/venv/bin/python train.py --seed $s --steps 60000 --curriculum 0.5 --eval-every 2000 --out runs > train_seed$s.log 2>&1 &
done
wait
