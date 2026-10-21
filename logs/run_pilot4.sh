#!/bin/bash
cd /root/pilot
P=/root/venv/bin/python
for s in 3 4 5; do
  $P train.py --seed $s --steps 60000 --curriculum 0.5 --eval-every 2000 --out runs > train_seed$s.log 2>&1 &
done
for s in 0 1 2; do
  $P pilot4.py --seed $s > pilot4_seed$s.log 2>&1 &
done
wait
for s in 3 4 5; do
  $P pilot4.py --seed $s > pilot4_seed$s.log 2>&1 &
done
wait
echo PILOT4_DONE
