#!/bin/bash
cd /root/pilot
P=/root/venv/bin/python
for s in 0 1 2; do
  $P train.py --seed $s --pos rope --steps 60000 --curriculum 0.5 --eval-every 2000 --out runs_rope > train_rope_seed$s.log 2>&1 &
done
for s in 0 1 2; do
  $P pilot2.py --seed $s > pilot2_nope_seed$s.log 2>&1 &
done
wait
for s in 0 1 2; do
  $P pilot2.py --seed $s --runs runs_rope > pilot2_rope_seed$s.log 2>&1 &
done
wait
echo PILOT2_DONE
