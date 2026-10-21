#!/bin/bash
cd /root/pilot
P=/root/venv/bin/python
$P train.py --seed 97 --steps 60000 --curriculum 0.5 --eval-every 2000 --out probe/cur > probe_cur.log 2>&1 &
$P train.py --seed 96 --steps 100000 --eval-every 4000 --out probe/long > probe_long.log 2>&1 &
$P train.py --seed 95 --steps 60000 --curriculum 0.5 --batch 256 --eval-every 2000 --out probe/cur_b256 > probe_cur_b256.log 2>&1 &
$P train.py --seed 94 --steps 60000 --curriculum 0.5 --lr 3e-4 --eval-every 2000 --out probe/cur_lr3e4 > probe_cur_lr3e4.log 2>&1 &
wait
