#!/bin/bash
cd /root/pilot
P=/root/venv/bin/python
for s in 0 1 2; do
  $P pilot3.py a --seed $s --runs runs_rope > pilot3_a_seed$s.log 2>&1 &
  $P pilot3.py b --seed $s --runs runs > pilot3_b_nope_seed$s.log 2>&1 &
  $P pilot3.py b --seed $s --runs runs_rope > pilot3_b_rope_seed$s.log 2>&1 &
done
wait
echo PILOT3_DONE
