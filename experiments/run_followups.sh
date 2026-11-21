#!/bin/sh
# Follow-up runs (controls) on Qwen2.5-3B. Each writes its own log under results/.
export HF_HUB_DISABLE_XET=1
M=Qwen/Qwen2.5-3B-Instruct
C="-0.5 -0.25 0.0 0.25 0.5"
python3 experiments/exp3_steering.py --model $M --emotion sadness --no-random --coefs $C > results/exp3_sadness.log 2>&1
python3 experiments/exp3_steering.py --model $M --emotion anger   --no-random --coefs $C > results/exp3_anger.log 2>&1
python3 experiments/exp3_steering.py --model $M --emotion fear --layer 18 --no-random --coefs $C > results/exp3_fear_L18.log 2>&1
python3 experiments/exp1_behavioral.py --model $M --worlds standard --modes none irrelevant --out results/exp1_length_control > results/exp1_length_control.log 2>&1
python3 experiments/exp1_behavioral.py --model $M --worlds punish_only --modes none factual pain --out results/exp1_punish_only > results/exp1_punish_only.log 2>&1
echo ALL_DONE > results/followups.done
