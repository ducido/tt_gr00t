
source .venv/bin/activate

module load gcc/13.2.0
module load cuda/12.6.2

python gr00t/eval/run_gr00t_server.py \
    --model-path CP/GR00T-N1.6-3B \
    --embodiment-tag ROBOCASA_PANDA_OMRON \
    --use-sim-policy-wrapper 