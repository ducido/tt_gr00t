
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0


# export PYTHONPATH=/pfss/mlde/workspaces/mlde_wsp_MGPATH/VLA/duc/tt_gr00t:$PYTHONPATH

module load gcc/13.2.0
module load cuda/12.6.2

.venv/bin/python gr00t/eval/run_gr00t_server.py \
    --model-path CP/GR00T-N1.6-fractal \
    --embodiment-tag OXE_GOOGLE \
    --use-sim-policy-wrapper \
    --port 5510