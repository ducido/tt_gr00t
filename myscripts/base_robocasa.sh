
export CUDA_VISIBLE_DEVICES=0

TASKS=(
  robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env
)

action_horizon=8
EPISODES=30
N_envs=1
PORT=$1

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/robocasa/baseline_nenvs${N_envs}_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=600 \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR" # \
        #> "$LOG_DIR/${NAME}.txt" 2>&1

    echo "Finished task: $TASK"
    echo ""
done


# gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
#     --n_episodes 10 \
#     --policy_client_host 127.0.0.1 \
#     --policy_client_port 5555 \
#     --max_episode_steps=300 \
#     --env_name simpler_env_google/google_robot_pick_coke_can \
#     --n_action_steps 1 \
#     --n_envs 5