
export CUDA_VISIBLE_DEVICES=1

TASKS=(
#   simpler_env_google/google_robot_close_drawer
  simpler_env_google/google_robot_move_near
#   simpler_env_google/google_robot_open_drawer
#   simpler_env_google/google_robot_pick_coke_can
#   simpler_env_google/google_robot_place_in_closed_drawer
)



action_horizon=1
EPISODES=5
N_envs=1
PORT=$1

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/google_simpler_env/baseline_nenvs${N_envs}_eps${EPISODES}_ah${action_horizon}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy_store-action.py \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=300 \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR" # \
        # > "$LOG_DIR/${NAME}.txt" 2>&1

    echo "Finished task: $TASK"
    echo ""
done
