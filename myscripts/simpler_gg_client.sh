
export CUDA_VISIBLE_DEVICES=3

TASKS=(
  simpler_env_google/google_robot_close_drawer
  simpler_env_google/google_robot_move_near
  simpler_env_google/google_robot_open_drawer
  simpler_env_google/google_robot_pick_coke_can
  simpler_env_google/google_robot_place_in_closed_drawer
)



LOG_DIR="eval_logs/simpler_env/baseline_nas4"
mkdir -p "$LOG_DIR"

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    echo "Running task: $TASK"

    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --n_episodes 100 \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5555 \
        --max_episode_steps=300 \
        --env_name "$TASK" \
        --n_action_steps 4 \
        --n_envs 5 \
        > "$LOG_DIR/${NAME}.txt" 2>&1

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