
export CUDA_VISIBLE_DEVICES=3

TASKS=(
#   simpler_env_widowx/widowx_carrot_on_plate
  simpler_env_widowx/widowx_put_eggplant_in_basket
  simpler_env_widowx/widowx_spoon_on_towel
  simpler_env_widowx/widowx_stack_cube
)

knn_k=5


LOG_DIR="eval_logs/debug_knn_${knn_k}_testcode"
mkdir -p "$LOG_DIR"

for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    echo "Running task: $TASK"

    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --knn_k $knn_k \
        --n_episodes 50 \
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