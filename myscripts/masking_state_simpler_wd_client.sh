
export CUDA_VISIBLE_DEVICES=0

TASKS=(
  simpler_env_widowx/widowx_carrot_on_plate
  simpler_env_widowx/widowx_put_eggplant_in_basket
  simpler_env_widowx/widowx_spoon_on_towel
  simpler_env_widowx/widowx_stack_cube
)

action_horizon=10
EPISODES=50
N_envs=1

knn=3
n_candidates=24
search_opts="by grounded_sam_tracking alpha 0.2 num_repeats 24 n_candidates $n_candidates knn_k $knn"



for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/simpler_env/masking_state_knn_${knn}_ah_${action_horizon}_candidates_${n_candidates}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --algo "masking_state" \
        --search_opts $search_opts \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port 5556 \
        --max_episode_steps=300 \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR"  \
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