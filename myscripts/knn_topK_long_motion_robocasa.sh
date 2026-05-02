
export CUDA_VISIBLE_DEVICES=1
module load gcc/13.2.0
module load ffmpeg/7.0.2

TASKS=(
  robocasa_panda_omron/CoffeeSetupMug_PandaOmron_Env
)

action_horizon=8
EPISODES=50
N_envs=1
PORT=$1

knn=6
n_candidates=12
top_k=5
long_ah=16
search_opts="by grounded_sam_tracking alpha 0.2 num_repeats 24 n_candidates $n_candidates knn_k $knn top_k $top_k long_ah $long_ah"



for TASK in "${TASKS[@]}"; do
    NAME=$(basename "$TASK")

    LOG_DIR="eval_logs/gg_robot/knn_${knn}_topK_${top_k}_long_ah_${long_ah}_motion_ah_${action_horizon}_candidates_${n_candidates}_nenvs_${N_envs}/$NAME"
    VIDEO_DIR="$LOG_DIR/videos"
    mkdir -p "$LOG_DIR"
    mkdir -p "$VIDEO_DIR"

    echo "Running task: $TASK"

    gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
        --algo "knn_topK_long_motion" \
        --search_opts $search_opts \
        --n_episodes $EPISODES \
        --policy_client_host 127.0.0.1 \
        --policy_client_port $PORT \
        --max_episode_steps=300 \
        --env_name "$TASK" \
        --n_action_steps $action_horizon \
        --n_envs $N_envs \
        --video_dir "$VIDEO_DIR" \
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