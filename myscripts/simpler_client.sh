gr00t/eval/sim/SimplerEnv/simpler_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
    --n_episodes 10 \
    --policy_client_host 127.0.0.1 \
    --policy_client_port 5555 \
    --max_episode_steps=300 \
    --env_name simpler_env_google/google_robot_pick_coke_can \
    --n_action_steps 1 \
    --n_envs 5