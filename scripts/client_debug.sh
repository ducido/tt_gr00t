source .venv/bin/activate
which python
TASKS=(
  robocasa_panda_omron/OpenDrawer_PandaOmron_Env
  # robocasa_panda_omron/CloseSingleDoor_PandaOmron_Env
  # robocasa_panda_omron/CoffeePressButton_PandaOmron_Env
  # robocasa_panda_omron/CloseDoubleDoor_PandaOmron_Env
  # robocasa_panda_omron/CloseDrawer_PandaOmron_Env
  # robocasa_panda_omron/TurnOnMicrowave_PandaOmron_Env
  # robocasa_panda_omron/TurnOffMicrowave_PandaOmron_Env
)

LOG_DIR="eval_logs/debug"
mkdir -p "$LOG_DIR"

for TASK in "${TASKS[@]}"; do
  NAME=$(basename "$TASK")

  echo "Running task: $TASK"

  gr00t/eval/sim/robocasa/robocasa_uv/.venv/bin/python gr00t/eval/rollout_policy.py \
      --n_episodes 10 \
      --policy_client_host 127.0.0.1 \
      --policy_client_port 5555 \
      --max_episode_steps=720 \
      --env_name "$TASK" \
      --n_action_steps 8 \
      --n_envs 5 \
      --sigma 1.0 # > "$LOG_DIR/${NAME}.txt" 2>&1

  echo "Finished task: $TASK"
  echo ""
done