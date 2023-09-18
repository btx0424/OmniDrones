task_name=$1
algo_name=$2

python train.py headless=True task=$task_name algo=$algo_name  \
    eval_interval=80 task.visual_obs=True \
    wandb.group=Visual${task_name} wandb.run_name=Visual${task_name}-${algo_name} \
    wandb.entity=fengg \
    algo.critic_input=state \
    task.env.num_envs=15 