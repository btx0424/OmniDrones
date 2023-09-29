task_name=$1
algo_name=$2

python train.py headless=True task=$task_name algo=$algo_name  \
    eval_interval=80 \
    wandb.group=${task_name} wandb.run_name=${task_name}-${algo_name} \
    wandb.entity=fengg \
    task.env.num_envs=4096