# Demo Task with Trained Model

## How to run demo task with trained model

Open [demo_task.yaml](https://github.com/btx0424/OmniDrones/blob/main/examples/demo_task.yaml) and change the parameters in wandb field.

```bash
...
wandb:
  entity: finani
  project: omnidrones-public
  artifact_name: Hover-ppo
  artifact_version: rtx4090-20m-n128
...
```

## Single-Agent Tasks

| OmniDrones</br>Version | Task                  | Algorithm | GPU      | Total</br>Frames | # Envs | Eval</br>Interval | Save</br>Interval | Runtime     | -   | Entity | Project           | Artifact</br>Name         | Artifact</br>Version | Wandb</br>Link                                                               |
| ---------------------- | --------------------- | --------- | -------- | ---------------- | ------ | ----------------- | ----------------- | ----------- | --- | ------ | ----------------- | ------------------------- | -------------------- | ---------------------------------------------------------------------------- |
| 0.1.0                  | Hover                 | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 55m 46s | -   | finani | omnidrones-public | Hover-ppo                 | rtx4090-20m-n128     | [gptmuz5h](https://wandb.ai/finani/omnidrones-public/runs/gptmuz5h/overview) |
| 0.1.0                  | Track                 | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 51m 37s | -   | finani | omnidrones-public | Track-ppo                 | rtx4090-20m-n128     | [5ma07rww](https://wandb.ai/finani/omnidrones-public/runs/5ma07rww/overview) |
| 0.1.0                  | FlyThrough            | PPO       | RTX 4090 | 5M               | 32     | 1000              | 2000              | 00h 58m 02s | -   | finani | omnidrones-public | FlyThrough-ppo            | rtx4090-5m-n32       | [zksd1tis](https://wandb.ai/finani/omnidrones-public/runs/zksd1tis/overview) |
| 0.1.0                  | PayloadHover          | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 55m 05s | -   | finani | omnidrones-public | PayloadHover-ppo          | rtx4090-20m-n128     | [some26mz](https://wandb.ai/finani/omnidrones-public/runs/some26mz/overview) |
| 0.1.0                  | PayloadTrack          | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 54m 39s | -   | finani | omnidrones-public | PayloadTrack-ppo          | rtx4090-20m-n128     | [p3wxnx1e](https://wandb.ai/finani/omnidrones-public/runs/p3wxnx1e/overview) |
| 0.1.0                  | PayloadFlyThrough     | PPO       | RTX 4090 | 5M               | 32     | 1000              | 2000              | 00h 55m 24s | -   | finani | omnidrones-public | PayloadFlyThrough-ppo     | rtx4090-5m-n32       | [amd3nx6p](https://wandb.ai/finani/omnidrones-public/runs/amd3nx6p/overview) |
| 0.1.0                  | InvPendulumHover      | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 57m 01s | -   | finani | omnidrones-public | InvPendulumHover-ppo      | rtx4090-20m-n128     | [rpctr6t1](https://wandb.ai/finani/omnidrones-public/runs/rpctr6t1/overview) |
| 0.1.0                  | InvPendulumTrack      | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 00h 59m 06s | -   | finani | omnidrones-public | InvPendulumTrack-ppo      | rtx4090-20m-n128     | [xy17mm2i](https://wandb.ai/finani/omnidrones-public/runs/xy17mm2i/overview) |
| 0.1.0                  | InvPendulumFlyThrough | PPO       | RTX 4090 | 5M               | 32     | 1000              | 2000              | 00h 58m 17s | -   | finani | omnidrones-public | InvPendulumFlyThrough-ppo | rtx4090-5m-n32       | [i193m1z3](https://wandb.ai/finani/omnidrones-public/runs/i193m1z3/overview) |
| 0.1.0                  | Forest                | PPO       | RTX 4090 | 20M              | 128    | 1000              | 2000              | 01h 01m 10s | -   | finani | omnidrones-public | Forest-ppo                | rtx4090-20m-n128     | [pq30coja](https://wandb.ai/finani/omnidrones-public/runs/pq30coja/overview) |
| 0.1.0                  | Pinball               | PPO       | RTX 4090 | 5M               | 32     | 1000              | 2000              | 00h 54m 44s | -   | finani | omnidrones-public | Pinball-ppo               | rtx4090-5m-n32       | [xdjbajka](https://wandb.ai/finani/omnidrones-public/runs/xdjbajka/overview) |

## Multi-Agent Tasks

| OmniDrones</br>Version | Task                | Algorithm | GPU      | Total</br>Frames | # Envs | Eval</br>Interval | Save</br>Interval | Runtime     | -   | Entity | Project           | Artifact</br>Name       | Artifact</br>Version | Wandb</br>Link                                                               |
| ---------------------- | ------------------- | --------- | -------- | ---------------- | ------ | ----------------- | ----------------- | ----------- | --- | ------ | ----------------- | ----------------------- | -------------------- | ---------------------------------------------------------------------------- |
| 0.1.0                  | PlatformHover       | PPO       | RTX 4090 | 300K             | 2      | 1000              | 2000              | 00h 54m 27s | -   | finani | omnidrones-public | PlatformHover-ppo       | rtx4090-300k-n2      | [xvp9jgzn](https://wandb.ai/finani/omnidrones-public/runs/xvp9jgzn/overview) |
| 0.1.0                  | PlatformTrack       | PPO       | RTX 4090 | 300K             | 2      | 1000              | 2000              | 00h 55m 29s | -   | finani | omnidrones-public | PlatformTrack-ppo       | rtx4090-300k-n2      | [sdcl0vhm](https://wandb.ai/finani/omnidrones-public/runs/sdcl0vhm/overview) |
| 0.1.0                  | PlatformFlyThrough  | PPO       | RTX 4090 | 300K             | 2      | 1000              | 2000              | 00h 54m 34s | -   | finani | omnidrones-public | PlatformFlyThrough-ppo  | rtx4090-300k-n2      | [7w2ovagr](https://wandb.ai/finani/omnidrones-public/runs/7w2ovagr/overview) |
| 0.1.0                  | TransportHover      | PPO       | RTX 4090 | 150K             | 1      | 1000              | 2000              | 00h 59m 23s | -   | finani | omnidrones-public | TransportHover-ppo      | rtx4090-150k-n1      | [6uv6ao6s](https://wandb.ai/finani/omnidrones-public/runs/6uv6ao6s/overview) |
| 0.1.0                  | TransportTrack      | PPO       | RTX 4090 | 150K             | 1      | 1000              | 2000              | 00h 55m 43s | -   | finani | omnidrones-public | TransportTrack-ppo      | rtx4090-150k-n1      | [7keik7j3](https://wandb.ai/finani/omnidrones-public/runs/7keik7j3/overview) |
| 0.1.0                  | TransportFlyThrough | PPO       | RTX 4090 | 150K             | 1      | 1000              | 2000              | 00h 55m 08s | -   | finani | omnidrones-public | TransportFlyThrough-ppo | rtx4090-150k-n1      | [7hs3ujd7](https://wandb.ai/finani/omnidrones-public/runs/7hs3ujd7/overview) |

## How to add trained model (for contributors)

- Trained model is required
  - to be uploaded to wandb ***public project*** because the model can be downloaded.
  - to be including ***videos*** (eval_interval != -1)
