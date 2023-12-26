import os
from omni_drones import init_simulation_app

if __name__ == "__main__":
    init_simulation_app({"headless": True})

    from omni_drones.envs import IsaacEnv
    print(IsaacEnv.REGISTRY)

    single_tasks = [
        "Hover", "Track", "Forest", "Pinball",
        "InvPendulumHover", "InvPendulumTrack", "InvPendulumFlyThrough",
        "PayloadTrack", "PayloadFlyThrough"
    ]
    multi_tasks = [
        # "Formation",
        "PlatformHover",
        "PlatformTrack",
        "TransportHover",
        "TransportTrack",
    ]

    for task in single_tasks:
        with open(f"source/tasks/single/{task}.md", "w") as f:
            lines = [
                line.strip() + "\n"
                for line in IsaacEnv.REGISTRY[task].__doc__.splitlines(True)
            ]
            f.write(f"{task}\n" + "=" * len(task) + "\n\n")
            f.writelines(lines)
    
    for task in multi_tasks:
        with open(f"source/tasks/multi/{task}.md", "w") as f:
            lines = [
                line.strip() + "\n"
                for line in IsaacEnv.REGISTRY[task].__doc__.splitlines(True)
            ]
            f.write(f"{task}\n" + "=" * len(task) + "\n\n")
            f.writelines(lines)