from setuptools import find_packages, setup

setup(
    name="omni_drones",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core",
        "omegaconf",
        "wandb",
        "moviepy",
        "imageio",
        "plotly",
        "einops"
        "av", # for moviepy
        "pandas",
        # install by cloning from github
        # "tensordict" 
        # "torchrl",
    ],
)
