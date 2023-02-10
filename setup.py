from setuptools import setup, find_packages

setup(
    name="isaac_drones",
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
        # "torchinfo",
        # "torchopt"
    ]
)