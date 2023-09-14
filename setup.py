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
        "multielo @ git+https://github.com/djcunningham0/multielo.git@v0.4.0"
        # "torchinfo",
        # "torchopt"
    ],
)
