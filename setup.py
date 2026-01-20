from setuptools import find_packages, setup

setup(
    name="omni_drones",
    version="0.2.1",
    author="btx0424@SUSTech",
    keywords=["robotics", "rl"],
    packages=find_packages("."),
    install_requires=[
        "hydra-core>=1.3.2",
        "omegaconf>=2.3.0",
        "wandb>=0.22.0",
        "PyYAML>=6.0.2",
        "numpy>=1.26.0",
        "scipy>=1.15.3",
        "tqdm>=4.67.1",
        "einops>=0.8.1",
        "pandas>=2.3.2",
        "imageio>=2.37.0",
        "moviepy>=2.2.1",
        "av>=15.1.0",
        "plotly>=5.3.1",
        "tensordict>=0.10.0",
        "torchrl>=0.10.0",
    ],
)
