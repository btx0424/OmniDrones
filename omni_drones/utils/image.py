import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from typing import List, Union


def save_depth(imgs: Union[torch.Tensor, np.ndarray, List[np.ndarray]], save_path: str = './'):
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    elif isinstance(imgs, torch.Tensor):
        imgs = [imgs.detach().cpu().numpy()]

    for i in range(len(imgs)):
        assert imgs[i].ndim == 3 and imgs[i].shape[0] == 1
        depth_map_actual = imgs[i].clip(0, 100)

        fig, ax = plt.subplots()
        im = ax.imshow(depth_map_actual[0], cmap='jet')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Depth (m)', rotation=-90, va="bottom")

        plt.savefig(f"{save_path}/depth_map_{i}.png", dpi=300, bbox_inches='tight')

def save_image(imgs: Union[torch.Tensor, np.ndarray, List[np.ndarray]], save_path: str = './'):
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]
    elif isinstance(imgs, torch.Tensor):
        imgs = [imgs.detach().cpu().numpy()]

    for i in range(len(imgs)):
        assert imgs[i].ndim == 3 and imgs[i].shape[0] == 3

        image_pil = Image.fromarray(imgs[i].transpose(1, 2, 0))
        image_pil.save(f"{save_path}/image_{i}.png")