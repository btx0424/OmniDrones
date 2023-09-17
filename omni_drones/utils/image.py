# MIT License
# 
# Copyright (c) 2023 Botian Xu, Tsinghua University
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


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