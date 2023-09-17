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


import math
import random
import matplotlib.pyplot as plt
import torch

def poisson_disk_sampling(width, height, r, k=30):
    """
    Generate a set of points using Poisson disk sampling.
    
    Parameters
    ----------
    width : float
        Width of the area to sample.
    height : float
        Height of the area to sample.
    r : float
        Minimum distance between points.
    k : int
        Number of samples to try before rejecting a point.

    """
    cell_size = r / math.sqrt(2)
    grid_width = math.ceil(width / cell_size)
    grid_height = math.ceil(height / cell_size)
    grid = [None] * (grid_width * grid_height)
    process_list = []
    def grid_coords(p):
        return math.floor(p[0] / cell_size), math.floor(p[1] / cell_size)
    def distance(p0, p1):
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        return math.sqrt(dx * dx + dy * dy)
    def fits(p):
        x, y = grid_coords(p)
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if 0 <= x + dx < grid_width and 0 <= y + dy < grid_height:
                    neighbor = grid[(x + dx) + (y + dy) * grid_width]
                    if neighbor is not None and distance(p, neighbor) < r:
                        return False
        return True
    def process(p):
        grid_x, grid_y = grid_coords(p)
        grid[grid_x + grid_y * grid_width] = p
        process_list.append(p)
    def sample(p):
        for i in range(k):
            angle = 2 * math.pi * random.random()
            distance = r * random.random() + r
            new_p = (p[0] + distance * math.cos(angle), p[1] + distance * math.sin(angle))
            if 0 <= new_p[0] < width and 0 <= new_p[1] < height and fits(new_p):
                return new_p
        return None
    p = (width * random.random(), height * random.random())
    process(p)
    while process_list:
        i = random.randint(0, len(process_list)-1)
        p = process_list[i]
        new_p = sample(p)
        if new_p is not None:
            process(new_p)
        else:
            process_list.pop(i)
    return torch.tensor([p for p in grid if p is not None])