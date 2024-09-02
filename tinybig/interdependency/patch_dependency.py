# Copyright (c) 2024-Present
# Author: Jiawei Zhang <jiawei@ifmlab.org>
# Affiliation: IFM Lab, UC Davis


#########################
# Patch Interdependency #
#########################

import torch
import torch.nn.functional as F

from tinybig.interdependency import interdependency

import numpy as np


# def create_dependency_matrix(h, w, d, neighborhood_size=1):
#     n = h * w * d
#     D = np.zeros((n, n), dtype=bool)
#
#     for i in range(h):
#         for j in range(w):
#             for k in range(d):
#                 center_idx = d * (h * i + j) + k
#
#                 for di in range(-neighborhood_size, neighborhood_size + 1):
#                     for dj in range(-neighborhood_size, neighborhood_size + 1):
#                         for dk in range(-neighborhood_size, neighborhood_size + 1):
#                             if di == 0 and dj == 0 and dk == 0:
#                                 continue  # Skip the center pixel itself
#
#                             ni, nj, nk = i + di, j + dj, k + dk
#
#                             if 0 <= ni < h and 0 <= nj < w and 0 <= nk < d:
#                                 neighbor_idx = d * (h * ni + nj) + nk
#                                 D[center_idx, neighbor_idx] = True
#
#     return D


# def create_dependency_matrix_fast(h, w, d, neighborhood_size=1):
#     n = h * w * d
#
#     # Create coordinate matrices
#     y, x, z = np.meshgrid(np.arange(h), np.arange(w), np.arange(d), indexing='ij')
#     coords = np.stack([y, x, z], axis=-1).reshape(-1, 3)
#
#     # Create neighborhood offsets
#     offsets = np.array([(dy, dx, dz)
#                         for dy in range(-neighborhood_size, neighborhood_size + 1)
#                         for dx in range(-neighborhood_size, neighborhood_size + 1)
#                         for dz in range(-neighborhood_size, neighborhood_size + 1)
#                         if (dy, dx, dz) != (0, 0, 0)])
#
#     # Add offsets to coordinates
#     neighbors = coords[:, np.newaxis, :] + offsets[np.newaxis, :, :]
#
#     # Check which neighbors are valid
#     valid = np.all((neighbors >= 0) & (neighbors < [h, w, d]), axis=2)
#
#     # Convert coordinates to flattened indices
#     center_indices = np.arange(n)
#     neighbor_indices = neighbors[:, :, 0] * (w * d) + neighbors[:, :, 1] * d + neighbors[:, :, 2]
#
#     # Create sparse matrix
#     rows = np.repeat(center_indices, valid.sum(axis=1))
#     cols = neighbor_indices[valid]
#     data = np.ones_like(rows, dtype=bool)
#
#     D = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
#
#     return D

class cuboid_interdependency(interdependency):

    def forward(self, *args, **kwargs):
        pass


# import numpy as np
#
#
# def get_circular_region(m, n, center_i, center_j, radius):
#     # Create a grid of coordinates
#     y, x = np.ogrid[:m, :n]
# 
#     # Calculate distances from the center
#     distances = np.sqrt((x - center_j) ** 2 + (y - center_i) ** 2)
#
#     # Create a mask for points within the circle
#     mask = distances <= radius
#
#     # Get the indices where mask is True
#     indices = np.where(mask)
#
#     # Combine the indices into pairs
#     return list(zip(indices[0], indices[1]))
#
#
# # Example usage
# m, n = 10, 10  # matrix dimensions
# center_i, center_j = 5, 5  # center of the circle
# radius = 3  # radius of the circle
#
# result = get_circular_region(m, n, center_i, center_j, radius)
# print(result)

class cylinder_interdependency(interdependency):
