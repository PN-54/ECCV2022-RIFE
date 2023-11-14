import torch
import torch.nn as nn
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}


# def warp(tenInput, tenFlow):
#     k = (str(tenFlow.device), str(tenFlow.size()))
#     if k not in backwarp_tenGrid:
#         tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
#             1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
#         tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
#             1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
#         # tenHorizontal = torch.linspace(-1.0, 1.0, shape_3, device=device).view(
#         #     1, 1, 1, shape_3).expand(shape_0, -1, shape_2, -1)
#         # tenVertical = torch.linspace(-1.0, 1.0, shape_2, device=device).view(
#         #     1, 1, shape_2, 1).expand(shape_0, -1, -1, shape_3)

#         backwarp_tenGrid[k] = torch.cat(
#             [tenHorizontal, tenVertical], 1).to(device)

#     # tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((shape_3 - 1.0) / 2.0),
#     #                      tenFlow[:, 1:2, :, :] / ((shape_2 - 1.0) / 2.0)], 1)
#     tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0),
#                          tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0)], 1)

#     g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
#     return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)

def warp(tenInput, tenFlow, tenShape):
    k = (str(tenFlow.device), str(tenFlow.size()))
    if k not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenShape[3], device=device).view(
            1, 1, 1, tenShape[3]).expand(tenShape[0], -1, tenShape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenShape[2], device=device).view(
            1, 1, tenShape[2], 1).expand(tenShape[0], -1, -1, tenShape[3])

        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenShape[3] - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((tenShape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
