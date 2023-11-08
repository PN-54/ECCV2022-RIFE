import torch
import torch.nn as nn
import traceback

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backwarp_tenGrid = {}

shape_0 = 1
shape_1 = 2
shape_2 = None
shape_3 = None

def initWarp(shape3, shape2):
    global shape_3
    global shape_2
    shape_3 = shape3
    shape_2 = shape2


def warp(tenInput, tenFlow):
    global shape_3
    global shape_2
    global shape_1
    global shape_0
    k = (str(tenFlow.device), str(tenFlow.size()))
    # traceback.print_stack()
    print("A")
    print(tenFlow.shape[3])
    print("B")
    print(tenFlow.shape[2])
    print("C")
    print(tenFlow.shape[1])
    print("D")
    print(tenFlow.shape[0])
    if k not in backwarp_tenGrid:
        # print("TAKEN")
        # tenHorizontal = torch.linspace(-1.0, 1.0, 448, device=device).view(
        #     1, 1, 1, 448).expand(1, -1, 256, -1)
        # tenVertical = torch.linspace(-1.0, 1.0, 256, device=device).view(
        #     1, 1, 256, 1).expand(1, -1, -1, 448)
        # tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3], device=device).view(
        #     1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        # tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2], device=device).view(
        #     1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        tenHorizontal = torch.linspace(-1.0, 1.0, shape_3, device=device).view(
            1, 1, 1, shape_3).expand(shape_0, -1, shape_2, -1)
        tenVertical = torch.linspace(-1.0, 1.0, shape_2, device=device).view(
            1, 1, shape_2, 1).expand(shape_0, -1, -1, shape_3)
        backwarp_tenGrid[k] = torch.cat(
            [tenHorizontal, tenVertical], 1).to(device)

        shape_3 = int(shape_3 / 2)
        shape_2 = int(shape_2 / 2)

    tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((shape_3 - 1.0) / 2.0),
                         tenFlow[:, 1:2, :, :] / ((shape_2 - 1.0) / 2.0)], 1)
    # tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenFlow.shape[3] - 1.0) / 2.0),
    #                      tenFlow[:, 1:2, :, :] / ((tenFlow.shape[2] - 1.0) / 2.0)], 1)

    g = (backwarp_tenGrid[k] + tenFlow).permute(0, 2, 3, 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=g, mode='bilinear', padding_mode='border', align_corners=True)
