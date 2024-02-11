import os
import math
import time
import torch
import torch.distributed as dist
import numpy as np
import random
import argparse
import util
import logging

from model_micro.RIFE import Model
from dataset_zarr import *
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

device = torch.device("cuda")

log_path = '/scratch/sjsch/rife-micro-log-b64'

def get_learning_rate(step):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (args.epoch * args.step_per_epoch - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model, local_rank):
    step = 0
    nr_eval = 0
    dataset = VimeoDataset('train')
    sampler = DistributedSampler(dataset)
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, pin_memory=True, drop_last=True, sampler=sampler)
    args.step_per_epoch = train_data.__len__()
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=16, pin_memory=True, num_workers=8)
    print('training...')
    time_stamp = time.time()
    for epoch in range(args.epoch):
        sampler.set_epoch(epoch)
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            learning_rate = get_learning_rate(step) * args.world_size / 4
            pred, info = model.update(imgs, gt, learning_rate, training=True) # pass timestep if you are training RIFEm
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            if step % 200 == 1 and local_rank == 0:
                print('learning_rate', learning_rate, step)
                print('loss/l1', info['loss_l1'], step)
            if local_rank == 0:
                print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, local_rank)
        model.save_model(f'{log_path}/{epoch}.pkl', local_rank)
        dist.barrier()

def evaluate(model, val_data, nr_eval, local_rank):
    loss_l1_list = []
    psnr_list = []
    time_stamp = time.time()
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.        
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)

    eval_time_interval = time.time() - time_stamp

    if local_rank != 0:
        return
    print('psnr', np.array(psnr_list).mean(), nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=16, type=int, help='minibatch size')
    parser.add_argument('--learning_rate', default=16, type=int, help='minibatch size')
    parser.add_argument('--world_size', default=4, type=int, help='world size')
    parser.add_argument('--init_method', default='tcp://127.0.0.1:19837', type=str, help='')
    args = parser.parse_args()
    ngpus_per_node = torch.cuda.device_count()
    local_rank = int(os.environ.get("SLURM_LOCALID"))
    rank = int(os.environ.get('SLURM_NODEID'))*ngpus_per_node + local_rank
    torch.distributed.init_process_group(backend="nccl", world_size=args.world_size, init_method=args.init_method, rank=rank)
    torch.cuda.set_device(local_rank)
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    model = Model(local_rank, arbitrary=True)
    train(model, local_rank)
