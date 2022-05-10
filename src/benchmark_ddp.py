from __future__ import print_function

import argparse
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import models
import timeit
import numpy as np
import os, sys
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Benchmark settings
parser = argparse.ArgumentParser(description='PyTorch Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-pushpull', action='store_true', default=False,
                    help='use fp16 compression during byteps pushpull')

parser.add_argument('--model', type=str, default='resnet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=30,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=100,
                    help='number of benchmark iterations')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--profiler', action='store_true', default=False,
                    help='disables profiler')
parser.add_argument('--partition', type=int, default=None,
                    help='partition size')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--local_world_size', type=int, default=1)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True

def demo_basic():
    rank = dist.get_rank()
    print(f"Running basic DDP example on rank {rank}.")

    print(f'[{rank}] setting up fake data')
    datasets = []
    labels = []
    for _ in range(100):
        data = torch.rand(args.batch_size, 3, 224, 224)
        target = torch.LongTensor(args.batch_size).random_() % 1000
        if args.cuda:
            data, target = data.to(rank), target.to(rank)
        datasets.append(data)
        labels.append(target)
    idx = 0
    print(f'[{rank}] data prepared')

    # create model and move it to GPU with id rank
    model = models.resnet50(num_classes=args.num_classes).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    def step():
        nonlocal idx
        optimizer.zero_grad()
        outputs = ddp_model(datasets[idx%len(datasets)])
        F.cross_entropy(outputs, labels[idx%len(datasets)]).backward()
        optimizer.step()
        idx += 1


    print(f'[{rank}] warm up')
    time = timeit.timeit(step, number=args.num_warmup_batches)

    print(f'[{rank}] benchmark')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        img_secs.append(img_sec)
        if rank == 0:
            print('Iter #%d: %.1f img/sec' % (x, img_sec))
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    if rank == 0:
        print('Img/sec: %.1f +-%.1f' % (img_sec_mean, img_sec_conf))


def spmd_main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    }
    print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  
    dist.init_process_group(backend="nccl")
    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    demo_basic()

    # Tear down the process group
    dist.destroy_process_group()


if __name__ == '__main__':
    spmd_main()
