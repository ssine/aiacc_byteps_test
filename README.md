# Performance Testing on AIACC & BytePS

Nvidia Docker installation on baremetal machine:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
systemctl enable docker
systemctl start docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo
dnf clean expire-cache --refresh
dnf install -y nvidia-docker2
systemctl restart docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

## Horovod

### Original

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/src
horovodrun -np 8 python3.8 benchmark_horovod.py
```

### BytePS

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/src
bash ./run_bps.sh
```

### AIACC

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m registry.cn-beijing.aliyuncs.com/cto_office/perseus-training:centos7-cu11.0-pt1.7.1-py38-latest bash
cd /root/src
mpirun -np 8 -H aiacc:8 python3 ./benchmark_aiacc.py --model resnet50 --num-iters 100
```

## DDP

### Original

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/src
python3.8 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0  --use_env benchmark_ddp.py
```

### AIACC

Same as above.

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m registry.cn-beijing.aliyuncs.com/cto_office/perseus-training:centos7-cu11.0-pt1.7.1-py38-latest bash
cd /root/src
python3.8 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0  --use_env benchmark_ddp_aiacc.py
```

### BytePS

```bash
nvidia-docker run -it -v /root/test/src:/root/src --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/src
bash ./run_bps.sh
```
