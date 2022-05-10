# Single Machine

## Horovod

### Original

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/byteps/examples/
horovodrun -np 8 python3.8 benchmark_horovod.py
```

### BytePS

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/byteps/examples/
bash ./run_bps.sh
```

### AIACC

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m registry.cn-beijing.aliyuncs.com/cto_office/perseus-training:centos7-cu11.0-pt1.7.1-py38-latest bash
cd /root/byteps/examples/
mpirun -np 8 -H aiacc:8 python3 ./benchmark_aiacc.py --model resnet50 --num-iters 100
```

## DDP

### Original

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/byteps/examples/
python3.8 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0  --use_env benchmark_ddp.py
```

### AIACC

Same as above.

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m registry.cn-beijing.aliyuncs.com/cto_office/perseus-training:centos7-cu11.0-pt1.7.1-py38-latest bash
cd /root/byteps/examples/
python3.8 -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0  --use_env benchmark_ddp_aiacc.py
```

### BytePS

```bash
nvidia-docker run -it -v /root/test/byteps:/root/byteps --rm --net=host --shm-size=32768m sineliu/byteps bash
cd /root/byteps/examples/
bash ./run_bps.sh
```
