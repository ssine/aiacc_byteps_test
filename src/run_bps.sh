# now you are in docker environment
export NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # gpus list
export DMLC_WORKER_ID=0 # your worker id
export DMLC_NUM_WORKER=1 # one worker
export DMLC_ROLE=worker

# the following value does not matter for non-distributed jobs
export DMLC_NUM_SERVER=1
export DMLC_PS_ROOT_URI=10.0.0.1
export DMLC_PS_ROOT_PORT=1234

bpslaunch python3.8 ./benchmark_byteps.py --model resnet50 --num-iters 100
# bpslaunch python3.8 ./benchmark_byteps_ddp.py --model resnet50 --num-iters 100