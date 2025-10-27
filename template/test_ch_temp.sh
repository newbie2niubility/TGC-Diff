# export NCCL_P2P_DISABLE=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=4
export TORCH_HOME=/mnt/wanghonglie/.cache
CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node=1 --master_port=11111 test_temp.py \
    --log_name Chinese \
    --pretrain your_model_folder