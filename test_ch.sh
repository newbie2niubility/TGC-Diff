export OMP_NUM_THREADS=4
export TORCH_HOME=./.cache

CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node=1 --master_port=11121 test_ch.py \
    --log_name Chinese \
    --cfg ./configs/test.yaml \
    --pre_train ./weight/weight.pt \

    