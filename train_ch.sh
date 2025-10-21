export OMP_NUM_THREADS=4
export TORCH_HOME=./.cache

CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node=1 --master_port=14231 train_ch.py \
    --feat_model ./.cache/RN18_class_10400.pth \
    --log_name Chinese \
    --cfg ./configs/Chinese_textline_vae16.yaml \

    
