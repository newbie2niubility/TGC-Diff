export OMP_NUM_THREADS=4
export TORCH_HOME=./.cache

CUDA_VISIBLE_DEVICES=0, torchrun --nproc_per_node=1 --master_port=13111 finetune_ch.py \
    --log_name Chinese \
    --cfg ./configs/Chinese_textline_vae16_finetune.yaml \
    --pre_train ./output/Chinese-20251014_013327/model/600-ckpt.pt \
    --fine_tune

    