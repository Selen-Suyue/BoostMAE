export CUDA_VISIBLE_DEVICES=4,5,6,7

torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --master_addr=localhost --master_port=29500 \
main_pretrain.py \
    --batch_size 512 \
    --model mae_vit_base_patch \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --epochs 200 \
    --warmup_epochs 10 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path data/cifar10_data \
    --output_dir logs/mae_train \
    --log_dir logs/mae_train/tensorboard \
    --dist_url 'env://' 