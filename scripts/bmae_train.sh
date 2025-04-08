#!/bin/bash

export CUDA_VISIBLE_DEVICES=4,5,6,7 


NNODES=1               
NPROC_PER_NODE=4       
NODE_RANK=0            
MASTER_ADDR=localhost   
MASTER_PORT=29504 

TOTAL_EPOCHS=200        
BOOTSTRAP_STAGES=3   

MODEL_NAME="boost_mae_vit_base_patch" 
DATA_PATH="data/cifar10_data"    
BATCH_SIZE=512         
ACCUM_ITER=2            

BASE_LR=1.5e-4         
WEIGHT_DECAY=0.05
WARMUP_EPOCHS=10        


OUTPUT_DIR="logs/boost_mae_train_stages_${BOOTSTRAP_STAGES}"
LOG_DIR="${OUTPUT_DIR}/tensorboard"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
boost_main_pretrain.py \
    --batch_size ${BATCH_SIZE} \
    --epochs ${TOTAL_EPOCHS} \
    --accum_iter ${ACCUM_ITER} \
    --bootstrap_stages ${BOOTSTRAP_STAGES} \
    --model ${MODEL_NAME} \
    --mask_ratio 0.75 \
    --norm_pix_loss \
    --blr ${BASE_LR} --weight_decay ${WEIGHT_DECAY} \
    --min_lr 1e-6 \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --seed 42 \
    --dist_url 'env://' \
    --num_workers 10 \

echo "BoostMAE training script finished."
