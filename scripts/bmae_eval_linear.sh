export CUDA_VISIBLE_DEVICES=4,5,6,7 
NPROC_PER_NODE=4                   

NNODES=1                           
NODE_RANK=0                        
MASTER_ADDR=localhost              
MASTER_PORT=29513                  

DATA_PATH="./data/cifar10_data"    
NB_CLASSES=10                     
MODEL_NAME="vit_base_patch"     
INPUT_SIZE=32                     


PRETRAIN_STAGES=3
PRETRAIN_DIR="logs/boost_mae_train_stages_${PRETRAIN_STAGES}"
PRETRAIN_CHKPT="${PRETRAIN_DIR}/final_boostmae_model_stages_${PRETRAIN_STAGES}.pth"
echo "Using pre-trained checkpoint: ${PRETRAIN_CHKPT}"
if [ ! -f "${PRETRAIN_CHKPT}" ]; then
    echo "Error: Pre-trained checkpoint not found at ${PRETRAIN_CHKPT}"
    exit 1
fi


EPOCHS=100                         
BATCH_SIZE=512                    
ACCUM_ITER=1                       
BASE_LR=1e-3                       
WEIGHT_DECAY=0.0                   
WARMUP_EPOCHS=5                    

EXP_NAME="linprobe_stage${PRETRAIN_STAGES}"
OUTPUT_DIR="logs/${EXP_NAME}"
LOG_DIR="${OUTPUT_DIR}/tensorboard"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${LOG_DIR}

echo "Starting CIFAR-10 Linear Probing..."
torchrun --nproc_per_node=${NPROC_PER_NODE} --nnodes=${NNODES} --node_rank=${NODE_RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
main_linprobe.py \
    --model ${MODEL_NAME} \
    --finetune ${PRETRAIN_CHKPT} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM_ITER} \
    --blr ${BASE_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --min_lr 1e-6 \
    --nb_classes ${NB_CLASSES} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${LOG_DIR} \
    --seed 42 \
    --dist_url 'env://' \
    --num_workers 10 \
    --dist_eval \
    --input_size ${INPUT_SIZE} \

echo "Linear probing script finished."