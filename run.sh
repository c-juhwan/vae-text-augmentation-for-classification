TASK=augmentation
DATASET=SST2
GPU_NO=2
BS=32
LR=3e-3
MODEL_NAME=VAE_TextAug

clear
#python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} \
               --learning_rate=${LR} --batch_size=${BS}
