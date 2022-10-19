TASK=augmentation
DATASET=SST2
SEQ_LEN=50
GPU_NO=1
BS=32
LR=3e-4
OPT_OBJ=accuracy
MODEL_TYPE=transformer
MODEL_NAME=VDAE_TextAug_${MODEL_TYPE}

clear
#python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET --max_seq_len=$SEQ_LEN
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --learning_rate=${LR} --batch_size=${BS}
