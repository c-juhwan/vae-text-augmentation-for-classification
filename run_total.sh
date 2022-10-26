TASK=augmentation
DATASET=SST2
SEQ_LEN=50
GPU_NO=3
BS=64
LR=3e-4
EP=300
OPT_OBJ=accuracy
MODEL_TYPE=transformer
VARIATIONAL_TYPE=VAE
DENOSING_RATE=0.1
MODEL_NAME=${VARIATIONAL_TYPE}_D${DENOSING_RATE}_TextAug_${MODEL_TYPE}

clear
python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET --max_seq_len=$SEQ_LEN
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --variational_type=${VARIATIONAL_TYPE} --denosing_rate=${DENOSING_RATE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
python main.py --task=$TASK --job=testing --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --variational_type=${VARIATIONAL_TYPE} --denosing_rate=${DENOSING_RATE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
python main.py --task=$TASK --job=inference --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --variational_type=${VARIATIONAL_TYPE} --denosing_rate=${DENOSING_RATE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
######################################################################################
TASK=classification
BS=32
LR=5e-4
EP=300
MODEL_TYPE=transformer
MODEL_NAME=TextCls_${MODEL_TYPE}

clear
python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET --max_seq_len=$SEQ_LEN
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --training_dataset_aug=none \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
python main.py --task=$TASK --job=testing --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}

#######################
MODEL_NAME=TextCls_${MODEL_TYPE}+Aug ## 1% 향상

clear
#python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET --max_seq_len=$SEQ_LEN
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --training_dataset_aug=model \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
python main.py --task=$TASK --job=testing --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}

#######################
MODEL_NAME=TextCls_${MODEL_TYPE}+AugOnly

clear
#python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET --max_seq_len=$SEQ_LEN
python main.py --task=$TASK --job=training --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} --training_dataset_aug=model_only \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
python main.py --task=$TASK --job=testing --task_dataset=$DATASET --device=cuda:${GPU_NO} \
               --model_name=${MODEL_NAME} --max_seq_len=$SEQ_LEN --optimize_objective=${OPT_OBJ} \
               --model_type=${MODEL_TYPE} \
               --learning_rate=${LR} --batch_size=${BS} --num_epochs=${EP}
