TASK=augmentation
DATASET=IMDB

#python main.py --task=$TASK --job=preprocessing --task_dataset=$DATASET
python main.py --task=$TASK --job=training --task_dataset=$DATASET