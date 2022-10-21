# Standara Library Modules
import time
import argparse
# Custom Modules
from jobs.preprocessing.aug_preprocessing import preprocessing as aug_preprocessing
from jobs.preprocessing.cls_preprocessing import preprocessing as cls_preprocessing
from jobs.train.aug_train import training as aug_training
from jobs.train.cls_train import training as cls_training
from jobs.test.aug_test import testing as aug_testing
from jobs.test.cls_test import testing as cls_testing
from jobs.inference.aug_inference import inference as aug_inference
from jobs.inference.cls_inference import inference as cls_inference
from utils import check_path, set_random_seed

def main(args:argparse.Namespace):
    # Set random seed
    if args.seed is not None:
        set_random_seed(args.seed)

    start_time = time.time()

    # Check if the path exists
    check_path(args.data_path)
    check_path(args.preprocessed_path)
    check_path(args.model_path)
    check_path(args.checkpoint_path)
    check_path(args.result_path)
    check_path(args.tensorboard_path)

    if args.job == None:
        raise ValueError('Please specify a job to run')
    else:
        if args.task == 'augmentation':
            if args.job == 'preprocessing':
                aug_preprocessing(args)
            elif args.job == 'training':
                aug_training(args)
            elif args.job == 'testing':
                aug_testing(args)
            elif args.job == 'inference':
                aug_inference(args)
            else:
                raise ValueError(f'Job {args.job} is not implemented'.format())
        elif args.task == 'classification':
            if args.job == 'preprocessing':
                cls_preprocessing(args)
            elif args.job == 'training':
                cls_training(args)
            elif args.job == 'testing':
                cls_testing(args)
            elif args.job == 'inference':
                cls_inference(args)
            else:
                raise ValueError(f'Job {args.job} is not implemented'.format())
        else:
            raise ValueError(f'Task {args.task} is not implemented'.format())

    elapsed_time = time.time() - start_time
    print(f'Completed {args.job}; Time elapsed: {elapsed_time:.2f} seconds')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Task
    task_list = ['augmentation', 'classification']
    parser.add_argument('--task', type=str, choices=task_list, default=None,
                        help='Task to be performed')
    job_list = ['preprocessing', 'training', 'resume_training', 'testing', 'inference']
    parser.add_argument('--job', type=str, choices=job_list, default=None,
                        help='Job to be performed; Must be specified.')
    dataset_list = ['AG_News', 'IMDB', 'MR', 'ProsCons', 'SST2', 'SST5', 'SUBJ', 'TREC']
    parser.add_argument('--task_dataset', type=str, choices=dataset_list, default=None,
                        help='Dataset to be used for the task; Must be specified.')

    # Path
    parser.add_argument('--data_path', type=str, default='./dataset/',
                        help='Path to the dataset before preprocessing.')
    parser.add_argument('--preprocessed_path', type=str, default='./preprocessed/',
                        help='Path to the dataset after preprocessing.')
    parser.add_argument('--model_path', type=str, default='./result_models/',
                        help='Path to the model after training.')
    parser.add_argument('--checkpoint_path', type=str, default='./result_checkpoints/',
                        help='Path to the checkpoint during training.')
    parser.add_argument('--result_path', type=str, default='./result_results/',
                        help='Path to the result after testing.')

    # Preprocessing
    parser.add_argument('--max_seq_len', type=int, default=30,
                        help='Maximum sequence length for each task.')

    # Model - Common Arguments
    parser.add_argument('--model_name', type=str, default='VAE_TextAug',
                        help='Name of the model.')
    model_list = ['rnn', 'gru', 'lstm', 'transformer', 'cnn']
    parser.add_argument('--model_type', type=str, choices=model_list, default='transformer',
                        help='Type of the model.')
    parser.add_argument('--embed_size', type=int, default=768,
                        help='Dimension of the embedding.')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Dimension of the hidden layer.')
    parser.add_argument('--latent_size', type=int, default=32,
                        help='Dimension of the latent layer.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers for GRU; Default is 2')
    parser.add_argument('--dropout_rate', type=float, default=0.2,
                        help='Dropout Rate; Default is 0.2')
    variational_list = ['AE', 'VAE', 'CVAE']
    parser.add_argument('--variational_type', type=str, choices=variational_list, default='vae',
                        help='Whether to use variational autoencoder; Default is VAE')
    parser.add_argument('--activation_func', type=str, default='relu',
                        help='Activation function for the model.')
    parser.add_argument('--denosing_rate', type=float, default=0.1,
                        help='Denosing rate for the denosing autoencoder.')
    parser.add_argument('--kl_lambda', type=float, default=-1,
                        help='Weight for KL divergence loss; Default is 0.1; if less than 0, use KL annealing')

    # Optimizer & Scheduler
    optim_list = ['SGD', 'Adam', 'AdamW']
    scheduler_list = ['None', 'StepLR', 'LambdaLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts', 'ReduceLROnPlateau']
    parser.add_argument('--optimizer', default='Adam', choices=optim_list, type=str,
                        help="Optimizer to use; Default is Adam")
    parser.add_argument('--scheduler', default='ReduceLROnPlateau', choices=scheduler_list, type=str,
                        help="Scheduler to use; Default is LambdaLR")

    # Training - Config
    parser.add_argument('--num_epochs', default=300, type=int,
                        help='Training epochs; Default is 300')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Num CPU Workers; Default is 2')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size; Default is 16')
    parser.add_argument('--learning_rate', default=3e-4, type=float,
                        help='Maximum learning rate of warmup scheduler; Default is 3e-4')
    parser.add_argument('--weight_decay', default=5e-4, type=float,
                        help='Weight decay; Default is 5e-4; If 0, no weight decay')
    parser.add_argument('--clip_grad_norm', default=0, type=int,
                        help='Gradient clipping norm; Default is 5')
    parser.add_argument('--early_stopping_patience', default=10, type=int,
                        help='Early stopping patience; No early stopping if None; Default is 10')
    objective_list = ['loss', 'accuracy']
    parser.add_argument('--optimize_objective', default='accuracy', type=str, choices=objective_list,
                        help='Objective to optimize; Default is accuracy')

    # Testing/Inference - Config
    parser.add_argument('--test_batch_size', default=1, type=int,
                        help='Batch size for test; Default is 1')

    # Vocabulary & Label
    parser.add_argument('--vocab_size', default=-1, type=int,
                        help='Vocabulary size; To be specified for each dataset')
    parser.add_argument('--vocab_min_freq', default=3, type=int,
                        help='Minimum frequency of words in vocabulary; Default is 3')
    parser.add_argument('--pad_id', default=0, type=int,
                        help='Pad id; Default is 0')
    parser.add_argument('--unk_id', default=1, type=int,
                        help='Unknown id; Default is 1')
    parser.add_argument('--bos_id', default=2, type=int,
                        help='Beginning of sentence id; Default is 2')
    parser.add_argument('--eos_id', default=3, type=int,
                        help='End of sentence id; Default is 3')
    parser.add_argument('--num_classes', default=-1, type=int,
                        help='Number of classes; To be specified for each dataset')

    # Device & Seed & Logging
    parser.add_argument('--device', default='cuda', type=str,
                        help='Device to use for training; Default is cuda')
    parser.add_argument('--seed', default=2022, type=int,
                        help='Random seed; Default is 2022')
    parser.add_argument('--use_tensorboard', default=True, type=bool,
                        help='Using tensorboard; Default is True')
    parser.add_argument('--tensorboard_path', default='./tensorboard_runs', type=str,
                        help='Tensorboard log path; Default is ./tensorboard_runs/')
    parser.add_argument('--log_freq', default=500, type=int,
                        help='Logging frequency; Default is 500')

    args = parser.parse_args()

    main(args)
