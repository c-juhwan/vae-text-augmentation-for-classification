# Standard Library Modules
import os
import sys
import pickle
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
from torch.utils.data import DataLoader
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.augmentation.aug_model import AugmentationModel as Model
from model.augmentation.dataset import AugmentationDataset as Dataset
from utils import TqdmLoggingHandler, write_log, get_torch_device, check_path

def inference(args:argparse.Namespace):
    device = get_torch_device(args.device)

    # Define logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_train = Dataset(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'train_{args.max_seq_len}.pkl'))
    dataloader_train = DataLoader(dataset_train, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_train.vocab_size
    args.num_classes = dataset_train.num_classes
    vocabulary = dataset_train.vocabulary
    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_train)} / {len(dataloader_train)}")

    # Get model instance
    write_log(logger, "Building model")
    model = Model(args)

    # Load trained model
    write_log(logger, "Loading trained model")
    load_model_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'final_{args.model_name}.pth.tar')
    model = model.to('cpu')
    checkpoint = torch.load(load_model_name, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model = model.eval()
    write_log(logger, f"Loaded trained model successfully from {load_model_name}")
    del checkpoint

    # Start inference
    total_data_dict = {
        'Text': [],
        'Label': [],
        'Num_Label': args.num_classes,
        'Vocab': vocabulary,
        'Vocab_Size': args.vocab_size
    }
    write_log(logger, "Start inference")
    for iter_idx, data_dicts in enumerate(tqdm(dataloader_train, total=len(dataloader_train), desc=f'Inference for Training data')):
        # Inference - Get batched data from dataloader
        input_seq = data_dicts['Text_Tensor'].to(device)
        input_label = data_dicts['Label_Tensor'].to(device)

        with torch.no_grad():
            output_prob, output_seq, mu, logvar = model.inference(input_seq, input_label)

        # cut off tokens after <eos> token from output_seq
        output_seq = output_seq.view(-1)
        try:
            output_seq = output_seq[:output_seq.tolist().index(args.eos_id)+1]
        except ValueError: # no <eos> token in output_seq
            # change last token to <eos> token
            output_seq[-1] = args.eos_id
            print(vocabulary.lookup_tokens(output_seq.tolist()))
            print(vocabulary.lookup_tokens(input_seq.view(-1).tolist()))

        # padding output_seq to max_seq_len
        output_seq = output_seq.to('cpu')
        output_seq = torch.cat([output_seq, torch.zeros(args.max_seq_len - len(output_seq), dtype=torch.long)], dim=0)

        #print(vocabulary.lookup_tokens(output_seq.tolist()))
        #print(vocabulary.lookup_tokens(input_seq.view(-1).tolist()))

        # Add augmented data to total_data_dict
        total_data_dict['Text'].append(output_seq.to('cpu'))
        total_data_dict['Label'].append(input_label.view(-1).squeeze().to('cpu'))
        # Add original data to total_data_dict
        total_data_dict['Text'].append(input_seq.view(-1).to('cpu'))
        total_data_dict['Label'].append(input_label.squeeze().to('cpu'))

    # Save total_data to pickle file
    check_path(os.path.join(args.result_path, args.task, args.task_dataset))
    with open(os.path.join(args.result_path, args.task, args.task_dataset, f'train_{args.max_seq_len}+model_aug.pkl'), 'wb') as f:
        pickle.dump(total_data_dict, f)

    write_log(logger, "Finished inference")
