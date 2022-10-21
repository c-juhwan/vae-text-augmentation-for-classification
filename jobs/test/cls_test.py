# Standard Library Modules
import os
import sys
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.classification.cls_model import ClassificationModel as Model
from model.classification.dataset import ClassificationDataset as Dataset
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device

def testing(args:argparse.Namespace):
    device = get_torch_device(args.device)

    # Define logger and tensorboard writer
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter(" %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.use_tensorboard:
        writer = SummaryWriter(os.path.join(args.tensorboard_path, get_tb_exp_name(args)))
        writer.add_text('args', str(args))

    # Load dataset and define dataloader
    write_log(logger, "Loading data")
    dataset_test = Dataset(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'test_{args.max_seq_len}.pkl'))
    dataloader_test = DataLoader(dataset_test, batch_size=args.test_batch_size, num_workers=args.num_workers,
                                 shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_test.vocab_size
    args.num_classes = dataset_test.num_classes
    vocabulary = dataset_test.vocabulary
    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Test dataset size / iterations: {len(dataset_test)} / {len(dataloader_test)}")

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

    # Get loss function
    ClassificationLoss = nn.CrossEntropyLoss()

    # Start testing
    test_loss = 0
    test_acc = 0
    write_log(logger, "Start testing")
    for iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get batched data from dataloader
        input_seq = data_dicts['Text_Tensor'].to(device)
        target = data_dicts['Label_Tensor'].to(device)

        # Test - Forward pass
        with torch.no_grad():
            output = model(input_seq)
            loss = ClassificationLoss(output, target)
            accuracy = (torch.argmax(output, dim=-1) == target).float().mean()

        # Test - Logging
        test_loss += loss.item()
        test_acc += accuracy.item()
        if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{iter_idx}/{len(dataloader_test)}] - Loss: {loss.item():.4f}")
            write_log(logger, f"TEST - Iter [{iter_idx}/{len(dataloader_test)}] - Acc: {accuracy.item():.4f}")

    # Test - Check loss and accuracy
    test_loss /= len(dataloader_test)
    test_acc /= len(dataloader_test)
    write_log(logger, f"TEST - Total Loss: {test_loss:.4f}")
    write_log(logger, f"TEST - Total Acc: {test_acc:.4f}")

    # Test - Log to tensorboard
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss/Classification', test_loss, 0)
        writer.add_scalar('TEST/Accuracy', test_acc, 0)
        writer.add_text('TEST/Result', f'Total Loss: {test_loss:.4f} / Total Acc: {test_acc:.4f}', 0)

    writer.close()
