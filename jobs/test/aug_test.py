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
from model.augmentation.aug_model import AugmentationModel as Model
from model.augmentation.dataset import AugmentationDataset as Dataset
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

    # Get Loss function
    ReconLoss = nn.NLLLoss(ignore_index=args.pad_id)

    # Start testing
    test_recon_loss = 0
    test_kl_loss = 0
    test_acc = 0
    write_log(logger, "Start testing")
    for iter_idx, data_dicts in enumerate(tqdm(dataloader_test, total=len(dataloader_test), desc=f'Testing')):
        # Test - Get batched data from dataloader
        input_seq = data_dicts['Text_Tensor'].to(device)
        input_label = data_dicts['Label_Tensor'].to(device)
        target = input_seq[:, 1:].contiguous() # Remove <bos> token
        target = target.view(-1) # Flatten target for NLLLoss

        # Test - Forward pass
        with torch.no_grad():
            output_prob, output_seq, mu, logvar = model.inference(input_seq, input_label)
            #output_prob, mu, logvar = model(input_seq, input_label)

            # Remove padding to calculate accuracy
            output_prob = output_prob.view(-1, args.vocab_size)
            non_pad_mask = target.ne(args.pad_id)
            output_prob = output_prob[non_pad_mask]
            target = target[non_pad_mask]
            accuracy = (output_prob.argmax(dim=-1) == target).float().mean()

            recon_loss = ReconLoss(output_prob, target)
            if args.variational_type in ['vae', 'cvae']:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            else:
                kl_loss = torch.Tensor([0]).to(device)
            total_loss = recon_loss + kl_loss

        # Test - Logging
        test_recon_loss += recon_loss.item()
        test_kl_loss += kl_loss.item()
        test_acc += accuracy.item()
        if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_test) - 1:
            write_log(logger, f"TEST - Iter [{iter_idx}/{len(dataloader_test)}] - Loss: {total_loss.item():.4f}")
            write_log(logger, f"TEST - Iter [{iter_idx}/{len(dataloader_test)}] - Acc: {accuracy.item():.4f}")

    # Test - Check loss and accuracy
    test_recon_loss /= len(dataloader_test)
    test_kl_loss /= len(dataloader_test)
    test_acc /= len(dataloader_test)
    write_log(logger, f"TEST - Total Loss: {test_recon_loss + test_kl_loss:.4f}")
    write_log(logger, f"TEST - Total Acc: {test_acc:.4f}")

    # Test - Log to tensorboard
    if args.use_tensorboard:
        writer.add_scalar('TEST/Loss/Reconstruction', test_recon_loss, 0)
        writer.add_scalar('TEST/Loss/KL', test_kl_loss, 0)
        writer.add_scalar('TEST/Loss/Total', (test_recon_loss+test_kl_loss), 0)
        writer.add_scalar('TEST/Accuracy', test_acc, 0)
        writer.add_text('TEST/Result', f'Total Loss: {test_recon_loss + test_kl_loss:.4f} / Total Acc: {test_acc:.4f}', 0)

    writer.close()
