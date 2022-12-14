# Standard Library Modules
import os
import sys
import shutil
import logging
import argparse
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
# Custom Modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.augmentation.aug_model import AugmentationModel as Model
from model.augmentation.aug_model import kl_annealing
from model.augmentation.dataset import AugmentationDataset as Dataset
from model.optimizer.optimizer import get_optimizer
from model.optimizer.scheduler import get_scheduler
from utils import TqdmLoggingHandler, write_log, get_tb_exp_name, get_torch_device, check_path

def training(args:argparse.Namespace) -> None:
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
    dataset_dict, dataloader_dict = {}, {}
    dataset_dict['train'] = Dataset(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'train_{args.max_seq_len}.pkl'))
    dataset_dict['valid'] = Dataset(os.path.join(args.preprocessed_path, args.task, args.task_dataset, f'valid_{args.max_seq_len}.pkl'))
    dataloader_dict['train'] = DataLoader(dataset_dict['train'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=True, pin_memory=True, drop_last=True)
    dataloader_dict['valid'] = DataLoader(dataset_dict['valid'], batch_size=args.batch_size, num_workers=args.num_workers,
                                          shuffle=False, pin_memory=True, drop_last=False)
    args.vocab_size = dataset_dict['train'].vocab_size
    args.num_classes = dataset_dict['train'].num_classes
    vocabulary = dataset_dict['train'].vocabulary
    write_log(logger, "Loaded data successfully")
    write_log(logger, f"Train dataset size / iterations: {len(dataset_dict['train'])} / {len(dataloader_dict['train'])}")
    write_log(logger, f"Valid dataset size / iterations: {len(dataset_dict['valid'])} / {len(dataloader_dict['valid'])}")

    # Get model instance
    write_log(logger, "Building model")
    model = Model(args)
    model = model.to(device)

    # Get optimizer/scheduler/scaler
    write_log(logger, "Building optimizer")
    optimizer = get_optimizer(model, args) # Default is Adam
    scheduler = get_scheduler(optimizer, len(dataloader_dict['train']), args)
    scaler = GradScaler()

    # Get Loss function
    ReconLoss = nn.NLLLoss(ignore_index=args.pad_id)

    # If resume training, load from checkpoint
    start_epoch = 0
    if args.job == 'resume_training':
        write_log(logger, "Resuming training model")
        load_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset,
                                            f'checkpoint_{args.model_name}.pth.tar')
        model = model.to('cpu')
        checkpoint = torch.load(load_checkpoint_name, map_location='cpu')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            scheduler = None
        scaler.load_state_dict(checkpoint['scaler'])
        model = model.to(device)
        write_log(logger, f"Loaded training model from epoch {start_epoch}")
        del checkpoint

    # Start training
    best_epoch_idx = 0
    best_valid_objective_value = None
    early_stopping_counter = 0
    for epoch_idx in range(start_epoch, args.num_epochs):
        # Train model
        model = model.train()
        train_recon_loss = 0
        train_kl_loss = 0
        train_acc = 0
        kl_lambda = 0 if args.variational_type == 'none' else kl_annealing(epoch_idx, args.num_epochs) if args.kl_lambda < 0 else args.kl_lambda

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['train'], total=len(dataloader_dict['train']), desc=f'Training - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Train - Get batched data from dataloader
            input_seq = data_dicts['Text_Tensor'].to(device)
            input_label = data_dicts['Label_Tensor'].to(device)
            target = input_seq[:, 1:].contiguous() # Remove <bos> token
            target = target.view(-1) # Flatten target for NLLLoss

            # Train - Forward pass
            with autocast():
                output_prob, mu, logvar = model(input_seq, input_label)

                # Remove padding to calculate accuracy
                output_prob = output_prob.view(-1, args.vocab_size)
                non_pad_mask = target.ne(args.pad_id)
                output_prob = output_prob[non_pad_mask]
                target = target[non_pad_mask]
                accuracy = (output_prob.argmax(dim=-1) == target).float().mean()

                recon_loss = ReconLoss(output_prob, target) # NLLLoss
                if args.variational_type in ['vae', 'cvae']:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                else:
                    kl_loss = torch.Tensor([0]).to(device)
                total_loss = recon_loss + kl_lambda*kl_loss

            # Train - Backward pass
            optimizer.zero_grad()
            scaler.scale(total_loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            if args.scheduler in ['StepLR', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']:
                scheduler.step() # These schedulers require step() after every training iteration

            # Train - Logging
            train_recon_loss += recon_loss.item()
            train_kl_loss += kl_loss.item()
            train_acc += accuracy.item()
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['train']) - 1:
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Loss: {total_loss.item():.4f}")
                write_log(logger, f"TRAIN - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['train'])}] - Acc: {accuracy.item():.4f}")

            # Train - Log to tensorboard
            if args.use_tensorboard:
                writer.add_scalar('TRAIN/Learning_Rate', optimizer.param_groups[0]['lr'],
                                  epoch_idx * len(dataloader_dict['train']) + iter_idx)

        if args.use_tensorboard:
            writer.add_scalar('TRAIN/Loss/Reconstruction', train_recon_loss / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Loss/KL', train_kl_loss / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Loss/Total', (train_recon_loss + kl_lambda*train_kl_loss) / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/Accuracy', train_acc / len(dataloader_dict['train']), epoch_idx)
            writer.add_scalar('TRAIN/KL_Lambda', kl_lambda, epoch_idx)

        # Validate model
        model = model.eval()
        valid_recon_loss = 0
        valid_kl_loss = 0
        valid_acc = 0

        for iter_idx, data_dicts in enumerate(tqdm(dataloader_dict['valid'], total=len(dataloader_dict['valid']), desc=f'Validating - Epoch [{epoch_idx}/{args.num_epochs}]')):
            # Validate - Get batched data from dataloader
            input_seq = data_dicts['Text_Tensor'].to(device)
            input_label = data_dicts['Label_Tensor'].to(device)
            target = input_seq[:, 1:].contiguous()
            target = target.view(-1)

            # Validate - Forward pass
            with torch.no_grad():
                output_prob, mu, logvar = model(input_seq, input_label)

                # Remove padding to calculate accuracy
                output_prob = output_prob.view(-1, args.vocab_size)
                target = target.view(-1)
                non_pad_mask = target.ne(args.pad_id)
                output_prob = output_prob[non_pad_mask]
                target = target[non_pad_mask]
                accuracy = (output_prob.argmax(dim=-1) == target).float().mean()

                recon_loss = ReconLoss(output_prob, target) # NLLLoss
                if args.variational_type in ['vae', 'cvae']:
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                else:
                    kl_loss = torch.Tensor([0]).to(device)
                total_loss = recon_loss + kl_lambda*kl_loss

            # Validate - Logging
            valid_recon_loss += recon_loss.item()
            valid_kl_loss += kl_loss.item()
            valid_acc += accuracy.item()
            if iter_idx % args.log_freq == 0 or iter_idx == len(dataloader_dict['valid']) - 1:
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Loss: {total_loss.item():.4f}")
                write_log(logger, f"VALID - Epoch [{epoch_idx}/{args.num_epochs}] - Iter [{iter_idx}/{len(dataloader_dict['valid'])}] - Acc: {accuracy.item():.4f}")

        print(vocabulary.lookup_tokens(output_prob.argmax(dim=-1).tolist()))
        print(vocabulary.lookup_tokens(target.tolist()))

        # Validation - Scheduler
        if args.scheduler == 'LambdaLR':
            scheduler.step()
        elif args.scheduler == 'ReduceLROnPlateau':
            scheduler.step(valid_recon_loss + kl_lambda*valid_kl_loss)

        # Validation - Check loss & Save Checkpoint
        valid_recon_loss /= len(dataloader_dict['valid'])
        valid_kl_loss /= len(dataloader_dict['valid'])
        valid_acc /= len(dataloader_dict['valid'])

        if args.optimize_objective == 'loss':
            valid_objective_value = valid_recon_loss + kl_lambda*valid_kl_loss
            valid_objective_value = -1 * valid_objective_value # Loss is minimized, but we want to maximize the objective value
        elif args.optimize_objective == 'accuracy':
            valid_objective_value = valid_acc

        if best_valid_objective_value is None or valid_objective_value > best_valid_objective_value:
            best_valid_objective_value = valid_objective_value
            best_epoch_idx = epoch_idx
            write_log(logger, f"Saving checkpoint for best valid {args.optimize_objective}...")
            early_stopping_counter = 0

            check_path(os.path.join(args.checkpoint_path, args.task, args.task_dataset))
            save_checkpoint_name = os.path.join(args.checkpoint_path, args.task, args.task_dataset,
                                                f'checkpoint_{args.model_name}.pth.tar')
            torch.save({
                        'epoch': epoch_idx,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict() if scheduler is not None else None,
                        'scaler': scaler.state_dict(),
            }, save_checkpoint_name)
            write_log(logger, f'VALID - Best valid at epoch {best_epoch_idx} - {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')
            write_log(logger, f'VALID - Saved checkpoint to {save_checkpoint_name}')
        else:
            early_stopping_counter += 1
            write_log(logger, f'VALID - Worse than epoch {best_epoch_idx} - Current {args.optimize_objective}: {abs(valid_objective_value):.4f} - Best {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')

        # Validation - Log to tensorboard
        if args.use_tensorboard:
            writer.add_scalar('VALID/Loss/Reconstruction', valid_recon_loss, epoch_idx)
            writer.add_scalar('VALID/Loss/KL', valid_kl_loss, epoch_idx)
            writer.add_scalar('VALID/Loss/Total', (valid_recon_loss + kl_lambda*valid_kl_loss), epoch_idx)
            writer.add_scalar('VALID/Accuracy', valid_acc, epoch_idx)

        # Validation - Early Stopping
        if early_stopping_counter >= args.early_stopping_patience:
            write_log(logger, f'VALID - Early stopping at epoch {epoch_idx}')
            break

    write_log(logger, f'Done! Best epoch: {best_epoch_idx} - Best {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')
    if args.use_tensorboard:
        writer.add_text('VALID/Best', f'Best epoch: {best_epoch_idx} - Best {args.optimize_objective}: {abs(best_valid_objective_value):.4f}')

    # Save best checkpoint model as final model
    check_path(os.path.join(args.model_path, args.task, args.task_dataset))
    save_final_name = os.path.join(args.model_path, args.task, args.task_dataset,
                                   f'final_{args.model_name}.pth.tar')
    shutil.copyfile(save_checkpoint_name, save_final_name)
    write_log(logger, f'Saved final model to {save_final_name}')
    writer.close()
