# Pytorch Modules
import torch
from torch.optim.lr_scheduler import StepLR, LambdaLR, CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau

def get_scheduler(optimizer, dataloader_length, args) -> torch.optim.lr_scheduler:
    if args.scheduler == 'StepLR':
        epoch_step = args.num_epochs // 8
        return StepLR(optimizer, step_size=dataloader_length*epoch_step, gamma=0.8)
    elif args.scheduler == 'LambdaLR':
        lr_lambda = lambda epoch: 0.95 ** epoch
        return LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif args.scheduler == 'CosineAnnealingLR':
        T_max = args.num_epochs // 8
        eta_min = args.learning_rate * 0.01
        return CosineAnnealingLR(optimizer, T_max=dataloader_length*T_max, eta_min=eta_min)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        T_0 = args.num_epochs // 8
        T_mult = 2
        eta_min = args.learning_rate * 0.01
        return CosineAnnealingWarmRestarts(optimizer, T_0=dataloader_length*T_0,
                                           T_mult=dataloader_length*T_mult, eta_min=eta_min)
    elif args.scheduler == 'ReduceLROnPlateau':
        patience = args.early_stopping_patience // 2 if args.early_stopping_patience > 0 else args.num_epochs // 10
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)
    elif args.scheduler == 'None' or args.scheduler is None:
        return None
    else:
        raise ValueError(f'Unknown scheduler option {args.scheduler}')
