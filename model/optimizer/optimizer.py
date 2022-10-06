# Pytorch Modules
import torch

def get_optimizer(model, args) -> torch.optim.Optimizer:
    if args.weight_decay > 0:
        if args.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        elif args.optimizer == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        else:
            raise ValueError(f'Unknown optimizer option {args.optimizer}')
    else:
        if args.optimizer == 'SGD':
            return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'Adam':
            return torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        elif args.optimizer == 'AdamW':
            return torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        else:
            raise ValueError(f'Unknown optimizer option {args.optimizer}')
