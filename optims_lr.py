import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau

def get_optimizers(args, model):
    optimizer_alpha = None

    # If lr scheduler is defined with lambda, the lr is multiplied with the initial lr
    if args.lr_sched_type=='const':
        init_lr = 1
    else:
        init_lr = args.lr

    if args.optimname=='sgd':
        optimizer_w = optim.SGD(model.parameters(), lr=init_lr)
    elif args.optimname=='momentum':
        optimizer_w = optim.SGD(model.parameters(), lr=init_lr, momentum=args.momentum, nesterov=args.nesterov)
    elif args.optimname=='adam':
        optimizer_w = optim.Adam(model.parameters(), lr=init_lr, betas=(args.beta1, args.beta2), eps=1e-08)
    else:
        raise NotImplementedError
    
    return optimizer_w

def get_lr_scheduler(args, optimizer):
    if args.lr_sched_type=='const':
        lr_scheduler = lambda epoch: args.lr
        return LambdaLR(optimizer, lr_lambda=lr_scheduler)
    elif args.lr_sched_type=='gammaT':
        lr_scheduler = lambda epoch: 1./((epoch+1)**args.lr_gamma)
        return LambdaLR(optimizer, lr_lambda=lr_scheduler)
    elif args.lr_sched_type=='at_epoch':
        return MultiStepLR(optimizer, args.lr_epoch_list, args.lr_drop_factor)    
    else:
        raise NotImplementedError
