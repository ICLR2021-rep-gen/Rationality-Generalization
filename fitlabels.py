import argparse, os
import torch
import torch.nn as nn
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR10
from tqdm import tqdm

import numpy as np
import numpy.random as npr

from data_settings import get_dataset
from model_settings import Net
from optims_lr import get_optimizers, get_lr_scheduler
from model import Model
from regularizarion import l1_regularization
import wandb

import copy
from complexity import estimate_complexity, estimate_complexity_new, predict


# train or test for one epoch
def train_val(net, data_loader, train_optimizer, l1_weight=0):
    is_train = train_optimizer is not None
    net.train() if is_train else net.eval()

    total_loss, total_correct_1, total_correct_5, total_correct_1_clean, total_correct_5_clean, total_num, data_bar = 0.0, 0.0, 0.0, 0.0, 0.0, 0, tqdm(data_loader)
    with (torch.enable_grad() if is_train else torch.no_grad()):
        for data, target, clean_target in data_bar:
            data, target, clean_target = data.cuda(non_blocking=True), target.cuda(non_blocking=True), clean_target.cuda(non_blocking=True)
            out = net(data)

            loss = loss_criterion(out, target)
            if l1_weight != 0 and is_train:
                loss += l1_regularization(net.eval_net, l1_weight)

            if is_train:
                train_optimizer.zero_grad()
                loss.backward()
                train_optimizer.step()

            total_num += data.size(0)
            total_loss += loss.item() * data.size(0)
            prediction = torch.argsort(out, dim=-1, descending=True)

            ## Noisy lables
            total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            ## Clean labels

            total_correct_1_clean += torch.sum((prediction[:, 0:1] == clean_target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
            total_correct_5_clean += torch.sum((prediction[:, 0:5] == clean_target.unsqueeze(dim=-1)).any(dim=-1).float()).item()

            data_bar.set_description('{} Epoch: [{}/{}] Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
                                     .format('Train' if is_train else 'Test', epoch, epochs, total_loss / total_num,
                                             total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))

    return total_loss / total_num, total_correct_1 / total_num * 100, total_correct_5 / total_num * 100, total_correct_1_clean / total_num * 100, total_correct_5_clean / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation')

    ### Feature options
    parser.add_argument('--from_features', action='store_true')
    parser.add_argument('--feature_path', type=str, help='Path to trained features')
    parser.add_argument('--local_path', type=str, help='Path to download trained features if GCS')
    parser.add_argument('--model_path', type=str, help='Path to pre-trained model')
    parser.add_argument('--dataname', required=True, type=str, help='CIFAR10/CIFAR100/ImageNet')

    parser.add_argument('--complexity_cmi', action='store_true')
    
    ### Training options
    parser.add_argument('--batch_size', type=int, default=512, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', type=int, default=100, help='Number of sweeps over the dataset to train')
    
    # Optimizer
    parser.add_argument('--optimname', type=str, help='sgd, momentum, adam')
    parser.add_argument('--momentum', type=float, help='Momentum for SGD+momentum')
    parser.add_argument('--nesterov', action='store_true')
    parser.add_argument('--beta1',default=0.9, type=float, help='beta1 for Adam')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for Adam')

    # Learning rate
    parser.add_argument('--lr_sched_type', help='const, sqrtT')
    parser.add_argument('--lr', type=float, help='const, sqrtT LR: init LR value')
    parser.add_argument('--lr_gamma', type=float, help='Learning rate drops as 1/(T)**lr_gamma')
    parser.add_argument('--lr_drop_factor', type=float, help='decay: Factor to multiply LR by at for MultiStepLR')
    parser.add_argument('--lr_epoch_list', type=int, default = 0, nargs='*', help='at_epoch: Epochs to change the learning rate to value from previous argument')

    ### Evaluation options
    parser.add_argument('--eval_loss', default='ce', help="MSE/CE")
    parser.add_argument('--train_noise_prob', default=0., type=float, help='Noise added to labels')
    parser.add_argument('--eval_type', type=str, help='linear, fc')
    parser.add_argument('--weight_decay', type=float, help='L2 regularization value', default=0.)
    parser.add_argument('--l1_regularization', type=float, help='weight of L1 regularization', default=0.)

    # Non-linear evaluation options
    parser.add_argument('--eval_arch', type=str, help='Specify a predefined architecture type')
    parser.add_argument('--eval_nl_width', type=int, help='Eval FC network width - same for all layers')
    parser.add_argument('--eval_nl_depth', type=int, help='Eval FC network depth')
    parser.add_argument('--eval_nl_bn', action='store_true', help='Add BN to non-linear eval network')

    # wandb settings
    parser.add_argument('--log_predictions', action='store_true')
    parser.add_argument('--wandb_project_name', type=str)
    parser.add_argument('--savepath', type=str)    

    args = parser.parse_args()
    print(args)

    ######## Initial setup
    # Wandb

    wandb.init(project=args.wandb_project_name)
    wandb.config.update(args)
    wandb_run_id = wandb.run.get_url().split('/')[-1]

    # Data storage
    results_dir = os.path.join(args.savepath, args.wandb_project_name, wandb_run_id)
    try:
        os.makedirs(results_dir)
    except OSError:
        pass

    batch_size, epochs = args.batch_size, args.epochs    

    #######################################################
    ######### Get the dataset with noise added ############
    #######################################################
    train_data, test_data, num_classes, noise_inds, feature_size = get_dataset(args)
    
    # Make dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    train_noisy_loader = DataLoader(train_data, batch_size=batch_size, sampler=SubsetRandomSampler(noise_inds),
                                    shuffle=False, num_workers=16, pin_memory=True)
    ####### Model
    model = Net(num_class=num_classes, args=args, feature_size=train_data.features.shape[1], ).cuda()
    print(model)

    ###### Optimizer + LR schedule
    assert args.weight_decay == 0 or args.l1_regularization == 0, "We probably don't want both l1 and l2 regularization"
    optimizer = get_optimizers(args, model.eval_net) #initial_lr = 1.
    lr_scheduler = get_lr_scheduler(args, optimizer)

    for param_group in optimizer.param_groups:
        if args.weight_decay is not None: param_group['weight_decay'] = args.weight_decay

    ##### Loss
    if args.eval_loss == 'ce':
        loss_criterion = nn.CrossEntropyLoss()
    elif args.eval_loss == 'mse':
        lossfn = nn.MSELoss()
        loss_criterion = lambda prediction, target: lossfn(prediction, torch.eye(num_classes, device=target.device)[target])
    else:
        raise NotImplementedError

    ## Log init -- TODO: Make a function for this
    epoch = 0
    train_loss, train_acc_1, train_acc_5, train_clean_acc_1, train_clean_acc_5 = train_val(model, train_loader, None)
    test_loss, test_acc_1, test_acc_5, _, _ = train_val(model, test_loader, None)
    if args.train_noise_prob > 0.:
        train_noisy_loss, _, _, train_noisy_acc_1, train_noisy_acc_5 = train_val(model, train_noisy_loader, None)

    metrics = {}
    metrics['Linear Train Loss'] = train_loss
    metrics['Linear Test Loss'] = test_loss
    
    metrics['Linear Test Acc @ 1'] = test_acc_1    
    metrics['Linear Train Acc @ 1'] = train_acc_1
    metrics['Linear Train Acc @ 1 - Clean'] = train_clean_acc_1

    if args.train_noise_prob > 0.:
        metrics['Linear Noisy Acc @ 1 - Clean'] = train_noisy_acc_1

    wandb.log(metrics)    

    best_acc = 0.0
    for epoch in range(1, epochs + 1):


        train_loss, train_acc_1, train_acc_5, train_clean_acc_1, train_clean_acc_5 = train_val(model,
                                                                                               train_loader,
                                                                                               optimizer,
                                                                                               args.l1_regularization)
        lr_scheduler.step()
        test_loss, test_acc_1, test_acc_5, _, _ = train_val(model, test_loader, None)
        if args.train_noise_prob > 0.:        
            train_noisy_loss, _, _, train_noisy_acc_1, train_noisy_acc_5 = train_val(model, train_noisy_loader, None)
        if test_acc_1 > best_acc:
            best_acc = test_acc_1
            print(os.path.join(results_dir, 'eval_model.pt'))
            try:
                torch.save(model.state_dict(), os.path.join(results_dir, 'eval_model.pt'))
            except:
                pass

        metrics = {}
        metrics['Linear Train Loss'] = train_loss
        metrics['Linear Test Loss'] = test_loss

        metrics['Linear Test Acc @ 1'] = test_acc_1    
        metrics['Linear Train Acc @ 1'] = train_acc_1
        metrics['Linear Train Acc @ 1 - Clean'] = train_clean_acc_1

        if args.train_noise_prob > 0.:
            metrics['Linear Noisy Acc @ 1 - Clean'] = train_noisy_acc_1

        wandb.log(metrics)

    #####3 Log predictions
    if args.log_predictions:
        preds = predict(model, train_data.features)
        wandb.run.summary["Predictions"] = preds
        wandb.run.summary["Clean Labels"] = train_data.clean_targets.numpy()
        wandb.run.summary["Noisy Labels"] = np.array(train_data.targets)
        wandb.run.summary.update()

    if args.train_noise_prob > 0.:
        if args.complexity_cmi:
            complexity = estimate_complexity_new(model, train_data, num_classes)
        else:
            complexity = estimate_complexity(model, train_data, num_classes)
        wandb.run.summary["Single Run Complexity"] = np.sqrt(0.5*complexity)/args.train_noise_prob
