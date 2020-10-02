import utils
import copy
from torchvision.datasets import CIFAR10

import numpy as np
import numpy.random as npr


def get_dataset(args):
    if args.from_features:
        train_data = utils.trained_features(args.feature_path, train=True)
        test_data = utils.trained_features(args.feature_path, train=False)
        feature_size = train_data.features.shape[1]
    else:
        if args.dataname=='CIFAR10':

            if args.augment:
                tr_transform = utils.train_transform
            else:
                tr_transform = utils.test_transform
        
            train_data = CIFAR10(root='data', train=True, transform=tr_transform, download=True)
            test_data = CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
        else:
            raise NotImplementedError
        feature_size = None
        train_data.clean_targets = copy.deepcopy(train_data.targets)

    num_classes = (args.dataname=='CIFAR10')*10 + (args.dataname=='CIFAR100')*100 + (args.dataname=='ImageNet')*1000

    #### Add noise
    new_labels = np.array(copy.deepcopy(train_data.targets))

    p = args.train_noise_prob

    ####### Noise with probability p
    cm = (p/(num_classes-1))*np.ones((num_classes, num_classes))
    np.fill_diagonal(cm, 1-p)

    for k in range(num_classes):
        k_indices = list(np.where(np.array(train_data.targets)==k)[0])
        new_labels[k_indices] = npr.choice(num_classes, len(k_indices), p=cm[:, k])

    noise_inds = np.where(np.array(train_data.targets)!=new_labels)[0]
    print(f'Number of noisy points {len(noise_inds)}')
    train_data.targets = list(new_labels)
    print(f'Number of noisy points {len(np.where(np.array(train_data.targets)!=np.array(train_data.clean_targets))[0])}')

    return train_data, test_data, num_classes, noise_inds, feature_size
