#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import copy
from tqdm import tqdm 

from conf import settings
from utils import get_network, get_test_dataloader, get_cal_dataloader

import sys
import os

def test_loop(net, loader, args, desc='Testing'):
    correct_1 = 0.0
    correct_5 = 0.0

    with torch.no_grad():
        for (image, label) in tqdm(loader, desc=desc, total=len(loader)):

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
            
            
            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()
    
    return correct_1, correct_5

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    PATH = os.getenv("DQPATH", '')

    sys.path.append(PATH)
    sys.path.append(os.path.join(PATH, "backend"))

    from backend import convert_model, QuantMode

    corrupt = eval(os.getenv("corrupt", "0")) == 1
    per_channel = eval(os.getenv("ch", "1")) == 1

    sampling_stride = eval(os.getenv("s_s", "1")) 

    range_std = eval(os.getenv("s_s", "(5, 25, 5)"))

    cal_size = eval(os.getenv("cal_size", "256"))
    verb = eval(os.getenv("verb", "0"))


    print()
    print("LOG+++++ CALIBRATE ESTIMATE MODE")
    print("LOG+++++ Corrupt =", corrupt)
    print("LOG+++++ Per_Channel =", per_channel)

    net = get_network(args) 
    net.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    net.eval()

    cifar100_cal_loader = get_cal_dataloader(
        settings.CIFAR100_MEAN,
        settings.CIFAR100_STD,
        cal_size=cal_size,
        num_workers=4,
        batch_size=args.b,
        )

    confss = {
        'global':{
            'def': {'per_channel':per_channel},
            QuantMode.ESTIMATE: {
                'Conv2d': {
                    'e_std': None,
                    'sampling_stride': sampling_stride,
                },
                'Linear': {
                    'e_std': None
                }
            },
            QuantMode.STATIC: {'cal_size': cal_size}

        },
        'layers':{
            '23': {'skip': True},
        }
    }

    # Define the parameter grid for e_std values
    conv_stds = list(range(range_std[0], range_std[1], range_std[2]))
    linear_stds = list(range(range_std[0], range_std[1], range_std[2]))
    
    best_acc_1 = 0.0
    best_params = {'conv_std': None, 'linear_std': None}
    best_correct_1 = 0
    best_correct_5 = 0

    # Perform grid search over all parameter combinations
    for conv_std in conv_stds:
        for linear_std in linear_stds:
            # Create a deep copy of the original config to avoid side effects
            current_conf = copy.deepcopy(confss)
            current_conf['global'][QuantMode.ESTIMATE]['Conv2d']['e_std'] = conv_std
            current_conf['global'][QuantMode.ESTIMATE]['Linear']['e_std'] = linear_std
            
            # Convert the model with current parameters
            c_net = convert_model(net, mode=QuantMode.ESTIMATE, config=current_conf)
            
            # Evaluate the model
            correct_1, correct_5 = test_loop(c_net, cifar100_cal_loader, args, desc=f"Cal ({conv_std}, {linear_std})")
            acc_1 = correct_1 / len(cifar100_cal_loader.dataset)
            
            # Update best results if current accuracy is higher
            if acc_1 > best_acc_1:
                best_acc_1 = acc_1
                best_params['conv_std'] = conv_std
                best_params['linear_std'] = linear_std
                best_correct_1 = correct_1
                best_correct_5 = correct_5
    
    print(best_params)

    # Update the original config with the best parameters
    confss['global'][QuantMode.ESTIMATE]['Conv2d']['e_std'] = best_params['conv_std']
    confss['global'][QuantMode.ESTIMATE]['Linear']['e_std'] = best_params['linear_std']

    # Convert the final model using the best parameters
    c_net = convert_model(net, mode=QuantMode.ESTIMATE, config=confss)

    # Calculate the final accuracy metrics
    correct_1, correct_5 = best_correct_1, best_correct_5
    acc_1 = float(correct_1 / len(cifar100_cal_loader.dataset))
    acc_5 = float(correct_5 / len(cifar100_cal_loader.dataset))


    
