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

import json
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
    parser.add_argument('-ignore_quantization', action='store_true', default=False, help='import quantization backend')
    args = parser.parse_args()

    PATH = os.getenv("DQPATH", '')

    sys.path.append(PATH)
    sys.path.append(os.path.join(PATH, "backend"))

    from backend import convert_model, QuantMode

    mode = eval(os.getenv("dq", "0"))
    corrupt = eval(os.getenv("corrupt", "0")) == 1
    per_channel = eval(os.getenv("ch", "1")) == 1

    sampling_stride = eval(os.getenv("s_s", "1")) 
    c_std = eval(os.getenv("c_std", "3"))
    l_std = eval(os.getenv("l_std", "3"))

    cal_size = eval(os.getenv("cal_size", "16"))
    verb = eval(os.getenv("verb", "0"))


    print()
    print("LOG+++++ ESTIMATE MODE =", mode)
    print("LOG+++++ Corrupt =", corrupt)
    print("LOG+++++ Per_Channel =", per_channel)

    net = get_network(args) 
    net.load_state_dict(torch.load(args.weights, map_location=torch.device('cpu')))
    net.eval()

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_MEAN,
        settings.CIFAR100_STD,
        corrupt=corrupt,
        num_workers=4,
        batch_size=args.b,
    )

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
                    'e_std': c_std,
                    'sampling_stride': sampling_stride,
                },
                'Linear': {
                    'e_std': l_std
                }
            },
            QuantMode.STATIC: {'cal_size': cal_size}

        },
        'layers':{
            '23': {'skip': True},
        }
    }
    if mode == 1:
        net = convert_model(
            net,
            mode=QuantMode.ESTIMATE,
            config=confss
        )
        
        for (image, label) in tqdm(cifar100_cal_loader, desc="CALIBRATING", total=len(cifar100_cal_loader)):            
            output = net(image)

    elif mode == 2:
        net = convert_model(
            net,
            mode=QuantMode.DYNAMIC,
            config=confss
        )
    
    elif mode == 3:
        net = convert_model(
            net,
            mode=QuantMode.STATIC,
            config=confss
        )

        for (image, label) in tqdm(cifar100_cal_loader, desc="CALIBRATING", total=len(cifar100_cal_loader)):            
            output = net(image)
    
    net.eval()
    
    correct_1, correct_5 = test_loop(net, cifar100_test_loader, args)

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    
    acc_1 = float(correct_1 / len(cifar100_test_loader.dataset))
    acc_5 = float(correct_5 / len(cifar100_test_loader.dataset))

    print()
    print("Top 1 Acc: ", acc_1)
    print("Top 5 Acc: ", acc_5)
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))

    results_log = f"results/{str(QuantMode(mode)).split('.')[1]}__corr_{corrupt}_per_channel_{per_channel}"
    if mode==0:
        results_log = f"{results_log.split('_per_channel_')[0]}"
    os.makedirs(results_log, exist_ok=True)

    results_file = f"{args.net}_.json"
    if mode==1: #se estimate aggiungo il sampling stride al nome del file
        results_file = f"{results_file.split('.json')[0]}_stride_{sampling_stride}_c{c_std}_l{l_std}.json"
    if mode==3: #se static aggiungo il cal_size al nome del file
        results_file = f"{results_file.split('.json')[0]}_cal_{cal_size}.json"
        
    results = [{"top1": acc_1, "top5": acc_5}]

    with open(os.path.join(results_log, results_file), "w") as fout:
        json.dump(results, fout)

    print()
    print(f"++++++ Saved results in {os.path.join(results_log, results_file)}")
