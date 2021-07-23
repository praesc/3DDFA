#!/usr/bin/env python3
# coding: utf-8

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import time
import numpy as np
import yaml

from models import mobilenet_v1
from benchmark_aflw2000 import calc_nme as calc_nme_alfw2000
from benchmark_aflw2000 import ana as ana_alfw2000
from benchmark_aflw import calc_nme as calc_nme_alfw
from benchmark_aflw import ana as ana_aflw

from utils.ddfa import ToTensorGjz, NormalizeGjz, DDFACustomDataset, DDFATestDataset, reconstruct_vertex
from utils.tddfa_util import load_model
import argparse
from TDDFA_benchmark import TDDFA

cur_dir = os.path.dirname(os.path.abspath(__file__))


def get_dataLoader(root=os.path.join(cur_dir, 'test.data/AFLW2000-3D_crop'), filelists=os.path.join(cur_dir, 'test.data/AFLW2000-3D_crop.list'), 
                   batch_size=128, num_workers=4) -> data.DataLoader:
    dataset = DDFACustomDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def extract_param(checkpoint_fp, root='', filelists=None, arch='mobilenet_1', num_classes=62, device_ids=[0],
                  batch_size=128, num_workers=4, tddfa=None, external_model=None):
    
    
    if tddfa is None and external_model is None:
        model = getattr(mobilenet_v1, arch)(num_classes=num_classes)
        model = load_model(model, checkpoint_fp)
        model = model.to(torch.device('cuda'))
    elif external_model:
        model = external_model
       
    ##### QuantLab #####
    #from models.mobilenetv1_quantlab import MobileNetV1
    #map_location = {f'cuda:{i}': 'cuda:0' for i in range(8)}
    #model = MobileNetV1()
    #checkpoint = torch.load(checkpoint_fp, map_location=map_location)['net']
    #model.cuda()
    ###################
    
    dataset = DDFATestDataset(filelists=filelists, root=root,
                              transform=transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

    cudnn.benchmark = True
    if not tddfa:
        model.eval()

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs.cuda()
            if tddfa:
                output = tddfa(inputs, None)
            else:
                output = model(inputs)           
            for i in range(output.shape[0]):
                param_prediction = output[i].cpu().numpy().flatten()
                if tddfa:
                    param_prediction = param_prediction * tddfa.param_std + tddfa.param_mean
                outputs.append(param_prediction)
        
        outputs = np.array(outputs, dtype=np.float32)
        
    print(f'Extracting params take {time.time() - end: .3f}s')
    return outputs

def _benchmark_aflw(outputs):
    return ana_aflw(calc_nme_alfw(outputs))


def _benchmark_aflw2000(outputs):
    return ana_alfw2000(calc_nme_alfw2000(outputs))


def benchmark_alfw_params(params, tddfa):
    outputs = []
    if tddfa:
        outputs = tddfa.recon_vers(params, None)
        outputs = [output[:2, :] for output in outputs]
    else:
        for i in range(params.shape[0]):
            lm = reconstruct_vertex(params[i])        
            outputs.append(lm[:2, :])
    return _benchmark_aflw(outputs)


def benchmark_aflw2000_params(params, tddfa):
    outputs = []
    if tddfa:
        outputs = tddfa.recon_vers(params, None)
        outputs = [output[:2, :] for output in outputs]
    else:
        for i in range(params.shape[0]):
            lm = reconstruct_vertex(params[i])
            outputs.append(lm[:2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_pipeline(arch, checkpoint_fp, tddfa, model, complete: bool = False):
    device_ids = [0]

    def aflw():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root=os.path.join(cur_dir, 'test.data/AFLW_GT_crop'),
            filelists=os.path.join(cur_dir, 'test.data/AFLW_GT_crop.list'),
            arch=arch,
            device_ids=device_ids,
            batch_size=128,
            tddfa=tddfa,
            external_model=model)

        return benchmark_alfw_params(params, tddfa)

    def aflw2000():
        params = extract_param(
            checkpoint_fp=checkpoint_fp,
            root=os.path.join(cur_dir, 'test.data/AFLW2000-3D_crop'),
            filelists=os.path.join(cur_dir, 'test.data/AFLW2000-3D_crop.list'),
            arch=arch,
            device_ids=device_ids,
            batch_size=128, 
            tddfa=tddfa,
            external_model=model)

        return benchmark_aflw2000_params(params, tddfa)

    results = aflw2000()
    if complete:
        aflw()

    return results


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('--arch', default='mobilenet_1', type=str)
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str)
    parser.add_argument('-v2', default='', type=str)
    args = parser.parse_args()
    
    tddfa = None
    model = None
    if args.v2:
        cfg = yaml.load(open(args.v2), Loader=yaml.SafeLoader)
        tddfa = TDDFA(gpu_mode=True, **cfg)
        tddfa.load_model(args.checkpoint_fp)    

    benchmark_pipeline(args.arch, args.checkpoint_fp, tddfa, model, complete=True)


if __name__ == '__main__':
    main()
