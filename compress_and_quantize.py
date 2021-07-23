#!/usr/bin/python3

import argparse
import os
import torch

from compress import Compress, bcolors
from quantize import Quantize


def main():
    parser = argparse.ArgumentParser(description='3DDFA Compression')
    parser.add_argument('-a', '--arch', default='mobilenet_1', type=str,
            help='3DDFA achitecure')
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str,
            help='Checkpoint of pre-trained model')
    parser.add_argument('-v2', default='', type=str,
            help='Config file of 3DDFA V2 model')
    parser.add_argument('-t', '--type', default='channel', type=str,
            help='Type of compression, e.g., spatial, weight or channel')
    parser.add_argument('--auto', action='store_true',
            help='Automated compression')
    args = parser.parse_args()


    ## Compression ##
    print(bcolors.OKBLUE + '\n### COMPRESSION ###' + bcolors.ENDC)
    compress_tool = Compress(args.checkpoint_fp, args.arch, args.v2)

    # Evaluate inital model
    print(bcolors.YELLOW + '\nEvaluation of original model' + bcolors.ENDC)
    compress_tool.evaluate_model()

    # Spatial SVD compresion
    if args.type == 'spatial':
        print(bcolors.YELLOW + '\nSpatial SVD compresion' + bcolors.ENDC)
        if args.auto:
            model = compress_tool.spatial_svd_auto_mode()
        else:
            model = compress_tool.spatial_svd_manual_mode()
    # Weight SVD compression
    elif args.type == 'weight':
        print(bcolors.YELLOW + '\nWeight SVD compression' + bcolors.ENDC)
        if args.auto:
            model = compress_tool.weight_svd_auto_mode()
        else:
            model = compress_tool.weight_svd_manual_mode()
    # Channel Pruning
    elif args.type == 'channel':
        print(bcolors.YELLOW + '\nChannel Pruning' + bcolors.ENDC)
        if args.auto:
            model = compress_tool.channel_pruning_auto_mode()
        else:
            model = compress_tool.channel_pruning_manual_mode()
    else:
        print('\nOption not available')


    ## Quantization ##
    print(bcolors.OKBLUE + '\n### QUANTIZATION ###' + bcolors.ENDC)
    quantize_tool = Quantize(args.checkpoint_fp, args.arch, args.v2, input_shape=(1, 3, 120, 120),
                             external_model=model)

    # Evaluate inital model
    print(bcolors.YELLOW + '\nEvaluation of compressed model' + bcolors.ENDC)
    quantize_tool.evaluate_model()

    # Equalize initial model
    print(bcolors.YELLOW + '\nEqualizing model' + bcolors.ENDC)
    quantize_tool.cross_layer_equalization_auto()
    # Evaluate after equalization
    quantize_tool.evaluate_model()

    # Bias correction
    print(bcolors.YELLOW + '\nBias correction' + bcolors.ENDC)
    quantize_tool.bias_correction_empirical()
    # Evaluate after equalization
    quantize_tool.evaluate_model()

    # Quantize equalized model
    print(bcolors.YELLOW + '\nQuantizing model' + bcolors.ENDC)
    quantize_tool.quantsim_model(trainer_function=None, prefix='quantised_post')
    # Evaluate after quantization
    quantize_tool.evaluate_model()
    
    filename = f'{args.type}_compression_quant_checkpoint.pth.tar'
    state = {'state_dict': model.state_dict()}
    torch.save(state, os.path.join('q_models', filename))
    print(f'Model stored in {filename}')

if __name__ == "__main__":
    main()
