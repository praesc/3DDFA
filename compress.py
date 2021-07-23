#!/usr/bin/python3

import argparse
from decimal import Decimal
import os
import sys
import torch
import yaml

# Aimet-related imports
from aimet_common.defs import CostMetric, CompressionScheme, GreedySelectionParameters, RankSelectScheme
from aimet_torch.defs import WeightSvdParameters, SpatialSvdParameters, ChannelPruningParameters, \
    ModuleCompRatioPair
from aimet_torch.compress import ModelCompressor

# TDDFA-related imports
import models
from benchmark import benchmark_pipeline, get_dataLoader
from TDDFA_benchmark import TDDFA
from utils.tddfa_util import load_model


class bcolors:
    HEADER = '\033[95m'
    YELLOW = '\033[93m'
    OKBLUE = '\033[96m'
    OKGREEN = '\033[92m'
    ENDC = '\033[0m'


# Evaluation callback for 3DDFA V2
# It needs the full TDDFA object to calculate the accuracy right
def forward_pass_v2(model, eval_iterations: int, use_cuda = True) -> float:
        tddfa.model = model
        results = benchmark_pipeline('mobilenet_1', '', tddfa, None)
        return results[3]


class Compress(object):
    def __init__(self, checkpoint_fp: str, arch: str, v2: str):
        # Load the model
        self.tddfa = None
        if v2:
            cfg = yaml.load(open(v2), Loader=yaml.SafeLoader)
            self.tddfa = TDDFA(gpu_mode=True, **cfg)
            self.tddfa.load_model(checkpoint_fp)
            self.model = self.tddfa.model
            global tddfa
            tddfa = self.tddfa
        else:
            model = getattr(models, arch)(num_classes=62)           
            model = load_model(model, checkpoint_fp)
            self.model = model.to(torch.device('cuda'))

        self.model.eval()

    # Evaluation function for both V1 and V2
    # Calculates the accuracy of the model on AFLW2000
    def evaluate_model(self):
        model = None if self.tddfa is not None else self.model

        # Perform benchmark
        results = benchmark_pipeline('mobilenet_1', '', self.tddfa, model)

    # Evaluation callback for 3DDFA V1
    @staticmethod
    def forward_pass(model, eval_iterations: int, use_cuda = True) -> float:
        results = benchmark_pipeline('mobilenet_1', '', None, model)
        return results[3]

    def spatial_svd_auto_mode(self):
        # Specify the necessary parameters
        greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                                  num_comp_ratio_candidates=20)
        auto_params = SpatialSvdParameters.AutoModeParams(greedy_params,
                                                          modules_to_ignore=[])

        params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.auto,
                                      params=auto_params, multiplicity=8)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(self.model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.spatial_svd,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)     # Stats object can be pretty-printed easily

        return compressed_model

    def spatial_svd_manual_mode(self):
        model = self.model

        # Specify the necessary parameters
        manual_params = SpatialSvdParameters.ManualModeParams([ModuleCompRatioPair(model.conv1, 0.95),
                                                               ModuleCompRatioPair(model.dw2_1.conv_sep, 0.75),
                                                               ModuleCompRatioPair(model.dw2_2.conv_sep, 0.75),
                                                               ModuleCompRatioPair(model.dw3_1.conv_sep, 0.5),
                                                               ModuleCompRatioPair(model.dw3_2.conv_sep, 0.375),
                                                               ModuleCompRatioPair(model.dw4_1.conv_sep, 0.375),
                                                               ModuleCompRatioPair(model.dw4_2.conv_sep, 0.375),
                                                               ModuleCompRatioPair(model.dw5_1.conv_sep, 0.5),
                                                               ModuleCompRatioPair(model.dw5_2.conv_sep, 0.5),
                                                               ModuleCompRatioPair(model.dw5_3.conv_sep, 0.5),
                                                               ModuleCompRatioPair(model.dw5_4.conv_sep, 0.375),
                                                               ModuleCompRatioPair(model.dw5_5.conv_sep, 0.5),
                                                               ModuleCompRatioPair(model.dw5_6.conv_sep, 0.4),
                                                               ModuleCompRatioPair(model.dw6.conv_sep, 0.25)
                                                               ])
        params = SpatialSvdParameters(mode=SpatialSvdParameters.Mode.manual,
                                      params=manual_params)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.spatial_svd,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)    # Stats object can be pretty-printed easily

        return compressed_model

    def weight_svd_auto_mode(self):
        # Specify the necessary parameters
        greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                                  num_comp_ratio_candidates=20)
        rank_select = RankSelectScheme.greedy
        auto_params = WeightSvdParameters.AutoModeParams(rank_select_scheme=rank_select,
                                                         select_params=greedy_params,
                                                         modules_to_ignore=[])

        params = WeightSvdParameters(mode=WeightSvdParameters.Mode.auto,
                                     params=auto_params)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(self.model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.weight_svd,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)     # Stats object can be pretty-printed easily

        return compressed_model

    def weight_svd_manual_mode(self):
        model = self.model

        # Specify the necessary parameters
        manual_params = WeightSvdParameters.ManualModeParams([ModuleCompRatioPair(model.conv1, 0.05),
                                                              ModuleCompRatioPair(model.dw2_1.conv_sep, 0.7),
                                                              ModuleCompRatioPair(model.dw2_2.conv_sep, 0.5),
                                                              ModuleCompRatioPair(model.dw3_1.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw3_2.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw4_1.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw4_2.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw5_1.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw5_2.conv_sep, 0.5),
                                                              ModuleCompRatioPair(model.dw5_3.conv_sep, 0.6),
                                                              ModuleCompRatioPair(model.dw5_4.conv_sep, 0.4),
                                                              ModuleCompRatioPair(model.dw5_5.conv_sep, 0.75),
                                                              ModuleCompRatioPair(model.dw5_6.conv_sep, 0.5),
                                                              ModuleCompRatioPair(model.dw6.conv_sep, 0.2),
                                                              ModuleCompRatioPair(model.fc, 0.5)
                                                             ])
        params = WeightSvdParameters(mode=WeightSvdParameters.Mode.manual,
                                     params=manual_params, multiplicity=8)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.weight_svd,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)    # Stats object can be pretty-printed easily

        return compressed_model

    def channel_pruning_auto_mode(self):
        # Specify the necessary parameters
        greedy_params = GreedySelectionParameters(target_comp_ratio=Decimal(0.8),
                                                  num_comp_ratio_candidates=20)
        auto_params = ChannelPruningParameters.AutoModeParams(greedy_params,
                                                              modules_to_ignore=[self.model.conv1])

        data_loader = get_dataLoader()
        params = ChannelPruningParameters(data_loader=data_loader,
                                          num_reconstruction_samples=500,
                                          allow_custom_downsample_ops=True,
                                          mode=ChannelPruningParameters.Mode.auto,
                                          params=auto_params)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(self.model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.channel_pruning,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)     # Stats object can be pretty-printed easily

        return compressed_model

    def channel_pruning_manual_mode(self):
        model = self.model

        # Specify the necessary parameters
        manual_params = ChannelPruningParameters.ManualModeParams([ModuleCompRatioPair(model.conv1, 1),
                                                                   ModuleCompRatioPair(model.dw2_1.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw2_2.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw3_1.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw3_2.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw4_1.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw4_2.conv_sep, 0.7),
                                                                   ModuleCompRatioPair(model.dw5_1.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw5_2.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw5_3.conv_sep, 0.6),
                                                                   ModuleCompRatioPair(model.dw5_4.conv_sep, 0.7),
                                                                   ModuleCompRatioPair(model.dw5_5.conv_sep, 0.7),
                                                                   ModuleCompRatioPair(model.dw5_6.conv_sep, 0.7),
                                                                   ModuleCompRatioPair(model.dw6.conv_sep, 1)])

        data_loader = get_dataLoader()
        params = ChannelPruningParameters(data_loader=data_loader,
                                          num_reconstruction_samples=500,
                                          allow_custom_downsample_ops=True,
                                          mode=ChannelPruningParameters.Mode.manual,
                                          params=manual_params)

        # Single call to compress the model
        call_back = forward_pass_v2 if self.tddfa is not None else self.forward_pass
        results = ModelCompressor.compress_model(model,
                                                 eval_callback=call_back,
                                                 eval_iterations=1000,
                                                 input_shape=(1, 3, 120, 120),
                                                 compress_scheme=CompressionScheme.channel_pruning,
                                                 cost_metric=CostMetric.mac,
                                                 parameters=params)

        compressed_model, stats = results
        print(stats)    # Stats object can be pretty-printed easily
        
        return compressed_model


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

    filename = f'{args.type}_compression_checkpoint.pth.tar'
    state = {'state_dict': model.state_dict()}
    torch.save(state, os.path.join('q_models', filename))
    print(f'Model stored in {filename}')

if __name__ == "__main__":
    main()
