#!/usr/bin/python3

import argparse
import os
import sys
import torch
import torch.utils.data as data
import yaml

# Aimet-related imports
from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.quantsim import QuantizationSimModel
from aimet_torch.quantsim import QuantParams
from aimet_torch.utils import create_fake_data_loader
from aimet_torch import bias_correction

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


class Quantize(object):
    def __init__(self, checkpoint_fp: str, arch: str, v2: str,
                 input_shape: tuple, external_model: torch.nn.Module = None):

        self.input_shape = input_shape

        # Load the model
        self.tddfa = None
        if v2:
            cfg = yaml.load(open(v2), Loader=yaml.SafeLoader)
            self.tddfa = TDDFA(gpu_mode=True, **cfg)
            self.tddfa.load_model(checkpoint_fp)
            self.model = self.tddfa.model
        else:
            model = getattr(models, arch)(num_classes=62)
            model = load_model(model, checkpoint_fp)
            self.model = model.to(torch.device('cuda'))
        
        # To be used inn conjunction with another tool
        if external_model is not None:
            self.model = external_model
            self.tddfa.model = external_model

        self.model.eval()

    def evaluate_model(self):
        model = None if self.tddfa is not None else self.model
        
        benchmark_pipeline('mobilenet_1', '', self.tddfa, model)
    
    @staticmethod
    def forward_pass(model: torch.nn.Module, eval_iterations: int, use_cuda = True) -> float:
        data_loader = get_dataLoader()    
        model.eval()
    
        with torch.no_grad():
            for _, inputs in enumerate(data_loader):
                inputs = inputs[0].cuda()
                output = model(inputs)

    def quantsim_model(self, trainer_function, prefix: str = 'quantised'):
        # Config file copied from AIMET:
        # aimet/TrainingExtensions/common/src/python/aimet_common/quantsim_config/default_config.json
        config_file = 'configs/aimet_default_config.json'
        
        sim = QuantizationSimModel(self.model, default_output_bw=8, default_param_bw=8,
                                   dummy_input=torch.rand(self.input_shape).cuda(),
                                   config_file=config_file)
    
        # Quantize the untrained MNIST model
        sim.compute_encodings(forward_pass_callback=self.forward_pass, forward_pass_callback_args=0)
    
        # Fine-tune the model's parameter using training
        #trainer_function(model=sim.model, epochs=1, num_batches=100, use_cuda=gpu)
    
        # Export the model
        self.model = sim.model
        self.tddfa.model = sim.model
        sim.export(path='./q_models', filename_prefix=prefix, dummy_input=torch.rand(self.input_shape))

    def cross_layer_equalization_auto(self):
        equalize_model(self.model, self.input_shape)

    def bias_correction_empirical(self, weight_bw=8, act_bw=8):
        data_loader = get_dataLoader()

        params = QuantParams(weight_bw=weight_bw, act_bw=act_bw, round_mode="nearest")
    
        # Perform bias correction
        bias_correction.correct_bias(self.model, params, num_quant_samples=1000, data_loader=data_loader, num_bias_correct_samples=512)

    def bias_correction_analytical_and_empirical(self, weight_bw=8, act_bw=8):
        data_loader = get_dataLoader()
    
        # Find all BN + Conv pairs and remaning Conv from empirical BC
        module_prop_dict = bias_correction.find_all_conv_bn_with_activation(model, input_shape=self.input_shape)
    
        params = QuantParams(weight_bw=weight_bw, act_bw=act_bw, round_mode="nearest")
        
    
        # Perform bias correction
        bias_correction.correct_bias(self.model, params, num_quant_samples=1000, data_loader=data_loader, num_bias_correct_samples=512,
                                    conv_bn_dict=module_prop_dict, perform_only_empirical_bias_corr=False)


def main():
    parser = argparse.ArgumentParser(description='3DDFA Benchmark')
    parser.add_argument('-a', '--arch', default='mobilenet_1', type=str,
                        help='3DDFA achitecure')
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str,
                        help='Checkpoint of pre-trained model')
    parser.add_argument('-v2', default='', type=str,
                        help='Config file of 3DDFA V2 model')
    args = parser.parse_args()

    quantize_tool = Quantize(args.checkpoint_fp, args.arch, args.v2, input_shape=(1, 3, 120, 120),
                             external_model=None)

    # Evaluate inital model
    print(bcolors.YELLOW + '\nEvaluation of original model' + bcolors.ENDC)
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

if __name__ == "__main__":
    main()

