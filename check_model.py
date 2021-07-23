#!/usr/bin/python3

import argparse
import os
import sys
import torch

# Aimet-related imports
from aimet_torch.model_validator.model_validator import ModelValidator

# TDDFA-related imports
import models
from utils.tddfa_util import load_model


def main():
    parser = argparse.ArgumentParser(description='3DDFA Compression')
    parser.add_argument('-a', '--arch', default='mobilenet_1', type=str,
            help='3DDFA achitecure')
    parser.add_argument('-c', '--checkpoint-fp', default='models/phase1_wpdc_vdc.pth.tar', type=str,
            help='Checkpoint of pre-trained model')
    args = parser.parse_args()
    
    model = getattr(models, args.arch)(num_classes=62)
    model = load_model(model, args.checkpoint_fp)

    ModelValidator.validate_model(model, model_input=torch.rand(1, 3, 120, 120))


if __name__ == "__main__":
    main()
