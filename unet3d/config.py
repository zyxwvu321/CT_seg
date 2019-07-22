import argparse

import os
import torch
import yaml

DEFAULT_DEVICE = 'cuda:0'


def load_config():
    parser = argparse.ArgumentParser(description='UNet3D training')
    parser.add_argument('--config', type=str, help='Path to the YAML config file', required=True)
    parser.add_argument('--findlr', type=int, help='run findlr', default = 0)
    
    args = parser.parse_args()
    config = _load_config_yaml(args.config)
    config['findlr'] = args.findlr
    
    # Get a device to train on
    device = config.get('device', DEFAULT_DEVICE)
    config['device'] = torch.device(device)
    
    
    
#    # based on sigmoid setting, predefine some config values
#    
#    if config['model']['final_sigmoid'] is True:
#        #sigmoid
#        
#    else:
#    
    
    
    
    return config


def _load_config_yaml(config_file):
    return yaml.load(open(config_file, 'r'))
