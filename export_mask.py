#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 13:56:41 2019

@author: lab
"""

import os

import h5py
import numpy as np
import torch



from pathlib import Path
from tools.loggers import call_logger

#from tools.loggers import call_logger
#logger = utils.get_logger('UNet3DPredictor')

import os.path as osp
from tqdm import tqdm

import pandas as pd
import cv2


fd_h5_truth = '../data/h5_rsa'
fd_h5_pred  = '../checkpoints/3dunet_tvfl1_3dseres_c1b2_fd_rs_upad_upsamp/h5_pred'
fd_png_pred  = '../checkpoints/3dunet_tvfl1_3dseres_c1b2_fd_rs_upad_upsamp/png_pred'

h5_list = list(Path(fd_h5_truth).glob('*.h5'))

for h5_t in tqdm(h5_list):
    fn = h5_t.stem
    fn_pred = Path(fd_h5_pred)/(fn + '_predictions.h5')
    
    with h5py.File(str(h5_t), 'r') as ft:

        xyzs = ft['xyz'][...] 


    with h5py.File(str(fn_pred), 'r') as fp:

        lbs = fp['predictions'][...][0] 

    
    ws,we,hs,he,ds,de,hh,ww,dd = xyzs
    
    png_fd = Path(fd_png_pred)/fn
    png_fd.mkdir(parents=True,exist_ok = True)
    
    
    for zz in range(xyzs[-1]):
        lb_z_crp =  lbs[zz]
        img = np.zeros((hh,ww),dtype = 'uint8')
        
        img[hs:he+1,ws:we+1] = (lb_z_crp*255).astype('uint8')
        cv2.imwrite(str(png_fd/ (str(zz) +'.png')), img)
        
        
        
        
    
#def _get_output_file(dataset,out_path, suffix='_predictions'):
#    #return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'
#    return str(Path(out_path)/(Path(dataset.file_path).stem + suffix + '.h5'))
#
#
#def _get_dataset_names(config, number_of_datasets):
#    dataset_names = config.get('dest_dataset_name')
#    if dataset_names is not None:
#        if isinstance(dataset_names, str):
#            return [dataset_names]
#        else:
#            return dataset_names
#    else:
#        default_prefix = 'predictions'
#        if number_of_datasets == 1:
#            return [default_prefix]
#        else:
#            return [f'{default_prefix}{i}' for i in range(number_of_datasets)]
#
#
#
#def save_predictions(prediction_maps, output_file, dataset_names):
#    """
#    Saving probability maps to a given output H5 file. If 'average_channels'
#    is set to True average the probability_maps across the the channel axis
#    (useful in case where each channel predicts semantically the same thing).
#
#    Args:
#        prediction_maps (list): list of numpy array containing prediction maps in separate channels
#        output_file (string): path to the output H5 file
#        dataset_names (list): list of dataset names inside H5 file where the prediction maps will be saved
#    """
#    assert len(prediction_maps) == len(dataset_names), 'Each prediction map has to have a corresponding dataset name'
#    logger.info(f'Saving predictions to: {output_file}...')
#
#    with h5py.File(output_file, "w") as output_h5:
#        for prediction_map, dataset_name in zip(prediction_maps, dataset_names):
#            #logger.info(f"Creating dataset '{dataset_name}'...")
#            output_h5.create_dataset(dataset_name, data=prediction_map, compression="gzip")
#
#
#
#
#if __name__ == '__main__':   
#    # Load configuration
#    config = load_config()
#    
#    # Load model state
#    model_path = config['model_path']
#    
#    model_fd = Path(model_path).parent
#    
#    
#    logger = call_logger(log_file = str(model_fd/'test_log.txt'),log_name = 'UNetPredict')
#    
#
#
#    if 'output_path' in config.keys():
#        out_path = config['output_path']
#    else: 
#        out_path = str(model_fd/'h5_pred')
#   
#    os.makedirs(out_path,exist_ok = True)
#        
#    
#
#
#    logger.info('Loading HDF5 datasets...')
#    
#    
#    datasets_config = config['datasets']
#
#
#    logger.info('Loading HDF5 datasets...')
#    test_loaders = get_test_loaders(config)
#    l_test_loaders = (list(test_loaders))
#    
#
#    p_ids = list()
#    recpreF1 = list()
#    for test_loader in l_test_loaders:
#        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
#        
#        
#
#        output_file = _get_output_file(test_loader.dataset,out_path)
#        
#        # run the model prediction on the entire dataset and save to the 'output_file' H5
#      
#        dataset_names = _get_dataset_names(config, len(predictions))
#        
#        
#        save_predictions(predictions, output_file, dataset_names)
#        
#        predictionsoutput_file
#        output_file
#        dataset_names
#        ori_h5 = test_loader.dataset.file_path
#        
#        with h5py.File(ori_h5, 'r') as f:
#            label = f['label'][...]
#        
#        from sklearn.metrics import confusion_matrix
#        
#        #%%
#        if config['model']['final_sigmoid']:
#            mask_t = (predictions[0]>=0.5).astype('int')
#        else:
#            mask_t = (predictions[0][1]>=0.5).astype('int')
#        label_t = label
#
#    
#
#
