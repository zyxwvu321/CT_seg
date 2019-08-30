#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:13:53 2019

@author: lab
"""

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


fd_h5_truth = './data/h5_rsa'
fd_h5_pred  = 'checkpoints/3dunet_tvfl1_3dseres_c1b2_fd_rs_upad_upsamp/h5_pred'
fd_png_pred  = 'checkpoints/3dunet_tvfl1_3dseres_c1b2_fd_rs_upad_upsamp/png_pred'

#h5_list = list(Path(fd_h5_truth).glob('*.h5'))
h5_t = Path('data/h5_rsa/p_13.h5')
fn = h5_t.stem
fn_pred = Path(fd_h5_pred)/(fn + '_predictions.h5')


with h5py.File(str(h5_t), 'r') as ft:

    xyzs = ft['xyz'][...] 


with h5py.File(str(fn_pred), 'r') as fp:

    lbs = fp['predictions'][...][0] 


ws,we,hs,he,ds,de,hh,ww,dd = xyzs
zz = 0
lb_z_crp =  lbs[zz]
img = np.zeros((hh,ww),dtype = 'uint8')

img[hs:he+1,ws:we+1] = (lb_z_crp*255).astype('uint8')





with h5py.File(h5_t, 'r') as f:
    label = f['label'][...]



label_t = label.astype('int')
mask_pred = (lbs>0.5).astype('int')

cm = confusion_matrix(label_t.flatten(),  mask_pred.flatten()) 
fn = [( (label_t[i]==1)   * (mask_pred[i]==0)).sum() for i in range(label_t.shape[0])]
fp = [( (label_t[i]==0)   * (mask_pred[i]==1)).sum() for i in range(label_t.shape[0])]
#for h5_t in tqdm(h5_list):
#    fn = h5_t.stem
#    fn_pred = Path(fd_h5_pred)/(fn + '_predictions.h5')
#    
#    with h5py.File(str(h5_t), 'r') as ft:
#
#        xyzs = ft['xyz'][...] 
#
#
#    with h5py.File(str(fn_pred), 'r') as fp:
#
#        lbs = fp['predictions'][...][0] 
#
#    
#    ws,we,hs,he,ds,de,hh,ww,dd = xyzs
#    
#    png_fd = Path(fd_png_pred)/fn
#    png_fd.mkdir(parents=True,exist_ok = True)
#    
#    
#    for zz in range(xyzs[-1]):
#        lb_z_crp =  lbs[zz]
#        img = np.zeros((hh,ww),dtype = 'uint8')
#        
#        img[hs:he+1,ws:we+1] = (lb_z_crp*255).astype('uint8')
#        cv2.imwrite(str(png_fd/ (str(zz) +'.png')), img)
#        
#        