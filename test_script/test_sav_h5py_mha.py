#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:36:00 2019

@author: minjie
"""
import SimpleITK as sitk
import numpy as np
from pathlib import Path
import os
import cv2
import pydicom
import h5py

fn1 = 'resources/CT/01/data/A_1.mha'
fn2 = 'resources/CT/01/data/PV_1.mha'
dicom_fd = './resources/CT/01/data/'
name_chs = ['V','A','PV']
out_h5py =  './resources/ct01.h5'







itkimage = sitk.ReadImage(fn1)
labels = sitk.GetArrayFromImage(itkimage)[::-1,:,:].astype('int64')






flist = [str(fn) for fn in (Path(dicom_fd)/name_chs[0]).glob('*')]


n_slice = len(flist)

row,col = 512,512

raw_im = np.zeros((3,n_slice,row,col),dtype = 'float32')

labels_pos = np.where(labels==1)

z_min,z_max = labels_pos[0].min(),labels_pos[0].max()
y_min,y_max = labels_pos[1].min(),labels_pos[1].max()
x_min,x_max = labels_pos[2].min(),labels_pos[2].max()

z_min = max(0,z_min - 8)
z_max = min(labels.shape[0],z_max + 8)

y_min = max(0,y_min - 16)
y_max = min(labels.shape[1],y_max + 16)

x_min = max(0,x_min - 16)
x_max = min(labels.shape[2],x_max + 16)


labels = labels[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]



#%%
for fn in flist:
    
    with pydicom.dcmread(fn) as dc:
        img_dicom1 = (dc.pixel_array).copy()
        

    
    fn2 = fn.replace('/'+name_chs[0] + '/','/'+name_chs[1] + '/')
    fn3 = fn.replace('/'+name_chs[0] + '/','/'+name_chs[2] + '/')
    
    with pydicom.dcmread(fn2) as dc:
        img_dicom2 = (dc.pixel_array).copy()
    with pydicom.dcmread(fn3) as dc:
        img_dicom3 = (dc.pixel_array).copy()
        
        
    fname = Path(fn).stem
    idx = int(Path(fn).stem.replace('IM',''))
    
   
    
    raw_im[0,idx,:,:] = img_dicom1
    raw_im[1,idx,:,:] = img_dicom2
    raw_im[2,idx,:,:] = img_dicom3
    
    
rs = int(dc.RescaleSlope)
ri = int(dc.RescaleIntercept)
wc = 80#int(dc.WindowCenter)
wh = 240#int(dc.WindowWidth)
raw_im = rs * raw_im + ri
raw_im =  (raw_im.astype('float') - (wc - wh/2.0))/wh
#img_dicom = (img_dicom/2500.0).astype('float32')
raw_im = np.clip(raw_im,0.0,1.0)-0.5       
    
raw_im = raw_im[:,z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]    
    
with h5py.File(out_h5py, 'w') as f:
    f.create_dataset('label', data=labels) 
    f.create_dataset('raw', data=raw_im)

    