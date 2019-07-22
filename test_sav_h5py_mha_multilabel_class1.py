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
from  scipy.ndimage import binary_dilation


fn1 = 'resources/CT/01/data/A_1.mha'
fn2 = 'resources/CT/01/data/PV_1.mha'
dicom_fd = './resources/CT/01/data/'
name_chs = ['V','A','PV']
out_h5py =  './resources/ct01m_c1.h5'


n_class = 2
w_class = [0.1, 0.2]

w_class_in_roi = [1.0, 2.0]


itkimage1 = sitk.ReadImage(fn1)
labels1 = (sitk.GetArrayFromImage(itkimage1)[::-1,:,:]==1).astype('int') #dongmai






#labels0 = (1-labels1).astype('int')


labels1_dilate = binary_dilation(labels1,structure = np.ones((5,5,5))).astype(labels1.dtype)



labels = labels1.copy()#np.stack((labels0,labels1),axis = 0)
labels_dilate = (labels1_dilate>0).astype('int')

#%%
flist = [str(fn) for fn in (Path(dicom_fd)/name_chs[0]).glob('*')]
n_slice = len(flist)
row,col = 512,512



labels_pos = np.where(labels_dilate==1)
z_min,z_max = labels_pos[0].min(),labels_pos[0].max()
y_min,y_max = labels_pos[1].min(),labels_pos[1].max()
x_min,x_max = labels_pos[2].min(),labels_pos[2].max()

z_min = max(0,z_min - 8)
z_max = min(labels.shape[0],z_max + 8)

y_min = max(0,y_min - 16)
y_max = min(labels.shape[1],y_max + 16)

x_min = max(0,x_min - 16)
x_max = min(labels.shape[2],x_max + 16)


ex_x = 16-(x_max-x_min+1)%16
x_min = x_min - ex_x//2
x_max = x_max + (ex_x  - ex_x//2)
ex_y = 16-(y_max-y_min+1)%16
y_min = y_min - ex_y//2
y_max = y_max + (ex_y  - ex_y//2)

#%%


xyz = [x_min,x_max,y_min,y_max,z_min,z_max]

labels = labels[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]

labels_dilate = labels_dilate[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]




#%%
weights = w_class[0]*np.ones_like(labels,dtype = 'float32')


weights[labels_dilate==1] = w_class_in_roi[0]


weights = weights.astype('float32')
labels = labels.astype('uint8')
#%%
raw_im = np.zeros((3,n_slice,row,col),dtype = 'float32')
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
    
raw_im = raw_im[:,z_min:z_max+1,y_min:y_max+1,x_min:x_max+1].astype('float32')    


#%%

#%%
#write data    
#with h5py.File(out_h5py, 'w') as f:
#    f.create_dataset('label', data=labels,compression='lzf') 
#    f.create_dataset('raw', data=raw_im,compression='lzf')
#    f.create_dataset('weight', data=weights,compression='lzf')
#    f.create_dataset('xyz', data=xyz)
#
##%%
#with h5py.File(out_h5py, 'w') as f:
#    f.create_dataset('label', data=labels,compression='gzip') 
#    f.create_dataset('raw', data=raw_im,compression='gzip')
#    f.create_dataset('weight', data=weights,compression='gzip')
#    f.create_dataset('xyz', data=xyz)

#%%
with h5py.File(out_h5py, 'w') as f:
    f.create_dataset('label', data=labels) 
    f.create_dataset('raw', data=raw_im)
    f.create_dataset('weight', data=weights)
    f.create_dataset('xyz', data=xyz)
