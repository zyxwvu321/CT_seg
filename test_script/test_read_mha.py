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

fn1 = 'resources/CT/01/data/A_1.mha'
fn2 = 'resources/CT/01/data/PV_1.mha'
fn3 = 'resources/CT/01/data/PV_2.mha'




dicom_fd = './resources/CT/01/data/V'
out_fd =  './resources/CT/01/data/Vout'
os.makedirs(out_fd,exist_ok = True)






itkimage = sitk.ReadImage(fn1)
ct_scan = sitk.GetArrayFromImage(itkimage)[::-1,:,:]
origin = np.array(list(reversed(itkimage.GetOrigin())))
spacing = np.array(list(reversed(itkimage.GetSpacing())))




itkimage1 = sitk.ReadImage(fn2)
ct_scan1 = sitk.GetArrayFromImage(itkimage1)
origin1= np.array(list(reversed(itkimage1.GetOrigin())))
spacing1 = np.array(list(reversed(itkimage1.GetSpacing())))




itkimage2 = sitk.ReadImage(fn3)
ct_scan2 = sitk.GetArrayFromImage(itkimage2)
origin2= np.array(list(reversed(itkimage2.GetOrigin())))
spacing2 = np.array(list(reversed(itkimage2.GetSpacing())))





flist = [str(fn) for fn in Path(dicom_fd).glob('*')]
#%%
for fn in flist:
    
    with pydicom.dcmread(fn) as dc:
        img_dicom = (dc.pixel_array).copy()
        
        rs = int(dc.RescaleSlope)
        ri = int(dc.RescaleIntercept)
        wc = 80#int(dc.WindowCenter)
        wh = 240#int(dc.WindowWidth)
        img_dicom = rs * img_dicom + ri
        img_dicom =  (img_dicom.astype('float') - (wc - wh/2.0))/wh
        #img_dicom = (img_dicom/2500.0).astype('float32')
        img_dicom = np.clip(img_dicom,0.0,1.0)   
    
    fname = Path(fn).stem
    idx = int(Path(fn).stem.replace('IM',''))
    
    
    
    out_fn_im = str(Path(out_fd)/(fname + '.png'))
    out_fn_mask1 = str(Path(out_fd)/(fname + '_m1.png'))
    out_fn_mask2 = str(Path(out_fd)/(fname + '_m2.png'))
    out_fn_mask3 = str(Path(out_fd)/(fname + '_m3.png'))
    

    cv2.imwrite(out_fn_im, (img_dicom*255.0).astype('uint8')) # save in lossless format to avoid colors changing
    
    mask1 = ct_scan[idx]

    cv2.imwrite(out_fn_mask1, (mask1*255.0).astype('uint8')) # save in lossless format to avoid colors changing


#    mask2 = ct_scan1[idx]/144.0
#    #color2 = cv2.applyColorMap(mask2, cv2.COLORMAP_JET)
#    cv2.imwrite(out_fn_mask2, (mask2*255.0).astype('uint8')) # save in lossless format to avoid colors changing
#    
#    mask3 = ct_scan2[idx]/8.0
#    #color3 = cv2.applyColorMap(mask1, cv2.COLORMAP_JET)
#    cv2.imwrite(out_fn_mask3, (mask3*255.0).astype('uint8')) # save in lossless format to avoid colors changing