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
w_class = [1.0, 0.2]

w_class_in_roi = [2.0, 2.0]


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
z_max = min(labels.shape[0]-1,z_max + 8)

y_min = max(0,y_min - 16)
y_max = min(labels.shape[1]-1,y_max + 16)

x_min = max(0,x_min - 16)
x_max = min(labels.shape[2]-1,x_max + 16)


ex_x = 16-(x_max-x_min+1)%16
x_min = x_min - ex_x//2
x_max = x_max + (ex_x  - ex_x//2)
ex_y = 16-(y_max-y_min+1)%16
y_min = y_min - ex_y//2
y_max = y_max + (ex_y  - ex_y//2)

#%%


xyz = [x_min,x_max,y_min,y_max,z_min,z_max, labels.shape[2],labels.shape[1],labels.shape[0]]

labels = labels[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]

labels_dilate = labels_dilate[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]


labels_dilate = labels_dilate - labels

#%%
weights = w_class[0]*np.ones_like(labels,dtype = 'float32')


weights[labels_dilate==1] = w_class_in_roi[0]


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
#%% Now mask branch detect and thickness calc.
from skimage.morphology import skeletonize_3d
from scipy import signal,ndimage
from skimage import measure
import pandas as pd
def det_branchpoint(anno_yxz, skel):

    # perform skeletonization
    skel_filt = signal.convolve(skel, np.ones((3,3,3)), mode='same', method='direct')

    skel_branch = ((skel_filt >3).astype('int') * skel).astype('int')
    return skel_branch
    

anno_yxz = np.transpose(labels,(1,2,0))
img_yxz  = np.transpose(raw_im,(2,3,1,0))


skel = skeletonize_3d(anno_yxz.astype('uint8')).astype('int')
skel_branch = det_branchpoint(anno_yxz, skel)


skel_pt = np.argwhere(skel-skel_branch>0.5)
branch_pt = np.argwhere(skel_branch>0.5)



dt_anno = ndimage.distance_transform_edt(anno_yxz, sampling=[1.0,1.0,1.777])

#max_at_anno = ndimage.filters.maximum_filter(dt_anno, size=(3,3,3))
#b_local_max = (dt_anno==max_at_anno).astype('int')




skel_label = skel-skel_branch
skel_label_bw =measure.label(skel_label, connectivity=anno_yxz.ndim)

props = measure.regionprops(skel_label_bw,dt_anno)
#bwlabel skel, delete branch points

prop_np = list()
prop_np_valid = list()
map_label = np.arange(len(props)+1)

st_idx = 1
for prop in props:
    me_r = prop.mean_intensity
    mi_r = prop.min_intensity
    ma_r = prop.max_intensity
    n_area = prop.area
    
    if (n_area>2 and me_r>4) or (n_area>3):
        # use this seg
        map_label[prop.label] = st_idx
        st_idx = st_idx + 1
        prop_np_valid.append([me_r, mi_r,ma_r,n_area])
        
    else:
        #not valid seg, remove in skel_label
        map_label[prop.label] = 0
        cl = prop.coords.T
        skel_label[cl[0],cl[1],cl[2]] = 0
    
    prop_np.append([me_r, mi_r,ma_r,n_area])
prop_np = np.array(prop_np)
prop_np_valid = np.array(prop_np_valid)

skel_label_bw_map = map_label[skel_label_bw]

#d1 = ndimage.distance_transform_edt(a)
#d2 = ndimage.distance_transform_edt(a, sampling=[2,1])
#%%
dt_anno_to_skel,dt_anno_to_idx = ndimage.distance_transform_edt(1.0- skel_label, sampling=[1.0,1.0,1.777],return_indices=True)


anno_yxz_label = np.zeros_like(anno_yxz)
fg_yxz = np.where(anno_yxz) 
n_point = len(fg_yxz[0])

y_res = dt_anno_to_idx[0, fg_yxz[0],fg_yxz[1],fg_yxz[2]]
x_res = dt_anno_to_idx[1, fg_yxz[0],fg_yxz[1],fg_yxz[2]]
z_res = dt_anno_to_idx[2, fg_yxz[0],fg_yxz[1],fg_yxz[2]]

anno_yxz_idx = skel_label_bw_map[y_res,x_res,z_res]


anno_yxz_label[fg_yxz] = anno_yxz_idx


#%% test pixel value at different labels
props_pixel_ch1 = measure.regionprops(anno_yxz_label,img_yxz[...,0])
props_pixel_ch2 = measure.regionprops(anno_yxz_label,img_yxz[...,1])
props_pixel_ch3 = measure.regionprops(anno_yxz_label,img_yxz[...,2])

# save the region's prop:     area_skel area_ann mean_ch1 mean_ch2 mean_ch3

prop_img = np.zeros((prop_np_valid.shape[0], 6),dtype = 'float')
for idx in range(prop_np_valid.shape[0]):
    n_area = props_pixel_ch1[idx].area
    
    m_ch1  = props_pixel_ch1[idx].mean_intensity
    m_ch2  = props_pixel_ch2[idx].mean_intensity
    m_ch3  = props_pixel_ch3[idx].mean_intensity
    m_thick = prop_np_valid[idx][0]
    n_area_skel =  prop_np_valid[idx][3]

    prop_img[idx] = np.array([n_area_skel,n_area,m_thick,m_ch1,m_ch2,m_ch3])
    

weights =np.transpose(weights,(1,2,0))

for idx in range(prop_np_valid.shape[0]):
    w_gain = max(2.0,100.0/ pow(prop_img[idx][2],2))
    
    weights[anno_yxz_label==idx+1] = w_gain

weights = np.transpose(weights,(2,0,1))

np.set_printoptions(suppress= True,precision = 4)


df = pd.DataFrame(data= prop_img,columns = ['area_skel','area_ann','thick','m_ch1','m_ch2','m_ch3'])
print(df)

#%%

weights = weights.astype('float32')
labels = labels.astype('uint8')

#with h5py.File(out_h5py, 'w') as f:
#    f.create_dataset('label', data=labels,compression='lzf') 
#    f.create_dataset('raw', data=raw_im,compression='lzf')
#    f.create_dataset('weight', data=weights,compression='lzf')
#    f.create_dataset('xyz', data=xyz)
with h5py.File(out_h5py, 'w') as f:
    f.create_dataset('label', data=labels) 
    f.create_dataset('raw', data=raw_im)
    f.create_dataset('weight', data=weights)
    f.create_dataset('xyz', data=xyz)
