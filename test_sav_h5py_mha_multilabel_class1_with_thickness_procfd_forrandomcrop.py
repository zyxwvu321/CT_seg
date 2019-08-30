#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:36:00 2019
generate h5 file of CT segmentation

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
from pathlib import Path
import os.path as  osp
import pandas as pd

from skimage.morphology import skeletonize_3d
from scipy import signal,ndimage
from skimage import measure
from tools.loggers import call_logger


logger = call_logger('../CTraw/gen_randcrop.log', 'GenData')
fd_all = '../CTraw/48_ct'
mask_h5 = '../CTraw/h5/h5_48'
h5_fd  = '../data/h5_rs48'


#fd_all = '../CTraw/22_ct'
#mask_h5 = '../CTraw/h5/h5_22'
#h5_fd  = '../data/h5_rsa'



debugim_fd  = '../data/debug_img'
name_chs = ['A','V','PV']


n_class = 2
w_class = [1.0, 0.2] #second term not used
w_class_in_roi = [2.0, 2.0]


df_pid = pd.read_csv('../CTraw/48ct_0830.csv',dtype={'patient':str,'value':int,'is_flip':int})
#df_pid = pd.read_csv('../CTraw/22_ct.csv',dtype={'patient':str,'value':int,'is_flip':int})

maskid =df_pid.set_index('patient').T.to_dict('int')['value']
mask_flip =df_pid.set_index('patient').T.to_dict('int')['is_flip']


path_patient_ids = [x/'data' for x in sorted(list(Path(fd_all).glob('*'))) if x.is_dir()]

logger.info(f'Start gen h5 file, total {len(path_patient_ids)} cases')

Path(h5_fd).mkdir(parents=True,exist_ok = True)

for patient_case in path_patient_ids:    
    fn1 = Path(patient_case)/'A_1.mha'
    fn2 = Path(patient_case)/'PV_1.mha'
    dicom_fd = Path(patient_case)
    
    
    p_id = patient_case.parent.stem
    out_h5py =  str(Path(h5_fd)/('p_' + p_id + '.h5'))
    
    
    logger.info(f'Proc Patient id: {p_id}')
    
    
    if os.path.exists(out_h5py)  or p_id not in np.array(df_pid['patient']):
        logger.info('h5 exist or pid not in csv, skip')
        continue

    #%% READ label
    
    mask_h5_fn = Path(mask_h5)/(p_id +'.h5')
    if osp.exists(str(mask_h5_fn)):
        with h5py.File(mask_h5_fn, 'r') as input_file:
            labels_h5 = input_file['mask'][...]
    else:
        raise ValueError('h5 mask file not exist')
    
    if mask_flip[p_id]==0:
    
        labels1 = (labels_h5[::-1,:,:]==maskid[p_id]).astype('int') #dongmai
    else:
        labels1 = (labels_h5==maskid[p_id]).astype('int') #dongmai
    
    
    
    #%% for some mask image , remove mis-label seperate masks, do measure label and select the largest component
    
    labels1_dilate = binary_dilation(labels1,structure = np.ones((5,5,5))).astype(labels1.dtype)
    label_bw =measure.label(labels1_dilate, connectivity=labels1_dilate.ndim)
    
    props_labelbw = measure.regionprops(label_bw)
    
    area_labelbw = [prop.area for prop in props_labelbw]
    if len(area_labelbw)>1:
        logger.info('warning: more than one region, with areas {}, select >=5'.format(area_labelbw))
    
        max_id = np.argmax(np.array(area_labelbw))+1
        
        #labels1 = (labels1==max_id).astype('int64')
        #labels1[label_bw!=max_id] = 0
        for idx,area in enumerate(area_labelbw):
            if area<5:
                labels1[label_bw==idx+1] = 0
        
    #bwlabel skel, delete branch points    
    
#    itkimage1 = sitk.ReadImage(str(fn1))
#    if mask_flip[p_id]==0:
#    
#        labels1 = (sitk.GetArrayFromImage(itkimage1)[::-1,:,:]==maskid[p_id]).astype('int') #dongmai
#    else:
#        labels1 = (sitk.GetArrayFromImage(itkimage1)==maskid[p_id]).astype('int') #dongmai
        
        
    labels1_dilate = binary_dilation(labels1,structure = np.ones((5,5,5))).astype(labels1.dtype)
    labels = labels1.copy()#np.stack((labels0,labels1),axis = 0)
    labels_dilate = (labels1_dilate>0).astype('int')
    logger.info('mask shape: {}'.format(labels1.shape))
    logger.info('Num of Mask Pixel: {}'.format(labels1.sum()))
    
    

    #%% MIN-MAX
    flist = [str(fn) for fn in (Path(dicom_fd)/name_chs[0]).glob('*')]
    n_slice = len(flist)
    row,col = 512,512
    
    
    labels_pos = np.where(labels_dilate==1)
    z_min,z_max = labels_pos[0].min(),labels_pos[0].max()
    y_min,y_max = labels_pos[1].min(),labels_pos[1].max()
    x_min,x_max = labels_pos[2].min(),labels_pos[2].max()
    
    z_min = max(0,z_min - 16)
    z_max = min(labels.shape[0]-1,z_max + 16)
    
    y_min = max(0,y_min - 32)
    y_max = min(labels.shape[1]-1,y_max + 32)
    
    x_min = max(0,x_min - 32)
    x_max = min(labels.shape[2]-1,x_max + 32)
    
    
    ex_x = 16-(x_max-x_min+1)%16
    x_min = x_min - ex_x//2
    x_max = x_max + (ex_x  - ex_x//2)
    ex_y = 16-(y_max-y_min+1)%16
    y_min = y_min - ex_y//2
    y_max = y_max + (ex_y  - ex_y//2)
    
    logger.info(f'mask Xmin = {x_min}, Xmax = {x_max}, Ymin = {y_min}, Ymax = {y_max}, Zmin = {z_min}, Zmax = {z_max}')    
    logger.info(f'mask sz_X = {x_max-x_min + 1}, sz_Y = {y_max - y_min + 1}, sz_Z = {z_max - z_min + 1}')
    xyz = [x_min,x_max,y_min,y_max,z_min,z_max, labels.shape[2],labels.shape[1],labels.shape[0]]


    #%% read dicom files
    raw_im = -np.ones((3,n_slice,row,col),dtype = 'float32')
    for idx,fn in enumerate(flist):
        
        
        fname = Path(fn).stem
        idx = int(Path(fn).stem.replace('IM',''))
        
        with pydicom.dcmread(fn) as dc:
            img_dicom1 = (dc.pixel_array).copy()
            assert dc.PhotometricInterpretation =='MONOCHROME2', 'check PhotometricInterpretation'
            if idx ==0:
                logger.info(f'mode 0 xyz pixelspacing = {dc.PixelSpacing}, {dc.SliceThickness}')
                xy_ps = float(dc.PixelSpacing[0])
                z_ps  = float(dc.SliceThickness)
                
                logger.info(f'A: RescaleSlope  = {dc.RescaleSlope}, RescaleIntercept = {dc.RescaleIntercept}, WindowCenter = {dc.WindowCenter}, WindowWidth = {dc.WindowWidth}')    
            
            
            rs = int(dc.RescaleSlope)
            ri = int(dc.RescaleIntercept)
            wc = 80#int(dc.WindowCenter)
            wh = 240#int(dc.WindowWidth)
            img_dicom1 = rs * img_dicom1 + ri
            img_dicom1 =  (img_dicom1.astype('float') - (wc - wh/2.0))/wh
            #img_dicom = (img_dicom/2500.0).astype('float32')
            #raw_im = np.clip(raw_im,0.0,1.0)-0.5               
            raw_im[0,idx,:,:] = 2*(np.clip(img_dicom1,0.0,3.0)-0.5)  #(-1,?)
 
                    
        
        fn2 = fn.replace('/'+name_chs[0] + '/','/'+name_chs[1] + '/')
        if osp.exists(str(fn2)):
            with pydicom.dcmread(fn2) as dc:
                img_dicom2 = (dc.pixel_array).copy()
                if idx ==0:
                    logger.info(f'mode 1 xyz pixelspacing = {dc.PixelSpacing}, {dc.SliceThickness}')
                    
                    logger.info(f'V: RescaleSlope  = {dc.RescaleSlope}, RescaleIntercept = {dc.RescaleIntercept}, WindowCenter = {dc.WindowCenter}, WindowWidth = {dc.WindowWidth}')    
            
            rs = int(dc.RescaleSlope)
            ri = int(dc.RescaleIntercept)
            wc = 80#int(dc.WindowCenter)
            wh = 240#int(dc.WindowWidth)
            img_dicom2 = rs * img_dicom2 + ri
            img_dicom2 =  (img_dicom2.astype('float') - (wc - wh/2.0))/wh
            #img_dicom = (img_dicom/2500.0).astype('float32')
            #raw_im = np.clip(raw_im,0.0,1.0)-0.5               
            raw_im[1,idx,:,:] = 2*(np.clip(img_dicom2,0.0,3.0)-0.5)  #(-1,?)
            
            
        fn3 = fn.replace('/'+name_chs[0] + '/','/'+name_chs[2] + '/')                
        if osp.exists(str(fn3)):
            with pydicom.dcmread(fn3) as dc:
                img_dicom3 = (dc.pixel_array).copy()
                if idx ==0:
                    logger.info(f'mode 2 xyz pixelspacing = {dc.PixelSpacing}, {dc.SliceThickness}')
            
                    logger.info(f'PV: RescaleSlope  = {dc.RescaleSlope}, RescaleIntercept = {dc.RescaleIntercept}, WindowCenter = {dc.WindowCenter}, WindowWidth = {dc.WindowWidth}')    
            
            rs = int(dc.RescaleSlope)
            ri = int(dc.RescaleIntercept)
            wc = 80#int(dc.WindowCenter)
            wh = 240#int(dc.WindowWidth)
            img_dicom3 = rs * img_dicom3 + ri
            img_dicom3 =  (img_dicom3.astype('float') - (wc - wh/2.0))/wh
            #img_dicom = (img_dicom/2500.0).astype('float32')
            #raw_im = np.clip(raw_im,0.0,1.0)-0.5               
            raw_im[2,idx,:,:] = 2*(np.clip(img_dicom3,0.0,3.0)-0.5)  #(-1,?)
            

    
    
    labels = labels[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]
    labels_dilate = labels_dilate[z_min:z_max+1,y_min:y_max+1,x_min:x_max+1]    
    labels_dilate = labels_dilate - labels
    raw_im = raw_im[:,z_min:z_max+1,y_min:y_max+1,x_min:x_max+1].astype('float32')   
    
    #%% init weight, boundary has higher weight
    weights = w_class[0]*np.ones_like(labels,dtype = 'float32')
    weights[labels_dilate==1] = w_class_in_roi[0]
    
    
    #%% sav images of V/A/PV
    anno_yxz = np.transpose(labels,(1,2,0))
    img_yxz  = np.transpose(raw_im,(2,3,1,0))
    
    for chs in range(n_slice):
        imgs = np.clip((255*(img_yxz[:,:,chs,:] + 1.0)/2.0),0.0,255.0).astype('uint8')
        masks = (255*anno_yxz[:,:,chs]).astype('uint8')
        fd_sav = Path(debugim_fd)/p_id
        os.makedirs(str(fd_sav),exist_ok = True)
        
        cv2.imwrite(str(fd_sav/('IM_' + str(chs) + '_ch1.png')), imgs[:,:,0])
        cv2.imwrite(str(fd_sav/('IM_' + str(chs) + '_ch2.png')), imgs[:,:,1])
        cv2.imwrite(str(fd_sav/('IM_' + str(chs) + '_ch3.png')), imgs[:,:,2])
        cv2.imwrite(str(fd_sav/('IM_' + str(chs) + '_rgb.png')), imgs)
        cv2.imwrite(str(fd_sav/('IM_' + str(chs) + '_mask.png')), masks)
        
    
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

    def det_branchpoint(anno_yxz, skel):
    
        # perform skeletonization
        skel_filt = signal.convolve(skel, np.ones((3,3,3)), mode='same', method='direct')
    
        skel_branch = ((skel_filt >3).astype('int') * skel).astype('int')
        return skel_branch
        
    

    
    skel = skeletonize_3d(anno_yxz.astype('uint8')).astype('int')
    pp_label = np.argwhere(skel)
    pp_label = pp_label[:, [1, 0, 2]] # x-y-z
    
    
    skel_branch = det_branchpoint(anno_yxz, skel)
    
    
    skel_pt = np.argwhere(skel-skel_branch>0.5)
    branch_pt = np.argwhere(skel_branch>0.5)
    
    
    
    dt_anno = ndimage.distance_transform_edt(anno_yxz, sampling=[xy_ps,xy_ps,z_ps])
    
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
    dt_anno_to_skel,dt_anno_to_idx = ndimage.distance_transform_edt(1.0- skel_label, sampling=[xy_ps,xy_ps,z_ps],return_indices=True)
    
    
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
    
    prop_img = np.zeros((prop_np_valid.shape[0], 7),dtype = 'float')
    for idx in range(prop_np_valid.shape[0]):
        n_area = props_pixel_ch1[idx].area
        
        m_ch1  = props_pixel_ch1[idx].mean_intensity
        m_ch2  = props_pixel_ch2[idx].mean_intensity
        m_ch3  = props_pixel_ch3[idx].mean_intensity
        m_thick = prop_np_valid[idx][0]
        n_area_skel =  prop_np_valid[idx][3]
        w_gain  = min(50.0,max(2.0,50.0/ pow(m_thick,2)))
        prop_img[idx] = np.array([n_area_skel,n_area,m_thick,m_ch1,m_ch2,m_ch3,w_gain])
        
    
    weights =np.transpose(weights,(1,2,0))
    
    for idx in range(prop_np_valid.shape[0]):
        w_gain = prop_img[idx][6]  #max(2.0,100.0/ pow(prop_img[idx][2],2))
        
        weights[anno_yxz_label==idx+1] = w_gain
    
    weights = np.transpose(weights,(2,0,1))
    
    np.set_printoptions(suppress= True,precision = 4)
    
    
    df = pd.DataFrame(data= prop_img,columns = ['area_skel','area_ann','thick','m_ch1','m_ch2','m_ch3','w_gain'])
    logger.info('divide area by skel and branch points')
    logger.info(f'total area =  {prop_np_valid.shape[0]},  min thick =  {prop_img[:,2].min()} ,  max thick =  {prop_img[:,2].max()} ')
        
        
    df.to_csv(str(Path(debugim_fd)/('p_' + p_id + '_skel_area.csv')))
    #logger.info(df)
    

    
    #%%
    
    weights = weights.astype('float32')
    labels = labels.astype('uint8')
    
    with h5py.File(out_h5py, 'w') as f:
        f.create_dataset('label', data=labels,compression='lzf') 
        f.create_dataset('raw', data=raw_im,compression='lzf')
        f.create_dataset('weight', data=weights,compression='lzf')
        f.create_dataset('skelpt', data=pp_label)
        f.create_dataset('xyz', data=xyz)
