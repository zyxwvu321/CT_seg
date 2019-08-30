#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:46:34 2019

@author: minjie
"""

from skimage.morphology import skeletonize_3d
import h5py
from scipy import signal

fpath_h5 = 'resources/ct01m_c1.h5' # input h5 file path
fpath_ply = 'pc1.ply' # output ply file path
th_conf = 0.25; # confidence threshold
id_class = 1; # class id


with h5py.File(fpath_h5, 'r') as f:
    anno_map = f['label'][...]
    
    
    
import h5py
import numpy as np


def det_branchpoint(anno_map, skel):
    D,H,W = anno_map.shape
    
    
    

# perform skeletonization
skel = skeletonize_3d(anno_map).astype('int')


det_branchpoint(anno_map, skel)





skel_filt = signal.convolve(skel, np.ones((3,3,3)), mode='same', method='direct')

skel_branch = ((skel_filt >3).astype('int') * skel).astype('int')



#%%
color_pt = (0,0,1)
color_pt_skeleton = (1,0,0)
color_bg = (0.8,0.8,0.9)
size_win = (800, 800)
pointSize = 4
pointSize_skeleton = 10

from render_pcl import pcRenderWithSkeleton
import render_pcl as rpcl

#pcRender(pc,color_pt, pointSize, color_bg,size_win)

pc_skeleton = np.argwhere(skel > 0.5)

    
pc_branch = np.argwhere(skel_branch > 0.5)

rpcl.pcRenderWithSkeletonWithCoord(pc_skeleton, pc_branch, color_pt, color_pt_skeleton, pointSize, pointSize_skeleton, color_bg, size_win)