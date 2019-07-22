#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 10:46:34 2019

@author: minjie
"""

from skimage.morphology import skeletonize_3d
import h5py
from scipy import signal

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt

    
    

import numpy as np

import render_pcl as rpcl

def det_branchpoint(anno_yxz, skel):

    # perform skeletonization
    
    

    skel_filt = signal.convolve(skel, np.ones((3,3,3)), mode='same', method='direct')

    skel_branch = ((skel_filt >3).astype('int') * skel).astype('int')
    return skel_branch
    
    
    



if __name__ == '__main__':
    
    
    #%%
    color_pt = (0,0,1)
    color_pt_skeleton = (1,0,0)
    color_bg = (0.8,0.8,0.9)
    size_win = (1080, 1080)
    pointSize = 4
    pointSize_skeleton = 10
    
    
    
    fpath_h5 = 'resources/ct01m_c1.h5' # input h5 file path
    fpath_ply = 'pc1.ply' # output ply file path
    th_conf = 0.25; # confidence threshold
    id_class = 1; # class id


    with h5py.File(fpath_h5, 'r') as f:
        anno_map = f['label'][...]
        img_slice = f['raw'][...]
    
    
    
    
    
    anno_yxz = np.transpose(anno_map,(1,2,0))
    img_yxz  = np.transpose(img_slice,(2,3,1,0))
    
    
    skel = skeletonize_3d(anno_yxz).astype('int')
    skel_branch = det_branchpoint(anno_yxz, skel)
    #pcRender(pc,color_pt, pointSize, color_bg,size_win)
    
    pc_skeleton = np.argwhere(skel > 0.5)
    
        
    pc_branch = np.argwhere(skel_branch > 0.5)
    
    #rpcl.pcRenderWithSkeletonWithCoord(pc_skeleton, pc_branch, color_pt, color_pt_skeleton, pointSize, pointSize_skeleton, color_bg,size_win)
    
    
    
   #%%% 

    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    skel_pt = np.argwhere(skel-skel_branch>0.5)
    branch_pt = np.argwhere(skel_branch>0.5)

    

    ax.scatter(skel_pt[:,0], skel_pt[:,1], skel_pt[:,2], marker='o')
    ax.scatter(branch_pt[:,0], branch_pt[:,1], branch_pt[:,2], marker='^')
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    
    plt.show()
    
    
    
    from sklearn.cluster import AgglomerativeClustering

    cluster = AgglomerativeClustering(n_clusters=None, affinity='euclidean', linkage='ward',distance_threshold = 3.0)  
    cluster_res = cluster.fit_predict(branch_pt)  
    #TODO, remove branch point in the leaf
    
    
    
    