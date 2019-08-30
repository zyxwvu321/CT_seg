#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:58:02 2019

@author: minjie
"""

import h5py

path = './resources/sample_patch.h5'


path_o = 'sample_patch_o.h5'
with h5py.File(path, 'r') as f:
    label = f['label'][...]
    raw   = f['raw'][...]
    
    
with h5py.File(path, 'r') as f:
    label = f['label'][...]
    raw   = f['raw'][...]



import os



with h5py.File('./resources/ct01m.h5', 'r') as f:
    label = f['label'][...]
    raw   = f['raw'][...]
    weight = f['weight'][...]




with h5py.File('./resources/ct01m_c1.h5', 'r') as f:
    label = f['label'][...]
    raw   = f['raw'][...]
    








from sklearn.metrics import confusion_matrix


with h5py.File('resources/ct01_predictions.h5', 'r') as f:



    mask0 = f['predictions'][...][1]



with h5py.File('resources/ct01.h5', 'r') as f:
    label = f['label'][...]



mask = (mask0>=0.5).astype('int')
label = label.astype('int')
cm = confusion_matrix(label.flatten(),  mask.flatten()) 



import matplotlib.pyplot as plt



idx = 100
plt.imshow(mask0[idx,:,:])
plt.imshow(label[idx,:,:].astype('float'))












#%%


import h5py

with h5py.File('resources/ct01m_predictions.h5', 'r') as f:

    mask0 = f['predictions'][...]

with h5py.File('resources/ct01m.h5', 'r') as f:
    label = f['label'][...]

from sklearn.metrics import confusion_matrix


for idx in range(4):
    mask_t = (mask0[idx]>=0.25).astype('int')
    label_t = label[idx]
    cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
    print(cm)



#%%
import h5py

with h5py.File('resources/ct01m_c1_predictions.h5', 'r') as f:

    mask0 = f['predictions'][...]

with h5py.File('resources/ct01m_c1.h5', 'r') as f:
    label = f['label'][...]

from sklearn.metrics import confusion_matrix



mask_t = (mask0[1]>=0.5).astype('int')
label_t = label
cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
print('ce cm')
print(cm)    
recall = cm[1,1]/(cm[1,0]+cm[1,1])  
precision = cm[1,1]/(cm[0,1]+cm[1,1])  
fscore = 2*recall*precision/(recall+precision)
print(f'recall: {recall:.4f} precision: {precision:.4f} Fscore: {fscore :.4f}')
    



#%%
import h5py

with h5py.File('resources/ct01m_c1_predictions_doubleconv.h5', 'r') as f:

    mask0 = f['predictions'][...]

with h5py.File('resources/ct01m_c1.h5', 'r') as f:
    label = f['label'][...]

from sklearn.metrics import confusion_matrix



mask_t = (mask0[1]>=0.5).astype('int')
label_t = label
cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
print('ce cm')
print(cm)    
recall = cm[1,1]/(cm[1,0]+cm[1,1])  
precision = cm[1,1]/(cm[0,1]+cm[1,1])  
fscore = 2*recall*precision/(recall+precision)
print(f'recall: {recall:.4f} precision: {precision:.4f} Fscore: {fscore :.4f}')
        
#%%
import h5py

with h5py.File('resources/ct01m_c1_predictions.h5', 'r') as f:

    mask0 = f['predictions'][...]

with h5py.File('resources/ct01m_c1.h5', 'r') as f:
    label = f['label'][...]

from sklearn.metrics import confusion_matrix

#%%

mask_t = (mask0[1]>=0.5).astype('int')
label_t = label
cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
print('ce cm')
print(cm)    
recall = cm[1,1]/(cm[1,0]+cm[1,1])  
precision = cm[1,1]/(cm[0,1]+cm[1,1])  
fscore = 2*recall*precision/(recall+precision)
print(f'recall: {recall:.4f} precision: {precision:.4f} Fscore: {fscore :.4f}')
            