#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:25:41 2019

@author: minjie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import importlib


from unet3d.SEUNet3D_blocks import Encoder, Decoder,   SingleConv,SEResNetBlock,FinalConv

def _get_module_name_cls(class_name):
    m = importlib.import_module('unet3d.SEUNet3D_blocks')
    clazz = getattr(m, class_name)
    return clazz
def _kaiming_init_(m: nn.Module):
    if isinstance(m, (nn.Conv3d,nn.Conv2d)):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d, nn.BatchNorm3d, nn.GroupNorm)):
        #pass
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
    
    elif isinstance(m, nn.Linear):    
#        pass
        stdv = 1. / np.sqrt(m.weight.size(1))
        nn.init.uniform_(m.weight, -stdv, stdv)
        if m.bias is not None:
            nn.init.uniform_(m.bias, -stdv, stdv)
            
            
            
class SEResUNet3D(nn.Module):
    """
    SEResUNet3D model implementation
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. NLLLoss (multi-class)
            or BCELoss (two-class) respectively)
        f_maps (int, tuple): number of feature maps at each level of the encoder; if it's an integer the number
            of feature maps is given by the geometric progression: f_maps ^ k, k=1,2,3,4,5
        final_sigmoid (bool): if True apply element-wise nn.Sigmoid after the
            final 1x1 convolution, otherwise apply nn.Softmax. MUST be True if nn.BCELoss (two-class) is used
            to train the model. MUST be False if nn.CrossEntropyLoss (multi-class) is used to train the model.
        conv_layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
        init_channel_number (int): number of feature maps in the first conv layer of the encoder; default: 64
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, encode_channels,   groups, strides, use_SE, SE_reduction,
                 final_sigmoid, kernel_size = 3, conv_layer_order='clg',use_deconv = True, basic_module_name='SEResNetBlock',
                 use_GP = False,group_final_conv = 8, GP_stride = (8,8,8),GP_layer_order = 'cl',GP_channels = (8,8),
                 is_test = False,**kwargs):
        super(SEResUNet3D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.is_test = is_test
        self.use_GP = use_GP
        GP_stride = tuple(GP_stride)
        GP_channels = tuple(GP_channels)
        if use_GP:
            self.ds_GP =  nn.AvgPool3d(GP_stride, stride=GP_stride)
            self.GP_stride =  GP_stride
            self.GP_conv1 = SingleConv(3,GP_channels[0], kernel_size=1, order='cl',stride = 1,padding=0)
            self.GP_conv2 = SingleConv(GP_channels[0],GP_channels[1], kernel_size=1, order='cl',stride = 1,padding=0)
            self.final_conv = nn.Conv3d(encode_channels[0] + GP_channels[1], out_channels, 1)

            
        # create encoder path consisting of Encoder modules. The length of the encoder is equal to `len(encode_channels)`
        # the first encode use SingleConv
        
        encoders = []
        for i, out_feature_num in enumerate(encode_channels):
            if i == 0:
                encoder = Encoder(in_channels, out_feature_num, kernel_size= kernel_size, basic_module=SingleConv,
                                  conv_layer_order=conv_layer_order, num_groups=groups[0],
                                  stride = strides[0], use_SE = use_SE[0], SE_reduction = SE_reduction[0])
            else:
                encoder = Encoder(encode_channels[i - 1], out_feature_num,  kernel_size= kernel_size, basic_module=_get_module_name_cls(basic_module_name),
                                  conv_layer_order=conv_layer_order, num_groups=groups[i],
                                  stride = strides[i],  use_SE = use_SE[i], SE_reduction = SE_reduction[i])
            
            encoders.append(encoder)

        self.encoders = nn.ModuleList(encoders)
        
        
        decoders = []
        
        decode_channels = list(reversed(encode_channels))
        groups = list(reversed(groups))
        strides = list(reversed(strides))
        use_SE = list(reversed(use_SE))
        SE_reduction = list(reversed(SE_reduction))
        
        for i in range(len(decode_channels) - 1):
            decoder = Decoder(decode_channels[i], decode_channels[i + 1], decode_channels[i + 1], kernel_size= kernel_size,basic_module=_get_module_name_cls(basic_module_name),
                              conv_layer_order=conv_layer_order,stride = strides[i],  num_groups=groups[i],
                              use_deconv = use_deconv, use_SE = use_SE[i], SE_reduction = SE_reduction[i])
            decoders.append(decoder)

        self.decoders = nn.ModuleList(decoders)


        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        if use_GP:
            self.x_final_conv = SingleConv( in_channels = decode_channels[-1], out_channels = decode_channels[-1], kernel_size=3, order=conv_layer_order, num_groups=group_final_conv)
        else:
            self.final_conv = FinalConv( in_channels = decode_channels[-1], out_channels = out_channels, kernel_size=3, order=conv_layer_order, num_groups=group_final_conv)
        #self.final_conv = nn.Conv3d(decode_channels[-1], out_channels, 1)

        if final_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Softmax(dim=1)
        
        self.init()

    def forward(self, x,GP = None):
        # encoder part
        encoders_features = []
        for encoder in self.encoders:
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
        
        if GP is not None:
            x = self.x_final_conv(x)
            GP_ds = self.ds_GP(GP)
            GP_ds = self.GP_conv1(GP_ds)
            GP_ds = self.GP_conv2(GP_ds)
            
            GP_out = F.interpolate(GP_ds, scale_factor = self.GP_stride , mode='trilinear') #'bilinear?
            
            
            x = torch.cat((x,GP_out),dim = 1)
            x = self.final_conv(x)
            
        else:
            x = self.final_conv(x)

        # apply final_activation (i.e. Sigmoid or Softmax) only for prediction. 
        if  self.is_test:
            x = self.final_activation(x)

        return x

    def init(self):
        #pass
        #self.two_view_resnet.apply(_kaiming_init_)
        self.encoders.apply(_kaiming_init_)
        self.decoders.apply(_kaiming_init_)
        self.final_conv.apply(_kaiming_init_)
        if self.use_GP:
            self.x_final_conv.apply(_kaiming_init_)
            self.GP_conv1.apply(_kaiming_init_)
            self.GP_conv2.apply(_kaiming_init_)


        
        #self.fc2.apply(_kaiming_init_)    