import torch
from torch import nn as nn
from torch.nn import functional as F
import math
class RunningBatchNorm3d(nn.Module):
    def __init__(self, nf, mom=0.1, eps=1e-5):
        super().__init__()
        self.mom,self.eps = mom,eps
        self.mults = nn.Parameter(torch.ones (nf,1,1,1))
        self.adds = nn.Parameter(torch.zeros(nf,1,1,1))
        self.register_buffer('sums', torch.zeros(1,nf,1,1,1))
        self.register_buffer('sqrs', torch.zeros(1,nf,1,1,1))
        self.register_buffer('batch', torch.tensor(0.))
        self.register_buffer('count', torch.tensor(0.))
        self.register_buffer('step', torch.tensor(0.))
        self.register_buffer('dbias', torch.tensor(0.))

    def update_stats(self, x):
        bs,nc,*_ = x.shape
        self.sums.detach_()
        self.sqrs.detach_()
        dims = (0,2,3,4)
        s = x.sum(dims, keepdim=True)
        ss = (x*x).sum(dims, keepdim=True)
        c = self.count.new_tensor(x.numel()/nc)
        if bs==1:
            mom1 = self.mom
        else:
            mom1 = 1 - (1-self.mom)/math.sqrt(bs-1)
        self.mom1 = self.dbias.new_tensor(mom1)
        self.sums.lerp_(s, self.mom1)
        self.sqrs.lerp_(ss, self.mom1)
        self.count.lerp_(c, self.mom1)
        self.dbias = self.dbias*(1-self.mom1) + self.mom1
        self.batch += bs
        self.step += 1

    def forward(self, x):
        if self.training: self.update_stats(x)
        sums = self.sums
        sqrs = self.sqrs
        c = self.count
        if self.step<100:
            sums = sums / self.dbias
            sqrs = sqrs / self.dbias
            c    = c    / self.dbias
        means = sums/c
        vars = (sqrs/c).sub_(means*means)
        if bool(self.batch < 20): vars.clamp_min_(0.01)
        x = (x-means).div_((vars.add_(self.eps)).sqrt())
        return x.mul_(self.mults).add_(self.adds)
def create_conv(in_channels, out_channels, kernel_size, order, num_groups,stride = 1, padding=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int): add zero-padding to the input

    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of gatchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias, padding=padding,stride = stride)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            assert not is_before_conv, 'GroupNorm MUST go after the Conv3d'
            # number of groups must be less or equal the number of channels
            if out_channels < num_groups:
                num_groups = out_channels
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                #modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
                modules.append(('batchnorm', RunningBatchNorm3d(in_channels)))
            else:
                #modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
                modules.append(('batchnorm', RunningBatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules

class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter

    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        add stride, remove maxpooling
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='clg', num_groups=8, stride = 1,padding=1,**kwargs):
        super(SingleConv, self).__init__()
        if kernel_size ==1:
            padding = 0
            
        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, stride = stride,padding=padding):
            self.add_module(name, module)
         
class SELayer(nn.Module):

    def __init__(self, channels, dims=3, reduction=16):
        super(SELayer, self).__init__()
        if dims==2:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)            
            self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        elif dims== 3:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, padding=0)            
            self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, padding=0)
        else:
            raise ValueError("dims of SE only =2/3")
            
            
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x
    
    
class SEResNetBlock(nn.Module):
    """
    Basic UNet block consisting of a SingleConv followed by the residual block.    
    Notice we use ELU instead of ReLU (order='cge') and put non-linearity after the groupnorm.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='clg', num_groups=8, stride = 1,  use_SE = True, SE_reduction = 8,**kwargs):
        #use pytorch setting, stride is applied in first conv
        
        super(SEResNetBlock, self).__init__()
        
        self.use_SE = use_SE
        
        # first convolution
        self.conv1 = SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order, num_groups=num_groups,stride = stride)

        n_order = order
        for c in 'rel':
            n_order = n_order.replace(c, '')
                # residual block
        self.conv2 = SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=n_order, num_groups=num_groups)
        # remove non-linearity from the 3rd convolution since it's going to be applied after adding the residual

        # create non-linearity separately
        if 'l' in order:
            self.non_linearity = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif 'e' in order:
            self.non_linearity = nn.ELU(inplace=True)
        else:
            self.non_linearity = nn.ReLU(inplace=True)
        
        if use_SE:
            self.SElayer = SELayer(channels = out_channels, reduction = SE_reduction)
                    
        if in_channels != out_channels:
            self.downsample = SingleConv(in_channels, out_channels, kernel_size=1, order=n_order, num_groups=num_groups,stride = stride,padding=0)
            
        else:
            self.downsample = lambda x: x
        
    def forward(self, x):
        
        residual = self.downsample(x)
        out = self.conv1(x)
        
        out = self.conv2(out)
        
        if self.use_SE:
            out = self.SElayer(out)
        
        out += residual
        out = self.non_linearity(out)

        return out
    


class Encoder(nn.Module):
    """
    A single module from the encoder path
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        conv_kernel_size (int): size of the convolving kernel
        
        basic_module(nn.Module):  SEResNetBlock 
        conv_layer_order (string): determines the order of layers
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 basic_module=SEResNetBlock, conv_layer_order='clg',num_groups=8, stride = 1, use_SE = True, SE_reduction = 8):
        super(Encoder, self).__init__()
        self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         stride = stride, use_SE = use_SE, SE_reduction = SE_reduction)
    def forward(self, x):
        x = self.basic_module(x)
        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsample layer
    (either learned ConvTranspose3d or interpolation) followed by a DoubleConv
    module.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        basic_module(nn.Module): either ResNetBlock or DoubleConv
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, encode_channels,  out_channels, kernel_size=3,
                 basic_module=SEResNetBlock, conv_layer_order='clg', stride = 1, 
                 num_groups=8,use_deconv = True, use_SE = True, SE_reduction = 8):
        super(Decoder, self).__init__()
        self.stride = stride
        
        if not use_deconv or stride==1:
            self.upsample = None 
            in_channels = in_channels + encode_channels
            #TODO: use PixelShuffle_ICNR to upscale featuremap
            
        else:
            # otherwise use ConvTranspose3d
            
            self.upsample = nn.ConvTranspose3d(in_channels,
                                               in_channels//2,
                                               kernel_size=2,
                                               stride=stride,
                                               padding=0,
                                               output_padding=0)
            # adapt the number of in_channels for the ExtResNetBlock
            in_channels = in_channels//2 + encode_channels

        self.basic_module = basic_module(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         order=conv_layer_order,
                                         num_groups=num_groups,
                                         use_SE = use_SE, SE_reduction = SE_reduction) #default stride= 1

    def forward(self, encoder_features, x):
        if self.upsample is None:
            #trilinear
            if self.stride>1:
                x = F.interpolate(x, scale_factor = self.stride , mode='trilinear') #'bilinear?

        else:
            # use ConvTranspose3d
            x = self.upsample(x)

        x = torch.cat((encoder_features, x), dim=1)
        
        x = self.basic_module(x)
        return x


class FinalConv(nn.Sequential):
    """
    A module consisting of a convolution layer (e.g. Conv3d+ReLU+GroupNorm3d) and the final 1x1 convolution
    which reduces the number of channels to 'out_channels'.
    with the number of output channels 'out_channels // 2' and 'out_channels' respectively.
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be change however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ReLU use order='cbr'.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
        num_groups (int): number of groups for the GroupNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='clg', num_groups=8):
        super(FinalConv, self).__init__()

        # conv1
        self.add_module('SingleConv', SingleConv(in_channels, in_channels, kernel_size, order, num_groups))

        # in the last layer a 1×1 convolution reduces the number of output channels to out_channels
        final_conv = nn.Conv3d(in_channels, out_channels, 1)
        self.add_module('final_conv', final_conv)
        
def icnr(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function."
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(torch.zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    k = k.contiguous().view([nf,ni,h,w]).transpose(0, 1)
    x.data.copy_(k)