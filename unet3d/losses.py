import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import importlib
from unet3d.utils import upad_batch, expand_targ

    


#def compute_per_channel_tversky(input, target, smooth = 1.0, ignore_index=None, weight=None, alpha = 0.5,beta = 0.5,sigmoid_normalization = False):
#    # assumes that input is a normalized probability
#    # alpha=beta=0.5 : dice coefficient
#    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
#    # alpha+beta=1   : produces set of F*-scores
#    # implemented by E. Moebel, 06/04/18
#
#    
#    targs = expand_targ(input,target,sigmoid_normalization)
#        
#    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"
#
#    # mask ignore_index if present
#    if ignore_index is not None:
#        mask = torch.ones_like(target)
#        mask[:,ignore_index,:,:,:] = 0.0
#        
#        mask.requires_grad = False
#
#        input = input * mask
#        target = target * mask
#
#    input = flatten(input)
#    target = flatten(targs).float()
#
#    
#    n_class = input.shape[0]
#    tversky_loss = torch.zeros(n_class)
#    
#    for idx in range(n_class):
#        if weight is None:
#            tp = (input[idx]*target[idx]).sum()
#            fn = ((1.0-input[idx])*target[idx]).sum()
#            fp = ((1.0-target[idx])*input[idx]).sum()
#        else:
#            tp = (input[idx]*target[idx] * weight.flatten()).sum()
#            fn = ((1.0-input[idx])*target[idx] * weight.flatten()).sum()
#            fp = ((1.0-target[idx])*input[idx] * weight.flatten()).sum()
#        
#        
#        tversky_loss[idx] = (tp+smooth)/(tp + alpha *fn + beta*fp + smooth)
#        
#        
#    return tversky_loss
def compute_per_channel_tversky(input, target, smooth = 1.0, ignore_index=None, weight=None, alpha = 0.5,beta = 0.5,sigmoid_normalization = False,
                                pad_width = None, slices = None,shape = None):
    # assumes that input is a normalized probability
    # alpha=beta=0.5 : dice coefficient
    # alpha=beta=1   : tanimoto coefficient (also known as jaccard)
    # alpha+beta=1   : produces set of F*-scores
    # implemented by E. Moebel, 06/04/18

    
    targs = expand_targ(input,target,sigmoid_normalization)
        
    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = torch.ones_like(target)
        mask[:,ignore_index,:,:,:] = 0.0
        
        mask.requires_grad = False

        input = input * mask
        target = target * mask

   
    nB,nC,nD,nH,nW = input.shape
    
    tversky_loss = torch.zeros(nC).type_as(input)
#    for bb in range(nB):
#        prob = input[bb]
#        targ = targs[bb]
#        if weight is not None:
#            ws = weight[bb]
#        
#        if pad_width is not None:
#            prob = upad_batch(prob, slices[bb],shape[bb], pad_width=pad_width)
#            targ = upad_batch(targ, slices[bb],shape[bb], pad_width=pad_width)
#            if weight is not None:
#                ws = upad_batch(ws, slices[bb],shape[bb], pad_width=pad_width)
#
#        for idx in range(nC):
#            if weight is None:
#                tp = (prob[idx]*targ[idx]).sum()
#                fn = ((1.0-prob[idx])*targ[idx]).sum()
#                fp = ((1.0-targ[idx])*prob[idx]).sum()
#            else:
#                tp = (prob[idx]*targ[idx] * ws).sum()
#                fn = ((1.0-prob[idx])*targ[idx] * ws).sum()
#                fp = ((1.0-targ[idx])*prob[idx] *ws).sum()
#        
#            tversky_loss[idx] += (tp+smooth)/(tp + alpha *fn + beta*fp + smooth)
#   tversky_loss = tversky_loss/ nB    
    
    # a test merge tp/fp/fn for all batch in tv calc.
    for idx in range(nC):
        tp,fn,fp = 0.0,0.0,0.0
        for bb in range(nB):
            prob = input[bb]
            targ = targs[bb]
            if weight is not None:
                ws = weight[bb]
            
            if pad_width is not None:
                prob = upad_batch(prob, slices[bb],shape[bb], pad_width=pad_width)
                targ = upad_batch(targ, slices[bb],shape[bb], pad_width=pad_width)
                if weight is not None:
                    ws = upad_batch(ws, slices[bb],shape[bb], pad_width=pad_width)
    
            
            if weight is None:
                tp += (prob[idx]*targ[idx]).sum()
                fn += ((1.0-prob[idx])*targ[idx]).sum()
                fp += ((1.0-targ[idx])*prob[idx]).sum()
            else:
                tp += (prob[idx]*targ[idx] * ws).sum()
                fn += ((1.0-prob[idx])*targ[idx] * ws).sum()
                fp += ((1.0-targ[idx])*prob[idx] *ws).sum()
            
        tversky_loss[idx] += (tp+smooth)/(tp + alpha *fn + beta*fp + smooth)
        
    
    return tversky_loss



#class TverskyLoss(nn.Module):
#    """Computes TverskyLoss,
#    Additionally allows per-pixel weights to be provided.
#    
#    """
#
#    def __init__(self, epsilon=1.0,  ignore_index=None, sigmoid_normalization=True,
#                 skip_first_target=True,use_log = False, alpha = 0.5,beta = 0.5,loss_gain = 1.0, **kwargs):
#        super(TverskyLoss, self).__init__()
#        self.epsilon = epsilon
#        self.alpha = alpha
#        self.beta = beta
#        
#        self.ignore_index = ignore_index
#        self.skip_first_target = skip_first_target
#        self.loss_gain = loss_gain
#        self.use_log = use_log
#
#
#
#        self.sigmoid_normalization = sigmoid_normalization
#        if sigmoid_normalization:
#            self.normalization = nn.Sigmoid()
#        else:
#            self.normalization = nn.Softmax(dim=1)
#
#
#
#    def forward(self, input, target,weight = None):
#        # get probabilities from logits
#        input = self.normalization(input)
#        if weight is not None:
#            weight.requires_grad_(False)
#
#        
#        
#        per_channel_dice = compute_per_channel_tversky(input, target, smooth=self.epsilon, ignore_index=self.ignore_index,
#                                                    weight=weight,alpha = self.alpha,beta =  self.beta,sigmoid_normalization = self.sigmoid_normalization)
#        
#        if self.use_log:
#            per_channel_dice = -torch.log(per_channel_dice + 1e-5)
#        else:
#            per_channel_dice = 1.0 - per_channel_dice
#        
#        #print(per_channel_dice)
#        # Average the Dice score across all channels/classes
#        if self.skip_first_target:
#            return self.loss_gain*torch.mean(per_channel_dice[1:])
#        else:
#            return self.loss_gain*torch.mean(per_channel_dice)

class TverskyLoss(nn.Module):
    """Computes TverskyLoss,
    Additionally allows per-pixel weights to be provided.
    
    """

    def __init__(self, epsilon=1.0,  ignore_index=None, sigmoid_normalization=True,
                 skip_first_target=True,use_log = False, alpha = 0.5,beta = 0.5,loss_gain = 1.0, pad_width= [4,4,8], **kwargs):
        super(TverskyLoss, self).__init__()
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        
        self.ignore_index = ignore_index
        self.skip_first_target = skip_first_target
        self.loss_gain = loss_gain
        self.use_log = use_log
        self.pad_width = pad_width



        self.sigmoid_normalization = sigmoid_normalization
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)



    def forward(self, input, target, shape=None, weight = None,slices = None):
        # get probabilities from logits
        input = self.normalization(input)
        if weight is not None:
            weight.requires_grad_(False)

        
        
        per_channel_dice = compute_per_channel_tversky(input, target, smooth=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight,alpha = self.alpha,beta =  self.beta,sigmoid_normalization = self.sigmoid_normalization,
                                                    pad_width = self.pad_width,slices = slices,shape=shape)
        
        if self.use_log:
            per_channel_dice = -torch.log(per_channel_dice + 1e-5)
        else:
            per_channel_dice = 1.0 - per_channel_dice
        
        #print(per_channel_dice)
        # Average the Dice score across all channels/classes
        if self.skip_first_target:
            return self.loss_gain*torch.mean(per_channel_dice[1:])
        else:
            return self.loss_gain*torch.mean(per_channel_dice)



#def compute_per_channel_fl(input, target,  ignore_index=None, weight=None, alpha = 0.75,gamma = 2.0,epsilon = 1e-5,sigmoid_normalization = False):
#    
#    targs = expand_targ(input,target,sigmoid_normalization)
#    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"
#
#    # mask ignore_index if present
#    if ignore_index is not None:
#        mask = torch.ones_like(target)
#        mask[:,ignore_index,:,:,:] = 0.0
#        
#        mask.requires_grad = False
#
#        input = input * mask
#        target = target * mask
#
#    input = flatten(input)
#    target = flatten(targs).float()
#
#    
#    n_class = input.shape[0]
#    fl_loss = torch.zeros(n_class)
#    
#   
#    for idx in range(n_class):
#        if idx==0 and sigmoid_normalization is False:
#            alpha0 = 1.0 - alpha
#        else:
#            alpha0 = alpha
#    
#        pr = input[idx]
#        tt = target[idx]
##        tt = target[idx]
##        pt = pr*tt + (1-pr)*(1-tt)
##        ww = alpha0*tt + (1-alpha0)*(1-tt)
##        
##        ww = ww * (1-pt).pow(gamma)
##        if weight is not None:
##            ww = ww * weight.flatten()
##        
##        ww = ww.detach()
##        
#        
#        logpt = (pr+epsilon).log() * tt + (1.0-pr+epsilon).log() * (1.0 - tt)
#        pt = logpt.exp()
#        logpt = alpha0 *tt *  logpt +  (1.0-alpha0) *(1.0-tt) *  logpt 
#        
#        
#
#        if weight is not None:
#            logpt = logpt * (weight.type_as(input).view(-1).requires_grad_(False))
#
#        fl_loss[idx] = -((1.0-pt)**gamma * logpt).mean()
#         
#    return fl_loss

def compute_per_channel_fl(input, target,  ignore_index=None, weight=None, alpha = 0.75,gamma = 2.0,epsilon = 1e-5,sigmoid_normalization = False,
                           pad_width = None, slices = None,shape = None):
    

    targs = expand_targ(input,target,sigmoid_normalization)    
    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = torch.ones_like(target)
        mask[:,ignore_index,:,:,:] = 0.0
        
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    nB,nC,nD,nH,nW = input.shape
        
    fl_loss = torch.zeros(nC).type_as(input)
    for bb in range(nB):
        prob = input[bb]
        targ = targs[bb]
        if weight is not None:
            ws = weight[bb]
        
        if pad_width is not None:
            prob = upad_batch(prob, slices[bb],shape[bb], pad_width=pad_width)
            targ = upad_batch(targ, slices[bb],shape[bb], pad_width=pad_width)
            if weight is not None:
                ws = upad_batch(ws, slices[bb],shape[bb], pad_width=pad_width)
        
    
        for idx in range(nC):
            if idx==0 and sigmoid_normalization is False:
                alpha0 = 1.0 - alpha
            else:
                alpha0 = alpha
    

            pr = prob[idx]
            tt = targ[idx]
            
            logpt = (pr+epsilon).log() * tt + (1.0-pr+epsilon).log() * (1.0 - tt)
            pt = logpt.exp()
            logpt = alpha0 *tt *  logpt +  (1.0-alpha0) *(1.0-tt) *  logpt 
            

            if weight is not None:
                logpt = logpt * (ws.type_as(pr).requires_grad_(False))
    
            fl_loss[idx] += -((1.0-pt)**gamma * logpt).mean()
        
    fl_loss = fl_loss/ nB    
               
    return fl_loss



#class FocalLoss1(nn.Module):
#    def __init__(self, gamma=2.0, alpha=0.75,  sigmoid_normalization=False,skip_first_target=True,loss_gain = 1.0, **kwargs):
#        super(FocalLoss1, self).__init__()
#        self.gamma = gamma
#        self.alpha = alpha
#        self.loss_gain = loss_gain
#        self.skip_first_target = skip_first_target
#        self.sigmoid_normalization = sigmoid_normalization
#        if sigmoid_normalization:
#            self.normalization = nn.Sigmoid()
#        else:
#            self.normalization = nn.Softmax(dim=1)
##        if isinstance(alpha,float): 
##            self.alpha = torch.Tensor([alpha,1-alpha])
##        if isinstance(alpha,list): 
##            self.alpha = torch.Tensor(alpha)
#    def forward(self, input, target,weight=None):
#
#        input = self.normalization(input)
#        if weight is not None:
#            weight.requires_grad_(False)
#        
#        
#        
#        per_channel_fl = compute_per_channel_fl(input, target, weight=weight,alpha = self.alpha,gamma =  self.gamma, sigmoid_normalization = self.sigmoid_normalization)
#        
#
#        
#        #print(per_channel_dice)
#        # Average the Dice score across all channels/classes
#        if self.skip_first_target:
#            return self.loss_gain*per_channel_fl[1:].sum()
#        else:
#            return self.loss_gain*per_channel_fl.sum()        
        
class FocalLoss1(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75,  sigmoid_normalization=False,skip_first_target=True,loss_gain = 1.0, pad_width= [4,4,8],**kwargs):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_gain = loss_gain
        self.skip_first_target = skip_first_target
        self.sigmoid_normalization = sigmoid_normalization
        self.pad_width = pad_width
        
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
#        if isinstance(alpha,float): 
#            self.alpha = torch.Tensor([alpha,1-alpha])
#        if isinstance(alpha,list): 
#            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target, shape=None, weight = None,slices = None):

        input = self.normalization(input)
        if weight is not None:
            weight.requires_grad_(False)
        
        
        per_channel_fl = compute_per_channel_fl(input, target, weight=weight,alpha = self.alpha,gamma =  self.gamma, sigmoid_normalization = self.sigmoid_normalization,
                                                 pad_width = self.pad_width,slices = slices,shape=shape)
        


        # Average the Dice score across all channels/classes
        if self.skip_first_target:
            return self.loss_gain*per_channel_fl[1:].sum()
        else:
            return self.loss_gain*per_channel_fl.sum()        


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)


class GeneralizedDiceLoss(nn.Module):
    """Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True):
        super(GeneralizedDiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)

        assert input.size() == target.size(), "'input' and 'target' must have the same shape"

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            input = input * mask
            target = target * mask

        input = flatten(input)
        target = flatten(target)

        target = target.float()
        target_sum = target.sum(-1)
        class_weights = Variable(1. / (target_sum * target_sum).clamp(min=self.epsilon), requires_grad=False)

        intersect = (input * target).sum(-1) * class_weights
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            intersect = weight * intersect
        intersect = intersect.sum()

        denominator = ((input + target).sum(-1) * class_weights).sum()

        return 1. - 2. * intersect / denominator.clamp(min=self.epsilon)

#class FocalLoss(nn.Module):
#
#
#    def __init__(self, weight=None, ignore_index=-1):
#        super(FocalLoss, self).__init__()
#        self.register_buffer('weight', weight)
#        self.ignore_index = ignore_index
#
#    def forward(self, input, target):
#        class_weights = self._class_weights(input)
#        if self.weight is not None:
#            weight = Variable(self.weight, requires_grad=False)
#            class_weights = class_weights * weight
#        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)
#
#    @staticmethod
#    def _class_weights(input):
#        # normalize the input first
#        input = F.softmax(input, _stacklevel=5)
#        flattened = flatten(input)
#        nominator = (1. - flattened).sum(-1)
#        denominator = flattened.sum(-1)
#        class_weights = Variable(nominator / denominator, requires_grad=False)
#        return class_weights

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True,loss_gain = 1.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_gain = loss_gain
        if isinstance(alpha,float): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        assert input.dim() == target.dim()+1, "'input' and 'target' dim -1 "
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return self.loss_gain*loss.mean()
        else: 
            return self.loss_gain*loss.sum()


class PixelWiseFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, size_average=True,loss_gain = 1.0):
        super(PixelWiseFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_gain = loss_gain
        
        
        if isinstance(alpha,float): 
            self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): 
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target,weight=None):
        assert input.dim() == target.dim()+1, "'input' and 'target' dim -1 "
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,D,H,W => N,C,D,*H*W
            input = input.transpose(1,2)    # N,C,D*H*W => N,D*H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,D*H*W,C => N*D*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input,dim = 1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input)
            at = self.alpha.gather(0,target.view(-1))
            logpt = logpt * at
            if weight is not None:
                logpt = logpt * (weight.type_as(input).view(-1).requires_grad_(False))

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: 
            return self.loss_gain*loss.mean()
        else: 
            return self.loss_gain*loss.sum()
        
        
class WeightedCrossEntropyLoss(nn.Module):
    """WeightedCrossEntropyLoss (WCE) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        class_weights = self._class_weights(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
            class_weights = class_weights * weight
        return F.cross_entropy(input, target, weight=class_weights, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, _stacklevel=5)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights


class BCELossWrapper:
    """
    Wrapper around BCE loss functions allowing to pass 'ignore_index' as well as 'skip_last_target' option.
    """

    def __init__(self, loss_criterion, ignore_index=-1, skip_last_target=False):
        if hasattr(loss_criterion, 'ignore_index'):
            raise RuntimeError(f"Cannot wrap {type(loss_criterion)}. Use 'ignore_index' attribute instead")
        self.loss_criterion = loss_criterion
        self.ignore_index = ignore_index
        self.skip_last_target = skip_last_target

    def __call__(self, input, target):
        if self.skip_last_target:
            target = target[:, :-1, ...]

        assert input.size() == target.size()

        masked_input = input
        masked_target = target
        if self.ignore_index is not None:
            mask = target.clone().ne_(self.ignore_index)
            mask.requires_grad = False

            masked_input = input * mask
            masked_target = target * mask

        return self.loss_criterion(masked_input, masked_target)




class PixelWiseBCEEntropyLoss(nn.Module):
    def __init__(self):
        super(PixelWiseBCEEntropyLoss, self).__init__()

    def forward(self, preds, targs, weights):
        
        #weights =weights.permute(1,2,3,0)
        #targs =targs.permute(1,2,3,0)
        assert targs.size() == weights.size()
        assert targs.size() == preds.size()
        
        return torch.nn.functional.binary_cross_entropy_with_logits(preds,targs.float(),weight=weights,reduction='mean')
        
        



class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, ignore_index=None):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        self.register_buffer('class_weights', class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, target, weights):
        assert target.size() == weights.size()
        # normalize the input
        log_probabilities = self.log_softmax(input)
        # standard CrossEntropyLoss requires the target to be (NxDxHxW), so we need to expand it to (NxCxDxHxW)
        target = expand_as_one_hot(target, C=input.size()[1], ignore_index=self.ignore_index)
        # expand weights
        weights = weights.unsqueeze(0)
        weights = weights.expand_as(input)

        # mask ignore_index if present
        if self.ignore_index is not None:
            mask = Variable(target.data.ne(self.ignore_index).float(), requires_grad=False)
            log_probabilities = log_probabilities * mask
            target = target * mask

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
            self.register_buffer('class_weights', class_weights)

        # resize class_weights to be broadcastable into the weights
        class_weights = self.class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()


class MSEWithLogitsLoss(MSELoss):
    """
    This loss combines a `Sigmoid` layer and the `MSELoss` in one single class.
    """

    def __init__(self):
        super(MSEWithLogitsLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, target):
        return super().forward(self.sigmoid(input), target)


class TagsAngularLoss(nn.Module):
    def __init__(self, tags_coefficients):
        super(TagsAngularLoss, self).__init__()
        self.tags_coefficients = tags_coefficients

    def forward(self, inputs, targets, weight):
        assert isinstance(inputs, list)
        # if there is just one output head the 'inputs' is going to be a singleton list [tensor]
        # and 'targets' is just going to be a tensor (that's how the HDF5Dataloader works)
        # so wrap targets in a list in this case
        if len(inputs) == 1:
            targets = [targets]
        assert len(inputs) == len(targets) == len(self.tags_coefficients)
        loss = 0
        for input, target, alpha in zip(inputs, targets, self.tags_coefficients):
            loss += alpha * square_angular_loss(input, target, weight)

        return loss

def compute_per_channel_dice(input, target, epsilon=1e-5, ignore_index=None, weight=None):
    # assumes that input is a normalized probability

    # input and target shapes must match, all 5D
    
    if input.dim()==target.dim()+1:
        #one-hot extension
        targs = torch.zeros_like(input)
        targs.scatter_(1, target.unsqueeze(1), 1.0)
    else:
        targs = target
        
    #assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    # mask ignore_index if present
    if ignore_index is not None:
        mask = torch.ones_like(target)
        mask[:,ignore_index,:,:,:] = 0.0
        
        mask.requires_grad = False

        input = input * mask
        target = target * mask

    input = flatten(input)
    target = flatten(targs)

    target = target.float()
    
    n_class = input.shape[0]
    dice_all = torch.zeros(n_class-1)
    
    smooth = 1.0
    alpha = 0.7
    
    
    for idx in range(1, n_class):
        tp = (input[idx]*target[idx]).sum()
        fn = ((1.0-input[idx])*target[idx]).sum()
        fp = ((1.0-target[idx])*input[idx]).sum()
        dice_all[idx-1] = (tp+smooth)/(tp + alpha *fn + (1-alpha)*fp + smooth)
    
    return dice_all
#    # Compute per channel Dice Coefficient
#    intersect = (input * target).sum(-1)
#    if weight is not None:
#        intersect = weight * intersect
#        denominator = (input + target).sum(-1) * weight
#    else:
#        denominator = (input + target).sum(-1)
#    return 2. * intersect / denominator.clamp(min=epsilon)

def square_angular_loss(input, target, weights=None):
    """
    Computes square angular loss between input and target directions.
    Makes sure that the input and target directions are normalized so that torch.acos would not produce NaNs.

    :param input: 5D input tensor (NCDHW)
    :param target: 5D target tensor (NCDHW)
    :param weights: 3D weight tensor in order to balance different instance sizes
    :return: per pixel weighted sum of squared angular losses
    """
    assert input.size() == target.size()
    # normalize and multiply by the stability_coeff in order to prevent NaN results from torch.acos
    stability_coeff = 0.999999
    input = input / torch.norm(input, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    target = target / torch.norm(target, p=2, dim=1).detach().clamp(min=1e-8) * stability_coeff
    # compute cosine map
    cosines = (input * target).sum(dim=1)
    error_radians = torch.acos(cosines)
    if weights is not None:
        return (error_radians * error_radians * weights).sum()
    else:
        return (error_radians * error_radians).sum()


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.reshape(C, -1)


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    assert input.dim() == 4

    shape = input.size()
    shape = list(shape)
    shape.insert(1, C)
    shape = tuple(shape)

    # expand the input tensor to Nx1xDxHxW
    src = input.unsqueeze(0).long()

    if ignore_index is not None:
        # create ignore_index mask for the result
        expanded_src = src.expand(shape)
        mask = expanded_src == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        src = src.clone()
        src[src == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, src, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, src, 1)


SUPPORTED_LOSSES = ['BCEWithLogitsLoss', 'CrossEntropyLoss', 'WeightedCrossEntropyLoss', 'PixelWiseCrossEntropyLoss',
                    'GeneralizedDiceLoss', 'DiceLoss', 'TagsAngularLoss', 'MSEWithLogitsLoss', 'MSELoss',
                    'SmoothL1Loss', 'L1Loss','FocalLoss','PixelWiseFocalLoss','TverskyLoss','FocalLoss1']


def get_loss_criterion(config):
    """
    Returns the loss function based on provided configuration
    :param config: (dict) a top level configuration object containing the 'loss' key
    :return: an instance of the loss function
    """
    #
    


    def _loss_class(class_name):
        m = importlib.import_module('unet3d.losses')
        clazz = getattr(m, class_name)
        return clazz

    
    
    
    if 'multi_loss' in config.keys():
        loss_config = config['multi_loss']
        losses = list()
        for  loss_config0 in loss_config.values():

            
            name = loss_config0['name']
            loss_class = _loss_class(name)
            losses.append( loss_class(**loss_config0))    
        return losses
    
    
    
    loss_config = config['loss']
    assert 'loss' in config, 'Could not find loss function configuration'
    
    name = loss_config['name']
    loss_class = _loss_class(name)
    

    




    ignore_index = loss_config.get('ignore_index', None)
    weight = loss_config.get('loss_weight', None)

    if weight is not None:
        # convert to cuda tensor if necessary
        weight = torch.tensor(weight).to(config['device'])

    if name == 'BCEWithLogitsLoss':
        skip_last_target = loss_config.get('skip_last_target', False)
        if ignore_index is None and not skip_last_target:
            return nn.BCEWithLogitsLoss()
        else:
            return BCELossWrapper(nn.BCEWithLogitsLoss(), ignore_index=ignore_index, skip_last_target=skip_last_target)
    elif name == 'CrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
#    elif name == 'CrossEntropyLoss':
#        if ignore_index is None:
#            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
#        return nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    
    elif name == 'WeightedCrossEntropyLoss':
        if ignore_index is None:
            ignore_index = -100  # use the default 'ignore_index' as defined in the CrossEntropyLoss
        return WeightedCrossEntropyLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'PixelWiseCrossEntropyLoss':
        return PixelWiseCrossEntropyLoss(class_weights=weight, ignore_index=ignore_index)
    
    elif name == 'PixelWiseBCEEntropyLoss':
        return PixelWiseBCEEntropyLoss()
    
    elif name == 'GeneralizedDiceLoss':
        return GeneralizedDiceLoss(weight=weight, ignore_index=ignore_index)
    elif name == 'DiceLoss':
        sigmoid_normalization = loss_config.get('sigmoid_normalization', True)
        skip_last_target = loss_config.get('skip_last_target', False)
        return DiceLoss(weight=weight, ignore_index=ignore_index, sigmoid_normalization=sigmoid_normalization,
                        skip_last_target=skip_last_target)
    elif name == 'TagsAngularLoss':
        tags_coefficients = loss_config['tags_coefficients']
        return TagsAngularLoss(tags_coefficients)
    elif name == 'MSEWithLogitsLoss':
        return MSEWithLogitsLoss()
    elif name == 'MSELoss':
        return MSELoss()
    elif name == 'SmoothL1Loss':
        return SmoothL1Loss()
    elif name == 'L1Loss':
        return L1Loss()
    elif name == 'FocalLoss':
        return FocalLoss()
    elif name == 'PixelWiseFocalLoss':
        return PixelWiseFocalLoss()
    elif name == 'TverskyLoss':
        return loss_class(**loss_config)
#        return TverskyLoss(epsilon = 1.0,sigmoid_normalization=False,skip_first_target=False,use_log = True,
#                           alpha = 0.5,beta = 0.5,loss_gain = 10.0)
    elif name == 'FocalLoss1':
        return loss_class(**loss_config)


    
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'. Supported losses: {SUPPORTED_LOSSES}")
