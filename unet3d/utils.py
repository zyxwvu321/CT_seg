import logging
import os
import shutil
import sys
import scipy.sparse as sparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import random 
import torch

def save_checkpoint(state, is_best, checkpoint_dir, logger=None):
    """Saves model and training parameters at '{checkpoint_dir}/last_checkpoint.pytorch'.
    If is_best==True saves '{checkpoint_dir}/best_checkpoint.pytorch' as well.

    Args:
        state (dict): contains model's state_dict, optimizer's state_dict, epoch
            and best evaluation metric value so far
        is_best (bool): if True state contains the best model seen so far
        checkpoint_dir (string): directory where the checkpoint are to be saved
    """

    def log_info(message):
        if logger is not None:
            logger.info(message)

    if not os.path.exists(checkpoint_dir):
        log_info(
            f"Checkpoint directory does not exists. Creating {checkpoint_dir}")
        os.mkdir(checkpoint_dir)

    last_file_path = os.path.join(checkpoint_dir, 'last_checkpoint.pytorch')
    #log_info(f"Saving last checkpoint to '{last_file_path}'")
    torch.save(state, last_file_path)
    if is_best:
        best_file_path = os.path.join(checkpoint_dir, 'best_checkpoint.pytorch')
        log_info(f"Saving best checkpoint to '{best_file_path}'")
        shutil.copyfile(last_file_path, best_file_path)


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.

    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied

    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model_state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer_state_dict'])

    return state


def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Logging to console
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s [%(threadName)s] %(levelname)s %(name)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def get_number_of_learnable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def find_maximum_patch_size(model, device):
    """Tries to find the biggest patch size that can be send to GPU for inference
    without throwing CUDA out of memory"""
    logger = get_logger('PatchFinder')
    in_channels = model.in_channels

    patch_shapes = [(64, 128, 128), (96, 128, 128),
                    (64, 160, 160), (96, 160, 160),
                    (64, 192, 192), (96, 192, 192)]

    for shape in patch_shapes:
        # generate random patch of a given size
        patch = np.random.randn(*shape).astype('float32')

        patch = torch \
            .from_numpy(patch) \
            .view((1, in_channels) + patch.shape) \
            .to(device)

        logger.info(f"Current patch size: {shape}")
        model(patch)

def expand_targ(input,target,sigmoid_normalization):
    if input.dim()==target.dim()+1:
        #one-hot extension
        if sigmoid_normalization:
            sz_input = list(input.size())
            sz_input[1] += 1            
            sz_input = tuple(sz_input)
            targs = torch.zeros(sz_input).type_as(input)
            targs.scatter_(1, target.unsqueeze(1), 1.0)
            targs = targs[:,1:]
            
        else:
            targs = torch.zeros_like(input)
            targs.scatter_(1, target.unsqueeze(1), 1.0)
    else:
        targs = target
    return targs
    
def unpad(probs, index, shape, pad_width=(4,8,8)):
    #unpad boundary part... SO... Stride at least <patch_shape -8 !!!!!
    def _new_slices(slicing, max_size,pad_width):
        if slicing.start  <= pad_width:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop >= max_size - pad_width:
            p_stop = None
            i_stop = slicing.stop
        else:
            p_stop = -pad_width if pad_width>0 else None
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, _,i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D,pad_width[0])
    p_y, i_y = _new_slices(i_y, H,pad_width[1])
    p_x, i_x = _new_slices(i_x, W,pad_width[2])

    probs_index = (p_c, p_z, p_y, p_x)
    index0 = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index0

def unpad_eval(probs, index, shape, pad_width=(4,8,8)):
    #unpad boundary part... SO... Stride at least <patch_shape -8 !!!!!
    # for prob map with no batch expand, i.e.   slice nCxnDxnHxnW

    def _new_slices(slicing, max_size,pad_width):
        if slicing.start  <= pad_width:
            p_start = 0
            i_start = 0
        else:
            p_start = pad_width
            i_start = slicing.start + pad_width

        if slicing.stop >= max_size - pad_width:
            p_stop = None
            i_stop = slicing.stop
        else:
            p_stop = -pad_width if pad_width>0 else None
            i_stop = slicing.stop - pad_width

        return slice(p_start, p_stop), slice(i_start, i_stop)

    D, H, W = shape

    i_c, i_z, i_y, i_x = index
    p_c = slice(0, probs.shape[0])

    p_z, i_z = _new_slices(i_z, D,pad_width[0])
    p_y, i_y = _new_slices(i_y, H,pad_width[1])
    p_x, i_x = _new_slices(i_x, W,pad_width[2])

    probs_index = (p_c, p_z, p_y, p_x)
    index0 = (i_c, i_z, i_y, i_x)
    return probs[probs_index], index0

def upad_batch(probs, slices,shape, pad_width=(4,8,8)):
    # for train/valid loss
    
    #unpad boundary part... SO... Stride at least <patch_shape -4,8,8 !!!!!
    # as patch has diff pad, proc one batch each time    
    def _new_slices(slicing, max_size,pad_width):
        if slicing.start <= pad_width:
            p_start = 0
        else:
            p_start = pad_width
            
        if slicing.stop >= max_size - pad_width:
            p_stop = None
        else:
            p_stop = -pad_width if pad_width>0 else None

        return slice(p_start, p_stop)

    D, H, W = shape
    
    _,i_z, i_y, i_x = slices

    p_z = _new_slices(i_z, D,pad_width[0])
    p_y = _new_slices(i_y, H,pad_width[1])
    p_x = _new_slices(i_x, W,pad_width[2])
    
    if probs.dim()==4:    
        nc = probs.shape[0]
        probs_index = (slice(0, nc,),) + (p_z, p_y, p_x)
    elif probs.dim()==3:
        probs_index = (p_z, p_y, p_x)
    else:
        raise ValueError('probs ndim err in loss')
    return probs[probs_index]


def create_feature_maps(init_channel_number, number_of_fmaps):
    return [init_channel_number * 2 ** k for k in range(number_of_fmaps)]


# Code taken from https://github.com/cremi/cremi_python
def adapted_rand(seg, gt, all_stats=False):
    """Compute Adapted Rand error as defined by the SNEMI3D contest [1]
    Formula is given as 1 - the maximal F-score of the Rand index
    (excluding the zero component of the original labels). Adapted
    from the SNEMI3D MATLAB script, hence the strange style.
    Parameters
    ----------
    seg : np.ndarray
        the segmentation to score, where each value is the label at that point
    gt : np.ndarray, same shape as seg
        the groundtruth to score against, where each value is a label
    all_stats : boolean, optional
        whether to also return precision and recall as a 3-tuple with rand_error
    Returns
    -------
    are : float
        The adapted Rand error; equal to $1 - \frac{2pr}{p + r}$,
        where $p$ and $r$ are the precision and recall described below.
    prec : float, optional
        The adapted Rand precision. (Only returned when `all_stats` is ``True``.)
    rec : float, optional
        The adapted Rand recall.  (Only returned when `all_stats` is ``True``.)
    References
    ----------
    [1]: http://brainiac2.mit.edu/SNEMI3D/evaluation
    """
    # just to prevent division by 0
    epsilon = 1e-6

    # segA is truth, segB is query
    segA = np.ravel(gt)
    segB = np.ravel(seg)
    n = segA.size

    n_labels_A = np.amax(segA) + 1
    n_labels_B = np.amax(segB) + 1

    ones_data = np.ones(n)

    p_ij = sparse.csr_matrix((ones_data, (segA[:], segB[:])), shape=(n_labels_A, n_labels_B))

    a = p_ij[1:n_labels_A, :]
    b = p_ij[1:n_labels_A, 1:n_labels_B]
    c = p_ij[1:n_labels_A, 0].todense()
    d = b.multiply(b)

    a_i = np.array(a.sum(1))
    b_i = np.array(b.sum(0))

    sumA = np.sum(a_i * a_i)
    sumB = np.sum(b_i * b_i) + (np.sum(c) / n)
    sumAB = np.sum(d) + (np.sum(c) / n)

    precision = sumAB / max(sumB, epsilon)
    recall = sumAB / max(sumA, epsilon)

    fScore = 2.0 * precision * recall / max(precision + recall, epsilon)
    are = 1.0 - fScore

    if all_stats:
        return are, precision, recall
    else:
        return are




def find_lr(model, loader, loss_fn, optim, init_value=1e-6, final_value=1.,beta=0.98,bloding_scale=10.,num=None):
    if not num:
        num = len(loader)
    mult = (final_value / init_value) ** (1 / num)
    lr = init_value

    for params in optim.param_groups:
        params['lr'] = lr
    model.train()
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model.to(device)
    
    for i, t in tqdm(enumerate(loader), total=num):
        if len(t) == 2:
            imgs, labels = t
            weight = None
        else:
            imgs, labels, weight = t
            weight = weight.to(device)
            
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        
        batch_num += 1

        optim.zero_grad()

        outputs = model(imgs)
        
        
        if isinstance(loss_fn,list):
            loss = 0.0
            for crit in loss_fn:
                if weight is None:
                    loss += crit(outputs, labels)
                else:
                    weight = weight.float()
                    loss += crit(outputs, labels, weight)
        else:
            if weight is None:
                loss = loss_fn(outputs, labels)
            else:
                loss = loss_fn(outputs, labels, weight)
                
                
        # Compute the smoothed loss
        avg_loss = beta * avg_loss + (1 - beta) * loss.item()
        smoothed_loss = avg_loss / (1 - beta ** batch_num)
        # Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > bloding_scale * best_loss:
            print('exited with best_loss at {}'.format(best_loss))
            plt.plot(log_lrs[10:-5], losses[10:-5])
            plt.show()
            return log_lrs, losses
        # Record the best loss
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
        # Store the values
        losses.append(smoothed_loss)
        log_lrs.append(math.log10(lr))
        # Do the SGD step
        # Update the lr for the next step

        loss.backward()
        optim.step()

        lr *= mult
        for params in optim.param_groups:
            params['lr'] = lr

    plt.plot(log_lrs[10:-5], losses[10:-5])
    plt.show()
    return log_lrs, losses



def set_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are suing GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manualSeed)
        torch.cuda.manual_seed_all(manualSeed)