import os

import h5py
import numpy as np
import torch

from datasets.hdf5 import get_test_loaders
from unet3d import utils
from unet3d.config import load_config
from unet3d.model import get_model

from pathlib import Path
from tools.loggers import call_logger

from datasets.hdf5 import HDF5Dataset

from torch.utils.data import DataLoader
#from tools.loggers import call_logger
#logger = utils.get_logger('UNet3DPredictor')
import collections
import os.path as osp
from tqdm import tqdm
def predict(model, data_loader, config):
    """
    Return prediction masks by applying the model on the given dataset

    Args:
        model (Unet3D): trained 3D UNet model used for prediction
        data_loader (torch.utils.data.DataLoader): input data loader
        out_channels (int): number of channels in the network output
        device (torch.Device): device to run the prediction on

    Returns:
         prediction_maps (numpy array): prediction masks for given dataset
    """

    def _volume_shape(dataset):
        # TODO: support multiple internal datasets
        raw = dataset.raws[0]
        if raw.ndim == 3:
            return raw.shape
        else:
            return raw.shape[1:]

    out_channels = config['model'].get('out_channels')
    if out_channels is None:
        out_channels = config['model']['dt_out_channels']

    prediction_channel = config.get('prediction_channel', None)
    if prediction_channel is not None:
        logger.info(f"Using only channel '{prediction_channel}' from the network output")

    device = config['device']
    output_heads = config['model'].get('output_heads', 1)

    logger.info(f'Running prediction on {len(data_loader)} patches...')
    # dimensionality of the the output (CxDxHxW)
    volume_shape = _volume_shape(data_loader.dataset)
    if prediction_channel is None:
        prediction_maps_shape = (out_channels,) + volume_shape
    else:
        # single channel prediction map
        prediction_maps_shape = (1,) + volume_shape

    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')

    pad_width = config['model'].get('pad_width',None) 




    # initialize the output prediction arrays
    prediction_maps = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]
    # initialize normalization mask in order to average out probabilities of overlapping patches
    normalization_masks = [np.zeros(prediction_maps_shape, dtype='float32') for _ in range(output_heads)]

    # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
    model.eval()
    # Run predictions on the entire input dataset
    with torch.no_grad():
        for tt in tqdm(data_loader):
            if len(tt)==2:
                patch,index = tt
            elif len(tt)==3:
                patch,gp,index = tt
            else:
                raise ValueError('len of loader is wrong')
            #logger.info(f'Predicting slice:{index}')

            # save patch index: (C,D,H,W)
            if prediction_channel is None:
                channel_slice = slice(0, out_channels)
            else:
                channel_slice = slice(0, 1)

            index = (channel_slice,) + tuple(index)

            # send patch to device
            patch = patch.to(device)
            
            # forward pass
            if len(tt)==2:
                predictions = model(patch)
            elif len(tt)==3:
                gp = gp.to(device)
                predictions = model(patch,gp)
            

            # wrap predictions into a list if there is only one output head from the network
            if output_heads == 1:
                predictions = [predictions]

            for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
                                                                      normalization_masks):
                # squeeze batch dimension and convert back to numpy array
                prediction = prediction.squeeze(dim=0).cpu().numpy()
                if prediction_channel is not None:
                    # use only the 'prediction_channel'
                    #logger.info(f"Using channel '{prediction_channel}'...")
                    prediction = np.expand_dims(prediction[prediction_channel], axis=0)

                # unpad in order to avoid block artifacts in the output probability maps
                u_prediction, u_index = utils.unpad(prediction, index, volume_shape,pad_width)



                # accumulate probabilities into the output prediction array
                prediction_map[u_index] += u_prediction
                # count voxel visits for normalization
                normalization_mask[u_index] += 1

    return [prediction_map / normalization_mask for prediction_map, normalization_mask in
            zip(prediction_maps, normalization_masks)]


#def predict(model, data_loader, output_file, config):
#    """
#    Return prediction masks by applying the model on the given dataset
#
#    Args:
#        model (Unet3D): trained 3D UNet model used for prediction
#        data_loader (torch.utils.data.DataLoader): input data loader
#        output_file (str): path to the output H5 file
#        config (dict): global config dict
#
#    """
#
#    def _volume_shape(dataset):
#        # TODO: support multiple internal datasets
#        raw = dataset.raws[0]
#        if raw.ndim == 3:
#            return raw.shape
#        else:
#            return raw.shape[1:]
#
#    out_channels = config['model'].get('out_channels')
#    if out_channels is None:
#        out_channels = config['model']['dt_out_channels']
#
#    prediction_channel = config.get('prediction_channel', None)
#    if prediction_channel is not None:
#        logger.info(f"Using only channel '{prediction_channel}' from the network output")
#
#    device = config['device']
#    output_heads = config['model'].get('output_heads', 1)
#
#    logger.info(f'Running prediction on {len(data_loader)} patches...')
#
#    # dimensionality of the the output (CxDxHxW)
#    volume_shape = _volume_shape(data_loader.dataset)
#    if prediction_channel is None:
#        prediction_maps_shape = (out_channels,) + volume_shape
#    else:
#        # single channel prediction map
#        prediction_maps_shape = (1,) + volume_shape
#
#    logger.info(f'The shape of the output prediction maps (CDHW): {prediction_maps_shape}')
#
#    with h5py.File(output_file, 'w') as f:
#        # allocate datasets for probability maps
#        prediction_datasets = _get_dataset_names(config, output_heads, prefix='predictions')
#        prediction_maps = [
#            f.create_dataset(dataset_name, shape=prediction_maps_shape, dtype='float32', chunks=True,
#                             compression='gzip')
#            for dataset_name in prediction_datasets]
#
#        # allocate datasets for normalization masks
#        normalization_datasets = _get_dataset_names(config, output_heads, prefix='normalization')
#        normalization_masks = [
#            f.create_dataset(dataset_name, shape=prediction_maps_shape, dtype='uint8', chunks=True,
#                             compression='gzip')
#            for dataset_name in normalization_datasets]
#
#        # Sets the module in evaluation mode explicitly, otherwise the final Softmax/Sigmoid won't be applied!
#        model.eval()
#        # Run predictions on the entire input dataset
#        with torch.no_grad():
#            for patch, index in data_loader:
#                #logger.info(f'Predicting slice:{index}')
#
#                # save patch index: (C,D,H,W)
#                if prediction_channel is None:
#                    channel_slice = slice(0, out_channels)
#                else:
#                    channel_slice = slice(0, 1)
#
#                index = (channel_slice,) + tuple(index)
#
#                # send patch to device
#                patch = patch.to(device)
#                # forward pass
#                predictions = model(patch)
#
#                # wrap predictions into a list if there is only one output head from the network
#                if output_heads == 1:
#                    predictions = [predictions]
#
#                for prediction, prediction_map, normalization_mask in zip(predictions, prediction_maps,
#                                                                          normalization_masks):
#                    # squeeze batch dimension and convert back to numpy array
#                    prediction = prediction.squeeze(dim=0).cpu().numpy()
#                    if prediction_channel is not None:
#                        # use only the 'prediction_channel'
#                        #logger.info(f"Using channel '{prediction_channel}'...")
#                        prediction = np.expand_dims(prediction[prediction_channel], axis=0)
#
#                    # unpad in order to avoid block artifacts in the output probability maps
#                    u_prediction, u_index = utils.unpad(prediction, index, volume_shape)
#                    # accumulate probabilities into the output prediction array
#                    prediction_map[u_index] += u_prediction
#                    # count voxel visits for normalization
#                    normalization_mask[u_index] += 1
#
#        # normalize the prediction_maps inside the H5
#        for prediction_map, normalization_mask, prediction_dataset, normalization_dataset in zip(prediction_maps,
#                                                                                                 normalization_masks,
#                                                                                                 prediction_datasets,
#                                                                                                 normalization_datasets):
#            # TODO: iterate block by block
#            # split the volume into 4 parts and load each into the memory separately
#            #logger.info(f'Normalizing {prediction_dataset}...')
#            z, y, x = volume_shape
#            mid_x = x // 2
#            mid_y = y // 2
#            prediction_map[:, :, 0:mid_y, 0:mid_x] /= normalization_mask[:, :, 0:mid_y, 0:mid_x]
#            prediction_map[:, :, mid_y:, 0:mid_x] /= normalization_mask[:, :, mid_y:, 0:mid_x]
#            prediction_map[:, :, 0:mid_y, mid_x:] /= normalization_mask[:, :, 0:mid_y, mid_x:]
#            prediction_map[:, :, mid_y:, mid_x:] /= normalization_mask[:, :, mid_y:, mid_x:]
#            #logger.info(f'Deleting {normalization_dataset}...')
#            del f[normalization_dataset]
#
#    return prediction_maps


def _get_output_file(dataset,out_path, suffix='_predictions'):
    #return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'
    return str(Path(out_path)/(Path(dataset.file_path).stem + suffix + '.h5'))

#def _get_dataset_names(config, number_of_datasets, prefix='predictions'):
#    dataset_names = config.get('dest_dataset_name')
#    if dataset_names is not None:
#        if isinstance(dataset_names, str):
#            return [dataset_names]
#        else:
#            return dataset_names
#    else:
#        if number_of_datasets == 1:
#            return [prefix]
#        else:
#            return [f'{prefix}{i}' for i in range(number_of_datasets)]


def _get_dataset_names(config, number_of_datasets):
    dataset_names = config.get('dest_dataset_name')
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            return [dataset_names]
        else:
            return dataset_names
    else:
        default_prefix = 'predictions'
        if number_of_datasets == 1:
            return [default_prefix]
        else:
            return [f'{default_prefix}{i}' for i in range(number_of_datasets)]








def save_predictions(prediction_maps, output_file, dataset_names):
    """
    Saving probability maps to a given output H5 file. If 'average_channels'
    is set to True average the probability_maps across the the channel axis
    (useful in case where each channel predicts semantically the same thing).

    Args:
        prediction_maps (list): list of numpy array containing prediction maps in separate channels
        output_file (string): path to the output H5 file
        dataset_names (list): list of dataset names inside H5 file where the prediction maps will be saved
    """
    assert len(prediction_maps) == len(dataset_names), 'Each prediction map has to have a corresponding dataset name'
    logger.info(f'Saving predictions to: {output_file}...')

    with h5py.File(output_file, "w") as output_h5:
        for prediction_map, dataset_name in zip(prediction_maps, dataset_names):
            #logger.info(f"Creating dataset '{dataset_name}'...")
            output_h5.create_dataset(dataset_name, data=prediction_map, compression="gzip")


#def _get_output_file(dataset,out_path, suffix='_predictions'):
#    #fn = os.path.splitext(dataset.file_path)[0]
#    return str(Path(out_path)/(Path(dataset.file_path).stem + suffix + '.h5'))
#    #return f'{os.path.splitext(dataset.file_path)[0]}{suffix}.h5'



if __name__ == '__main__':   
    # Load configuration
    config = load_config()
    
    # Load model state
    model_path = config['model_path']
    
    model_fd = Path(model_path).parent
    
    
    logger = call_logger(log_file = str(model_fd/'test_log.txt'),log_name = 'UNetPredict')
    
    # Create the model
    model = get_model(config)

    if 'output_path' in config.keys():
        out_path = config['output_path']
    else: 
        out_path = str(model_fd/'h5_pred')
    os.makedirs(out_path,exist_ok = True)
        
    
    logger.info(f'Loading model from {model_path}...')
    utils.load_checkpoint(model_path, model)
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])

    logger.info('Loading HDF5 datasets...')
    
    
    datasets_config = config['datasets']

    # get train and validation files
#    test_paths = datasets_config['test_path']
#    assert isinstance(test_paths, list)
#    # get h5 internal paths for raw and label
#    raw_internal_path = datasets_config['raw_internal_path']
#    # get train/validation patch size and stride
#    patch = tuple(datasets_config['patch'])
#    stride = tuple(datasets_config['stride'])
#    num_workers = datasets_config.get('num_workers', 1)
    
    logger.info('Loading HDF5 datasets...')
    test_loaders = get_test_loaders(config)
    l_test_loaders = (list(test_loaders))
    
    for test_loader in l_test_loaders:
        logger.info(f"Processing '{test_loader.dataset.file_path}'...")

        output_file = _get_output_file(test_loader.dataset,out_path)
        
        # run the model prediction on the entire dataset and save to the 'output_file' H5
        predictions = predict(model, test_loader, config)
        dataset_names = _get_dataset_names(config, len(predictions))
        
        
        save_predictions(predictions, output_file, dataset_names)
        
        
        
        ori_h5 = test_loader.dataset.file_path
        
        with h5py.File(ori_h5, 'r') as f:
            label = f['label'][...]
        
        from sklearn.metrics import confusion_matrix
        
        #%%
        if config['model']['final_sigmoid']:
            mask_t = (predictions[0]>=0.5).astype('int')
        else:
            mask_t = (predictions[0][1]>=0.5).astype('int')
        label_t = label
        cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
        logger.info(f'proc {ori_h5}')
        logger.info('ce cm')
        logger.info(cm)    
        recall = cm[1,1]/(cm[1,0]+cm[1,1])  
        precision = cm[1,1]/(cm[0,1]+cm[1,1])  
        fscore = 2*recall*precision/(recall+precision)
        logger.info(f'recall: {recall:.4f} precision: {precision:.4f} Fscore: {fscore :.4f}')
   

#    if len(test_paths)==1 and osp.isdir(test_paths[0]):
#        test_paths = [str(fn) for fn in sorted(list(Path(test_paths[0]).glob('*.h5')))]        
#    
#    for test_path in test_paths:
#        def my_collate(batch):
#            error_msg = "batch must contain tensors or slice; found {}"
#            if isinstance(batch[0], torch.Tensor):
#                return torch.stack(batch, 0)
#            elif isinstance(batch[0], slice):
#                return batch[0]
#            elif isinstance(batch[0], collections.Sequence):
#                transposed = zip(*batch)
#                return [my_collate(samples) for samples in transposed]
#    
#            raise TypeError((error_msg.format(type(batch[0]))))
#        # construct datasets lazily
#        dataset = HDF5Dataset(test_path, patch, stride, phase='test', raw_internal_path=raw_internal_path,
#                            transformer_config=datasets_config['transformer']) 
#
#        logger.info(f'Loading test set from: {dataset.file_path}...')
#        test_loader= DataLoader(dataset, batch_size=1, num_workers=num_workers,collate_fn=my_collate)
#
#    
#
#        logger.info(f"Processing '{test_loader.dataset.file_path}'...")
#        # run the model prediction on the entire dataset
#        predictions = predict(model, test_loader, config)
#        # save the resulting probability maps
#        output_file = _get_output_file(test_loader.dataset,out_path)
#        dataset_names = _get_dataset_names(config, len(predictions))
#        save_predictions(predictions, output_file, dataset_names)
#        
#        ori_h5 = test_loader.dataset.file_path
#
#       
#        
#        with h5py.File(ori_h5, 'r') as f:
#            label = f['label'][...]
#        
#        from sklearn.metrics import confusion_matrix
#        
#        #%%
#        
#        mask_t = (predictions[0][1]>=0.5).astype('int')
#        label_t = label
#        cm = confusion_matrix(label_t.flatten(),  mask_t.flatten()) 
#        logger.info(f'proc {ori_h5}')
#        logger.info('ce cm')
#        logger.info(cm)    
#        recall = cm[1,1]/(cm[1,0]+cm[1,1])  
#        precision = cm[1,1]/(cm[0,1]+cm[1,1])  
#        fscore = 2*recall*precision/(recall+precision)
#        logger.info(f'recall: {recall:.4f} precision: {precision:.4f} Fscore: {fscore :.4f}')





