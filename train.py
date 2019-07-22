import importlib

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from datasets.hdf5 import get_train_loaders
from unet3d.config import load_config
from unet3d.losses import get_loss_criterion
from unet3d.metrics import get_evaluation_metric
from unet3d.model import get_model
from unet3d.trainer import UNet3DTrainer
#from unet3d.utils import get_logger
from tools.loggers import call_logger
from unet3d.utils import get_number_of_learnable_parameters,set_seed
from pathlib import Path
import os 
 

def _create_trainer(config, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, logger):
    assert 'trainer' in config, 'Could not find trainer configuration'
    trainer_config = config['trainer']

    resume = trainer_config.get('resume', None)
    pre_trained = trainer_config.get('pre_trained', None)

    if resume is not None:
        # continue training from a given checkpoint
        return UNet3DTrainer.from_checkpoint(resume, model,
                                             optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, loaders,
                                             logger=logger)
    elif pre_trained is not None:
        # fine-tune a given pre-trained model
        return UNet3DTrainer.from_pretrained(pre_trained, model, optimizer, lr_scheduler, loss_criterion,
                                             eval_criterion, device=config['device'], loaders=loaders,
                                             max_num_epochs=trainer_config['epochs'],
                                             max_num_iterations=trainer_config['iters'],
                                             validate_after_iters=trainer_config['validate_after_iters'],
                                             log_after_iters=trainer_config['log_after_iters'],
                                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                                             logger=logger)
    else:
        # start training from scratch
        return UNet3DTrainer(model, optimizer, lr_scheduler, loss_criterion, eval_criterion,
                             config['device'], loaders, trainer_config['checkpoint_dir'],
                             max_num_epochs=trainer_config['epochs'],
                             max_num_iterations=trainer_config['iters'],
                             validate_after_iters=trainer_config['validate_after_iters'],
                             log_after_iters=trainer_config['log_after_iters'],
                             eval_score_higher_is_better=trainer_config['eval_score_higher_is_better'],
                             logger=logger)


def _create_optimizer(config, model):
    assert 'optimizer' in config, 'Cannot find optimizer configuration'
    optimizer_config = config['optimizer']
    if 'name' not in optimizer_config.keys():
        optimizer_config['name'] = 'Adam'
    
    if optimizer_config['name'] == 'Adam':
        
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_config['name'] == 'SGD':
        learning_rate = optimizer_config['learning_rate']
        weight_decay = optimizer_config['weight_decay']
        momentum = optimizer_config['momentum']
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    else:
        raise ValueError('unknown optimizer')
    return optimizer


def _create_lr_scheduler(config, optimizer):
    lr_config = config.get('lr_scheduler', None)
    if lr_config is None:
        # use ReduceLROnPlateau as a default scheduler
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20, verbose=True)
    else:
        class_name = lr_config.pop('name')
        m = importlib.import_module('torch.optim.lr_scheduler')
        clazz = getattr(m, class_name)
        # add optimizer to the config
        lr_config['optimizer'] = optimizer
        
        for key,value in lr_config.items():
            if isinstance(value, str):
                try:
                    lr_config[key] = eval(value)    
                except:
                    pass
                finally:
                    pass
                
        
        
        return clazz(**lr_config)


if __name__ == '__main__':
    # Create main logger
    #logger = get_logger('UNet3DTrainer')
    #Load and log experiment configuration
    config = load_config()
    
    chkpt_sav = config['trainer']['checkpoint_dir']
    path_log  = Path('../checkpoints')/chkpt_sav
    config['trainer']['checkpoint_dir'] = str(path_log)
    os.makedirs(path_log,exist_ok = True)
    
    
    
    #set seed
    manual_seed = config.get('manual_seed',0)
    set_seed(manual_seed)
    
    
    if config['findlr']:
        logger = call_logger(log_file =None,log_name = 'UNet3DTrainer')
    else:
        logger = call_logger(log_file = str((Path('../checkpoints')/chkpt_sav)/'log.txt'),log_name = 'UNet3DTrainer')
    

    logger.info(config)

    manual_seed = config.get('manual_seed', None)
    if manual_seed is not None:
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create the model
    model = get_model(config)
    # put the model on GPUs
    logger.info(f"Sending the model to '{config['device']}'")
    model = model.to(config['device'])
    # Log the number of learnable parameters
    logger.info(f'Number of learnable params {get_number_of_learnable_parameters(model)}')

    # Create loss criterion
    loss_criterion = get_loss_criterion(config)
    # Create evaluation metric
    eval_criterion = get_evaluation_metric(config)

    # Create data loaders
    loaders = get_train_loaders(config)
    config['n_iter_loader'] = len(loaders['train'])

    # Create the optimizer
    optimizer = _create_optimizer(config, model)

    # Create learning rate adjustment strategy
    lr_scheduler = _create_lr_scheduler(config, optimizer)
    
    
    if config['findlr']:
        
        from unet3d.utils import find_lr
        log_lrs, losses = find_lr(model, loaders['train'], loss_criterion, optimizer, init_value=1e-6, final_value=10.,beta=0.75,bloding_scale=5.,num = 200)
    else:
        # Create model trainer
        trainer = _create_trainer(config, model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
                                  loss_criterion=loss_criterion, eval_criterion=eval_criterion, loaders=loaders,
                                  logger=logger)
        # Start training
        trainer.fit()




