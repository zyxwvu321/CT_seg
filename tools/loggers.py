#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 14:22:48 2019

@author: lab
"""



import logging

def call_logger(log_file = None, log_name = 'my_logger'):
    
   
    
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)
    
    
    # set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # create a handler, write to log.txt
    # logging.FileHandler(self, filename, mode='a', encoding=None, delay=0)
    # A handler class which writes formatted logging records to disk files.
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
    # create another handler, for stdout in terminal
    # A handler class which writes logging records to a stream
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    
    # add handler to logger
    if log_file is not None:
        logger.addHandler(fh)
        
    logger.addHandler(sh)
    
    # log it
    logger.debug('Debug')
    logger.info('Info')
    return logger
