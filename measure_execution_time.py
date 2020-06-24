#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script tests the execution time of the DTLN model on a CPU.
Please use TF 2.2 for comparability.

Just run "python measure_execution_time.py"

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 13.05.2020

This code is licensed under the terms of the MIT-license.
"""

import time
import tensorflow as tf 
import numpy as np
import os

# only use the cpu
os.environ["CUDA_VISIBLE_DEVICES"]=''

if __name__ == '__main__':
    # loading model in saved model format
    model = tf.saved_model.load('./pretrained_model/dtln_saved_model')
    # mapping signature names to functions
    infer = model.signatures["serving_default"]
        
    exec_time = []
    # create random input for testing
    x = np.random.randn(1,512).astype('float32')
    for idx in range(1010):
        # run timer
        start_time = time.time()
        # infer one block
        y = infer(tf.constant(x))['conv1d_1']
        exec_time.append((time.time() - start_time))
    # ignore the first ten iterations
    print('Execution time per block: ' + 
          str( np.round(np.mean(np.stack(exec_time[10:]))*1000, 2)) + ' ms')

# Ubuntu 18.04          I5 6600k        @ 3.5 GHz:  0.65 ms (4 cores)
# Macbook Air mid 2012 	I7 3667U        @ 2.0 GHz:  1.4 ms  (2 cores)
# Raspberry Pi 3 B+     ARM Cortex A53  @ 1.4 GHz: 15.54    (4 cores)
