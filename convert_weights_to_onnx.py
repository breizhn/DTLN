#!/usr/bin/env python3
"""
Script to covert a .h5 weights file of the DTLN model to ONNX.
At the moment the conversion only works with TF 2.1 and not on Mac.

Example call:
    $python convert_weights_to_ONNX.py -m /name/of/the/model.h5 \
                                              -t name_target 
                              

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 03.07.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model, InstantLayerNormalization
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Multiply, Conv1D
import tensorflow as tf
import keras2onnx


if __name__ == '__main__':
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description='data evaluation')
    parser.add_argument('--weights_file', '-m',
                        help='path to .h5 weights file')
    parser.add_argument('--target_folder', '-t',
                        help='target folder for saved model')
    
    args = parser.parse_args()
    weights_file = args.weights_file
    dtln_class = DTLN_model()
    # check for type
    if weights_file.find('_norm_') != -1:
        norm_stft = True
        num_elements_first_core = 2 + dtln_class.numLayer * 3 + 2
    else:
        norm_stft = False
        num_elements_first_core = dtln_class.numLayer * 3 + 2
    # build model    
    dtln_class.build_DTLN_model_stateful(norm_stft=norm_stft)
    # load weights
    dtln_class.model.load_weights(weights_file)
    #### Model 1 ##########################
    mag = Input(batch_shape=(1, 1, (dtln_class.blockLen//2+1)))
    states_in_1 = Input(batch_shape=(1, dtln_class.numLayer, dtln_class.numUnits, 2))
    # normalizing log magnitude stfts to get more robust against level variations
    if norm_stft:
        mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
    else:
        # behaviour like in the paper
        mag_norm = mag
    # predicting mask with separation kernel  
    mask_1, states_out_1 = dtln_class.seperation_kernel_with_states(dtln_class.numLayer, 
                                                (dtln_class.blockLen//2+1), 
                                                mag_norm, states_in_1)
    
    model_1 = Model(inputs=[mag, states_in_1], outputs=[mask_1, states_out_1])
    
    #### Model 2 ###########################
    
    estimated_frame_1 = Input(batch_shape=(1, 1, (dtln_class.blockLen)))
    states_in_2 = Input(batch_shape=(1, dtln_class.numLayer, dtln_class.numUnits, 2))
    
    # encode time domain frames to feature domain
    encoded_frames = Conv1D(dtln_class.encoder_size,1,strides=1,
                            use_bias=False)(estimated_frame_1)
    # normalize the input to the separation kernel
    encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
    # predict mask based on the normalized feature frames
    mask_2, states_out_2 = dtln_class.seperation_kernel_with_states(dtln_class.numLayer, 
                                                dtln_class.encoder_size, 
                                                encoded_frames_norm, 
                                                states_in_2)
    # multiply encoded frames with the mask
    estimated = Multiply()([encoded_frames, mask_2]) 
    # decode the frames back to time domain
    decoded_frame = Conv1D(dtln_class.blockLen, 1, padding='causal',
                           use_bias=False)(estimated)
    
    model_2 = Model(inputs=[estimated_frame_1, states_in_2], 
                    outputs=[decoded_frame, states_out_2])
    
    # set weights to submodels
    weights = dtln_class.model.get_weights()
    
    model_1.set_weights(weights[:num_elements_first_core])
    model_2.set_weights(weights[num_elements_first_core:])
    # convert first model
    onnx_model = keras2onnx.convert_keras(model_1)
    temp_model_file = args.target_folder + '_1.onnx'
    keras2onnx.save_model(onnx_model, temp_model_file)
    # convert second model
    onnx_model = keras2onnx.convert_keras(model_2)
    temp_model_file = args.target_folder + '_2.onnx'
    keras2onnx.save_model(onnx_model, temp_model_file)
    
          
    print('ONNX conversion complete!')