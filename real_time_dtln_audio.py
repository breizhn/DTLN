#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a real time example how to implement DTLN tf light model with 
sounddevice. The script is based on the "wire.py" example of the sounddevice 
toolbox. If the command line shows "input underflow", restart the script.
If there are still a lot of dropouts, increase the latency.

First call:
    
    $ python real_time_dtln_audio.py --list-devices

to get your audio devices. In the next step call

    $ python real_time_dtln_audio.py -i in_device_idx -o out_device_idx

For .whl files of the tf light runtime go to: 
    https://www.tensorflow.org/lite/guide/python
    
    

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 01.07.2020

This code is licensed under the terms of the MIT-license.
"""


import numpy as np
import sounddevice as sd
import tflite_runtime.interpreter as tflite
import argparse


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    '-i', '--input-device', type=int_or_str,
    help='input device (numeric ID or substring)')
parser.add_argument(
    '-o', '--output-device', type=int_or_str,
    help='output device (numeric ID or substring)')

parser.add_argument('--latency', type=float, help='latency in seconds', default=0.2)
args = parser.parse_args(remaining)

# set some parameters
block_len_ms = 32 
block_shift_ms = 8
fs_target = 16000
# create the interpreters
interpreter_1 = tflite.Interpreter(model_path='./pretrained_model/model_1.tflite')
interpreter_1.allocate_tensors()
interpreter_2 = tflite.Interpreter(model_path='./pretrained_model/model_2.tflite')
interpreter_2.allocate_tensors()
# Get input and output tensors.
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
# create states for the lstms
states_1 = np.zeros(input_details_1[1]['shape']).astype('float32')
states_2 = np.zeros(input_details_2[1]['shape']).astype('float32')
# calculate shift and length
block_shift = int(np.round(fs_target * (block_shift_ms / 1000)))
block_len = int(np.round(fs_target * (block_len_ms / 1000)))
# create buffer
in_buffer = np.zeros((block_len)).astype('float32')
out_buffer = np.zeros((block_len)).astype('float32')


def callback(indata, outdata, frames, time, status):
    # buffer and states to global
    global in_buffer, out_buffer, states_1, states_2
    if status:
        print(status)
    # write to buffer
    in_buffer[:-block_shift] = in_buffer[block_shift:]
    in_buffer[-block_shift:] = np.squeeze(indata)
    # calculate fft of input block
    in_block_fft = np.fft.rfft(in_buffer)
    in_mag = np.abs(in_block_fft)
    in_phase = np.angle(in_block_fft)
    # reshape magnitude to input dimensions
    in_mag = np.reshape(in_mag, (1,1,-1)).astype('float32')
    # set tensors to the first model
    interpreter_1.set_tensor(input_details_1[1]['index'], states_1)
    interpreter_1.set_tensor(input_details_1[0]['index'], in_mag)
    # run calculation 
    interpreter_1.invoke()
    # get the output of the first block
    out_mask = interpreter_1.get_tensor(output_details_1[0]['index']) 
    states_1 = interpreter_1.get_tensor(output_details_1[1]['index'])   
    # calculate the ifft
    estimated_complex = in_mag * out_mask * np.exp(1j * in_phase)
    estimated_block = np.fft.irfft(estimated_complex)
    # reshape the time domain block
    estimated_block = np.reshape(estimated_block, (1,1,-1)).astype('float32')
    # set tensors to the second block
    interpreter_2.set_tensor(input_details_2[1]['index'], states_2)
    interpreter_2.set_tensor(input_details_2[0]['index'], estimated_block)
    # run calculation
    interpreter_2.invoke()
    # get output tensors
    out_block = interpreter_2.get_tensor(output_details_2[0]['index']) 
    states_2 = interpreter_2.get_tensor(output_details_2[1]['index']) 
    # write to buffer
    out_buffer[:-block_shift] = out_buffer[block_shift:]
    out_buffer[-block_shift:] = np.zeros((block_shift))
    out_buffer  += np.squeeze(out_block)
    # output to soundcard
    outdata[:] = np.expand_dims(out_buffer[:block_shift], axis=-1)
    


try:
    with sd.Stream(device=(args.input_device, args.output_device),
                   samplerate=fs_target, blocksize=block_shift,
                   dtype=np.float32, latency=args.latency,
                   channels=1, callback=callback):
        print('#' * 80)
        print('press Return to quit')
        print('#' * 80)
        input()
except KeyboardInterrupt:
    parser.exit('')
except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
    
