"""
Script to covert a .h weights file of the DTLN model to the saved model format.

Example call:
    $python convert_weights_to_saved_model.py -m /name/of/the/model.h5 \
                                              -t name_target_folder 
                              

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 24.06.2020

This code is licensed under the terms of the MIT-license.
"""

from DTLN_model import DTLN_model
import argparse


if __name__ == '__main__':
    # arguement parser for running directly from the command line
    parser = argparse.ArgumentParser(description='data evaluation')
    parser.add_argument('--weights_file', '-m',
                        help='path to .h5 weights file')
    parser.add_argument('--target_folder', '-t',
                        help='target folder for saved model')
    
    args = parser.parse_args()
    
    converter = DTLN_model()
    converter.create_saved_model(args.weights_file, args.target_folder)