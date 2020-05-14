# Dual-signal Transformation LSTM Network
Implementation of the stacked dual-signal transformation LSTM network (DTLN) for real-time noise suppression.


This model was handed in to the deep noise suppression challenge ([DNS-Challenge](https://github.com/microsoft/DNS-Challenge)) of Interspeech 2020. 


This approach combines a short-time Fourier transform (STFT) and a learned analysis and synthesis basis in a stacked-network approach with less than one million parameters. The model was trained on 500h of noisy speech provided by the challenge organizers. The network is capable of real-time processing (one frame in, one frame out) and reaches competitive results.
Combining these two types of signal transformations enables the DTLN to robustly extract information from magnitude spectra and incorporate phase information from the learned feature basis. The method shows state-of-the-art performance and outperforms the DNS-Challenge baseline by 0.24 points absolute in terms of the mean opinion score (MOS).
\
\
Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

### Contents of the repository:

* DTLN_model.py: \
  This file is containing the model, data generator and the training routine.
* run_training.py \
  Script to run the training. Before you can start the training with `$ python run_training.py`you have to set the paths to you training and validation data inside the script. The training script uses a default setup.
* run_evaluation.py \
  Script to process a folder with optional subfolders containing .wav files with a trained DTLN model. With the pretrained model delivered with this repository a folder can be processed as following: \
  `$ python run_evaluation.py -i /path/to/input/folder -o /path/for/processed/files -m ./pretrained_model/model.h5` \
  The evaluation script will create the new folder with the same structure as the input folder and the files will have the same name as the input files.

### Python dependencies:

The following packages will be required for this repository:
* tensorflow (2.X)
* librosa
* wavinfo 


All additional packages (numpy, soundfile, etc.) should be installed on the fly when using conda or pip. I recommend using conda environments or [pyenv](https://github.com/pyenv/pyenv) [virtualenv](https://github.com/pyenv/pyenv-virtualenv) for the python environment. For training a GPU with at least 5 GB of memory is required and Cuda 10.1 together with at least the Nvidia driver 418. If you use conda Cuda will be installed on the fly and you just need the driver. For evaluation only the CPU version of Tensorflow is enough. Everything was tested on Ubuntu 18.04.


### Training data preparation (Files will be uploaded later):

The training data used for this model can be downloaded from the DNS-Challenge [repository](https://github.com/microsoft/DNS-Challenge). Before cloning the repository make sure 'git-lfs' is installed. Copy the files from the folder DataPrep in this repository to the downloaded repo. The files containing the correct data configuration for this training setup and the naming convention in the creation script is changed to name noisy, speech and noise files the same. After this, run the script split_corpus.py to split the data in training and validation set. An 80:20 split is applied. 
  
