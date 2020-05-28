# Dual-signal Transformation LSTM Network

Implementation of the stacked dual-signal transformation LSTM network (DTLN) for real-time noise suppression.


This model was handed in to the deep noise suppression challenge ([DNS-Challenge](https://github.com/microsoft/DNS-Challenge)) of Interspeech 2020. 


This approach combines a short-time Fourier transform (STFT) and a learned analysis and synthesis basis in a stacked-network approach with less than one million parameters. The model was trained on 500h of noisy speech provided by the challenge organizers. The network is capable of real-time processing (one frame in, one frame out) and reaches competitive results.
Combining these two types of signal transformations enables the DTLN to robustly extract information from magnitude spectra and incorporate phase information from the learned feature basis. The method shows state-of-the-art performance and outperforms the DNS-Challenge baseline by 0.24 points absolute in terms of the mean opinion score (MOS).

For more information see the [paper](http://arxiv.org/abs/2005.07551).
\
\
Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

---
### Contents of the repository:

*  **DTLN_model.py** \
  This file is containing the model, data generator and the training routine.
*  **run_training.py** \
  Script to run the training. Before you can start the training with `$ python run_training.py`you have to set the paths to you training and validation data inside the script. The training script uses a default setup.
* **run_evaluation.py** \
  Script to process a folder with optional subfolders containing .wav files with a trained DTLN model. With the pretrained model delivered with this repository a folder can be processed as following: \
  `$ python run_evaluation.py -i /path/to/input -o /path/for/processed -m ./pretrained_model/model.h5` \
  The evaluation script will create the new folder with the same structure as the input folder and the files will have the same name as the input files.
* **measure_execution_time.py** \
  Script for measuring the execution time with the saved DTLN model in `./dtln_saved_model/`. For further information see this [section](#measuring-the-execution-time-of-the-dtln-model-with-the-savedmodel-format).
*  **pretrained_model/model.h5** \
  The model weights as used in the DNS-Challenge DTLN model.

---
### Python dependencies:

The following packages will be required for this repository:
* tensorflow (2.x)
* librosa
* wavinfo 


All additional packages (numpy, soundfile, etc.) should be installed on the fly when using conda or pip. I recommend using conda environments or [pyenv](https://github.com/pyenv/pyenv) [virtualenv](https://github.com/pyenv/pyenv-virtualenv) for the python environment. For training a GPU with at least 5 GB of memory is required. I recommend at least Tensorflow 2.1 with Nvidia driver 418 and Cuda 10.1. If you use conda Cuda will be installed on the fly and you just need the driver. For evaluation-only the CPU version of Tensorflow is enough. Everything was tested on Ubuntu 18.04.

---
### Training data preparation:

1. Clone the forked DNS-Challenge [repository](https://github.com/breizhn/DNS-Challenge). Before cloning the repository make sure `git-lfs` is installed. Also make sure your disk has enough space. I recommend downloading the data to an SSD for faster dataset creation.

2. Run `noisyspeech_synthesizer_multiprocessing.py` to create the dataset. `noisyspeech_synthesizer.cfg`was changed according to my training setup used for the DNS-Challenge. 

3. Run `split_dns_corpus.py`to divide the dataset in training and validation data. The classic 80:20 split is applied. This file was added to the forked repository by me.

---
### Run a training of the DTLN model:

1. Make sure all dependencies are installed in your python environment.

2. Change the paths to your training and validation dataset in `run_training.py`.

3. Run `$ python run_training.py`. 

One epoch takes around 21 minutes on a Nvidia RTX 2080 Ti when loading the training data from an SSD. 

---
### Measuring the execution time of the DTLN model with the SavedModel format:

In total there are three ways to measure the execution time for one block of the model: Running a sequence in Keras and dividing by the number of blocks in the sequence, building a stateful model in Keras and running block by block, and saving the stateful model in Tensorflow's SavedModel format and calling that one block by block. In the following I will explain how running the model in the SavedModel format, because it is the most portable version and can also be called from Tensorflow Serving.

A Keras model can be saved to the saved model format:
```python
import tensorflow as tf
'''
Building some model here
'''
tf.saved_model.save(your_keras_model, 'name_save_path')
```
Important here for real time block by block processing is, to make the LSTM layer stateful, so they can remember the states from the previous block.

The model can be imported with 
```python
model = tf.saved_model.load('name_save_path’)
```

For inference we now first call this for mapping signature names to functions
```python
infer = model.signatures[‘serving_default’]
```

and now for inferring the block `x` call
```python
y = infer(tf.constant(x))['conv1d_1']
```
This command gives you the result on the node `'conv1d_1'`which is our output node for real time processing. For more information on using the SavedModel format and obtaining the output node see this [Guide](https://www.tensorflow.org/guide/saved_model).

For making everything easier this repository provides a stateful DTLN SavedModel. 
For measuring the execution time call:
```
$ python measure_execution_time.py
```
When measuring with TF 2.2 following mean execution times were obtained for one 32 ms block:
System | Processor | #Cores | Execution Time
--- | --- | --- | ---
Ubuntu 18.04         | Intel I5 6600k @ 3.5 GHz | 4 | 0.65 ms
Macbook Air mid 2012 | Intel I7 3667U @ 2.0 GHz | 2 | 1.4 ms 
Raspberry Pi 3 B+    | ARM Cortex A53 @ 1.4 GHz | 4 | 15.54 ms


---
### Citing:


```
@article{westhausen2020dual,
  title={Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression},
  author={Westhausen, Nils L and Meyer, Bernd T},
  journal={arXiv preprint arXiv:2005.07551},
  year={2020}
}
