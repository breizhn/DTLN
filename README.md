# Dual-signal Transformation LSTM Network

+ Tensorflow 2.x implementation of the stacked dual-signal transformation LSTM network (DTLN) for real-time noise suppression.
+ This repository provides the code for training, infering and serving the DTLN model in python. It also provides pretrained models in SavedModel, TF-lite and ONNX format, which can be used as baseline for your own projects. The model is able to run with real time audio on a RaspberryPi.
+ If you are doing cool things with this repo, tell me about it. I am always curious about what you are doing with this code or this models.

---

The DTLN model was handed in to the deep noise suppression challenge ([DNS-Challenge](https://github.com/microsoft/DNS-Challenge)) and the paper was presented at Interspeech 2020. 


This approach combines a short-time Fourier transform (STFT) and a learned analysis and synthesis basis in a stacked-network approach with less than one million parameters. The model was trained on 500h of noisy speech provided by the challenge organizers. The network is capable of real-time processing (one frame in, one frame out) and reaches competitive results.
Combining these two types of signal transformations enables the DTLN to robustly extract information from magnitude spectra and incorporate phase information from the learned feature basis. The method shows state-of-the-art performance and outperforms the DNS-Challenge baseline by 0.24 points absolute in terms of the mean opinion score (MOS).

For more information see the [paper](https://www.isca-speech.org/archive/Interspeech_2020/pdfs/2631.pdf). The results of the DNS-Challenge are published [here](https://www.microsoft.com/en-us/research/dns-challenge/interspeech2020/finalresults). We reached a competitive 8th place out of 17 teams in the real time track.

---

For baseline usage and to reproduce the processing used for the paper run:
```bash
$ python run_evaluation.py -i in/folder/with/wav -o target/folder/processed/files -m ./pretrained_model/model.h5
```

---

The pretrained DTLN-aec (the DTLN applied to acoustic echo cancellation) can be found in the [DTLN-aec repository](https://github.com/breizhn/DTLN-aec).

---

Author: Nils L. Westhausen ([Communication Acoustics](https://uol.de/en/kommunikationsakustik) , Carl von Ossietzky University, Oldenburg, Germany)

This code is licensed under the terms of the MIT license.


---
### Citing:

If you are using the DTLN model, please cite:

```BibTex
@inproceedings{Westhausen2020,
  author={Nils L. Westhausen and Bernd T. Meyer},
  title={{Dual-Signal Transformation LSTM Network for Real-Time Noise Suppression}},
  year=2020,
  booktitle={Proc. Interspeech 2020},
  pages={2477--2481},
  doi={10.21437/Interspeech.2020-2631},
  url={http://dx.doi.org/10.21437/Interspeech.2020-2631}
}
```


---
### Contents of the README:

* [Results](#results)
* [Execution Times](#execution-times)
* [Audio Samples](#audio-samples)
* [Contents of the repository](#contents-of-the-repository)
* [Python dependencies](#python-dependencies)
* [Training data preparation](#training-data-preparation)
* [Run a training of the DTLN model](#run-a-training-of-the-dtln-model)
* [Measuring the execution time of the DTLN model with the SavedModel format](#measuring-the-execution-time-of-the-dtln-model-with-the-savedmodel-format)
* [Real time processing with the SavedModel format](#real-time-processing-with-the-savedmodel-format)
* [Real time processing with tf-lite](#real-time-processing-with-tf-lite)
* [Real time audio with sounddevice and tf-lite](#real-time-audio-with-sounddevice-and-tf-lite)
* [Model conversion and real time processing with ONNX](#model-conversion-and-real-time-processing-with-onnx)



---
### Results:

Results on the DNS-Challenge non reverberant test set:
Model | PESQ [mos] | STOI [%] | SI-SDR [dB] | TF version
--- | --- | --- | --- | ---
unprocessed | 2.45 | 91.52 | 9.07 |
NsNet (Baseline) | 2.70 | 90.56 | 12.57 |
 |  |  |  | 
DTLN (500h) | 3.04 | 94.76 | 16.34 | 2.1
DTLN (500h)| 2.98 | 94.75 | 16.20 | TF-light
DTLN (500h) | 2.95 | 94.47 | 15.71 | TF-light quantized
 |  |  |  | 
DTLN norm (500h) | 3.04 | 94.47 | 16.10 | 2.2
 |  |  |  | 
DTLN norm (40h) | 3.05 | 94.57 | 16.88 | 2.2
DTLN norm (40h) | 2.98 | 94.56 | 16.58 | TF-light
DTLN norm (40h) | 2.98 | 94.51 | 16.22 | TF-light quantized

* The conversion to TF-light slightly reduces the performance. 
* The dynamic range quantization of TF-light also reduces the performance a bit and introduces some quantization noise. But the audio-quality is still on a high level and the model is real-time capable on the Raspberry Pi 3 B+.
* The normalization of the log magnitude of the STFT does not decrease the model performance and makes it more robust against level variations.
* With data augmentation during training it is possible to train the DTLN model on just 40h of noise and speech data. If you have any question regarding this, just contact me.

[To contents](#contents-of-the-readme)

---

### Execution Times:

Execution times for SavedModel are measured with TF 2.2 and for TF-lite with the TF-lite runtime:
System | Processor | #Cores | SavedModel | TF-lite | TF-lite quantized
--- | --- | --- | --- | --- | ---
Ubuntu 18.04         | Intel I5 6600k @ 3.5 GHz | 4 | 0.65 ms | 0.36 ms | 0.27 ms
Macbook Air mid 2012 | Intel I7 3667U @ 2.0 GHz | 2 | 1.4 ms | 0.6 ms | 0.4 ms
Raspberry Pi 3 B+    | ARM Cortex A53 @ 1.4 GHz | 4 | 15.54 ms | 9.6 ms | 2.2 ms

For real-time capability the execution time must be below 8 ms.

[To contents](#contents-of-the-readme)

---

### Audio Samples:

Here some audio samples created with the tf-lite model. Sadly audio can not be integrated directly into markdown.

Noisy | Enhanced | Noise type
--- | --- | --- 
[Sample 1](https://cloudsync.uol.de/s/GFHzmWWJAwgQPLf) | [Sample 1](https://cloudsync.uol.de/s/p3M48y7cjkJ2ZZg) | Air conditioning
[Sample 2](https://cloudsync.uol.de/s/4Y2PoSpJf7nXx9T) | [Sample 2](https://cloudsync.uol.de/s/QeK4aH5KCELPnko) | Music
[Sample 3](https://cloudsync.uol.de/s/Awc6oBtnTpb5pY7) | [Sample 3](https://cloudsync.uol.de/s/yNsmDgxH3MPWMTi) | Bus 


[To contents](#contents-of-the-readme)

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
  Script for measuring the execution time with the saved DTLN model in `./pretrained_model/dtln_saved_model/`. For further information see this [section](#measuring-the-execution-time-of-the-dtln-model-with-the-savedmodel-format).
* **real_time_processing.py** \
  Script, which explains how real time processing with the SavedModel works. For more information see this [section](#real-time-processing-with-the-savedmodel-format).
+  **./pretrained_model/** \
   * `model.h5`: Model weights as used in the DNS-Challenge DTLN model.
   * `DTLN_norm_500h.h5`: Model weights trained on 500h with normalization of stft log magnitudes.
   * `DTLN_norm_40h.h5`: Model weights trained on 40h with normalization of stft log magnitudes.
   * `./dtln_saved_model`: same as `model.h5` but as a stateful model in SavedModel format.
   * `./DTLN_norm_500h_saved_model`: same as `DTLN_norm_500h.h5` but as a stateful model in SavedModel format.
   * `./DTLN_norm_40h_saved_model`: same as `DTLN_norm_40h.h5` but as a stateful model in SavedModel format.
   * `model_1.tflite` together with `model_2.tflite`: same as `model.h5` but as TF-lite model with external state handling.
   * `model_quant_1.tflite` together with `model_quant_2.tflite`: same as `model.h5` but as TF-lite model with external state handling and dynamic range quantization.
   * `model_1.onnx` together with `model_2.onnx`: same as `model.h5` but as ONNX model with external state handling.
   
[To contents](#contents-of-the-readme)
   
---
### Python dependencies:

The following packages will be required for this repository:
* TensorFlow (2.x)
* librosa
* wavinfo 


All additional packages (numpy, soundfile, etc.) should be installed on the fly when using conda or pip. I recommend using conda environments or [pyenv](https://github.com/pyenv/pyenv) [virtualenv](https://github.com/pyenv/pyenv-virtualenv) for the python environment. For training a GPU with at least 5 GB of memory is required. I recommend at least Tensorflow 2.1 with Nvidia driver 418 and Cuda 10.1. If you use conda Cuda will be installed on the fly and you just need the driver. For evaluation-only the CPU version of Tensorflow is enough. Everything was tested on Ubuntu 18.04.

Conda environments for training (with cuda) and for evaluation (CPU only) can be created as following:

For the training environment:
```shell
$ conda env create -f train_env.yml
```
For the evaluation environment:
```
$ conda env create -f eval_env.yml
```
For the tf-lite environment:
```
$ conda env create -f tflite_env.yml
```
The tf-lite runtime must be downloaded from [here](https://www.tensorflow.org/lite/guide/python).

[To contents](#contents-of-the-readme)

---
### Training data preparation:

1. Clone the forked DNS-Challenge [repository](https://github.com/breizhn/DNS-Challenge). Before cloning the repository make sure `git-lfs` is installed. Also make sure your disk has enough space. I recommend downloading the data to an SSD for faster dataset creation.

2. Run `noisyspeech_synthesizer_multiprocessing.py` to create the dataset. `noisyspeech_synthesizer.cfg`was changed according to my training setup used for the DNS-Challenge. 

3. Run `split_dns_corpus.py`to divide the dataset in training and validation data. The classic 80:20 split is applied. This file was added to the forked repository by me.

[To contents](#contents-of-the-readme)

---
### Run a training of the DTLN model:

1. Make sure all dependencies are installed in your python environment.

2. Change the paths to your training and validation dataset in `run_training.py`.

3. Run `$ python run_training.py`. 

One epoch takes around 21 minutes on a Nvidia RTX 2080 Ti when loading the training data from an SSD. 

[To contents](#contents-of-the-readme)

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
model = tf.saved_model.load('name_save_path')
```

For inference we now first call this for mapping signature names to functions
```python
infer = model.signatures['serving_default']
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

[To contents](#contents-of-the-readme)

---

### Real time processing with the SavedModel format:

For explanation look at `real_time_processing.py`. 

Here some consideration for integrating this model in your project:
* The sampling rate of this model is fixed at 16 kHz. It will not work smoothly with other sampling rates.
* The block length of 32 ms and the block shift of 8 ms are also fixed. For changing these values, the model must be retrained.
* The delay created by the model is the block length, so the input-output delay is 32 ms.
* For real time capability on your system, the execution time must be below the length of the block shift, so below 8 ms. 
* If can not give you support on the hardware side, regarding soundcards, drivers and so on. Be aware, a lot of artifacts can come from this side.

[To contents](#contents-of-the-readme)

---
### Real time processing with tf-lite:

With TF 2.3 it is finally possible to convert LSTMs to tf-lite. It is still not perfect because the states must be handled seperatly for a stateful model and tf-light does not support complex numbers. That means that the model is splitted in two submodels when converting it to tf-lite and the calculation of the FFT and iFFT is performed outside the model. I provided an example script for explaining, how real time processing with the tf light model works (```real_time_processing_tf_lite.py```). In this script the tf-lite runtime is used. The runtime can be downloaded [here](https://www.tensorflow.org/lite/guide/python). Quantization works now.

Using the tf-lite DTLN model and the tf-lite runtime the execution time on an old Macbook Air mid 2012 can be decreased to **0.6 ms**.

[To contents](#contents-of-the-readme)

---
### Real time audio with sounddevice and tf-lite:

The file ```real_time_dtln_audio.py```is an example how real time audio with the tf-lite model and the [sounddevice](https://github.com/spatialaudio/python-sounddevice) toolbox can be implemented. The script is based on the ```wire.py``` example. It works fine on an old Macbook Air mid 2012 and so it will probably run on most newer devices. In the quantized version it was sucessfully tested on an Raspberry Pi 3B +.

First check for your audio devices:
```
$ python real_time_dtln_audio.py --list-devices
```
Choose the index of an input and an output device and call:
```
$ python real_time_dtln_audio.py -i in_device_idx -o out_device_idx
```
If the script is showing too much ```input underflow``` restart the sript. If that does not help, increase the latency with the ```--latency``` option. The default value is 0.2 .

[To contents](#contents-of-the-readme)

---
### Model conversion and real time processing with ONNX:

Finally I got the ONNX model working. 
For converting the model TF 2.1 and keras2onnx is required. keras2onnx can be downloaded [here](https://github.com/onnx/keras-onnx) and must be installed from source as described in the README. When all dependencies are installed, call:
```
$ python convert_weights_to_onnx.py -m /name/of/the/model.h5 -t onnx_model_name
```
to convert the model to the ONNX format. The model is split in two parts as for the TF-lite model. The conversion does not work on MacOS.
The real time processing works similar to the TF-lite model and can be looked up in following file: ```real_time_processing_onnx.py ```
The ONNX runtime required for this script can be installed with:
```
$ pip install onnxruntime
```
The execution time on the Macbook Air mid 2012 is around 1.13 ms for one block.
