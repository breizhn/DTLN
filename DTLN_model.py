# -*- coding: utf-8 -*-
"""
This File contains everything to train the DTLN model.

For running the training see "run_training.py".
To run evaluation with the provided pretrained model see "run_evaluation.py".

Author: Nils L. Westhausen (nils.westhausen@uol.de)
Version: 24.06.2020

This code is licensed under the terms of the MIT-license.
"""


import os, fnmatch
import tensorflow.keras as keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Dense, LSTM, Dropout, \
    Lambda, Input, Multiply, Layer, Conv1D
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger, \
    EarlyStopping, ModelCheckpoint
import tensorflow as tf
import soundfile as sf
from wavinfo import WavInfoReader
from random import shuffle, seed
import numpy as np



class audio_generator():
    '''
    Class to create a Tensorflow dataset based on an iterator from a large scale 
    audio dataset. This audio generator only supports single channel audio files.
    '''
    
    def __init__(self, path_to_input, path_to_s1, len_of_samples, fs, train_flag=False):
        '''
        Constructor of the audio generator class.
        Inputs:
            path_to_input       path to the mixtures
            path_to_s1          path to the target source data
            len_of_samples      length of audio snippets in samples
            fs                  sampling rate
            train_flag          flag for activate shuffling of files
        '''
        # set inputs to properties
        self.path_to_input = path_to_input
        self.path_to_s1 = path_to_s1
        self.len_of_samples = len_of_samples
        self.fs = fs
        self.train_flag=train_flag
        # count the number of samples in your data set (depending on your disk,
        #                                               this can take some time)
        self.count_samples()
        # create iterable tf.data.Dataset object
        self.create_tf_data_obj()
        
    def count_samples(self):
        '''
        Method to list the data of the dataset and count the number of samples. 
        '''

        # list .wav files in directory
        self.file_names = fnmatch.filter(os.listdir(self.path_to_input), '*.wav')
        # count the number of samples contained in the dataset
        self.total_samples = 0
        for file in self.file_names:
            info = WavInfoReader(os.path.join(self.path_to_input, file))
            self.total_samples = self.total_samples + \
                int(np.fix(info.data.frame_count/self.len_of_samples))
    
         
    def create_generator(self):
        '''
        Method to create the iterator. 
        '''

        # check if training or validation
        if self.train_flag:
            shuffle(self.file_names)
        # iterate over the files  
        for file in self.file_names:
            # read the audio files
            noisy, fs_1 = sf.read(os.path.join(self.path_to_input, file))
            speech, fs_2 = sf.read(os.path.join(self.path_to_s1, file))
            # check if the sampling rates are matching the specifications
            if fs_1 != self.fs or fs_2 != self.fs:
                raise ValueError('Sampling rates do not match.')
            if noisy.ndim != 1 or speech.ndim != 1:
                raise ValueError('Too many audio channels. The DTLN audio_generator \
                                 only supports single channel audio data.')
            # count the number of samples in one file
            num_samples = int(np.fix(noisy.shape[0]/self.len_of_samples))
            # iterate over the number of samples
            for idx in range(num_samples):
                # cut the audio files in chunks
                in_dat = noisy[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                tar_dat = speech[int(idx*self.len_of_samples):int((idx+1)*
                                                        self.len_of_samples)]
                # yield the chunks as float32 data
                yield in_dat.astype('float32'), tar_dat.astype('float32')
              

    def create_tf_data_obj(self):
        '''
        Method to to create the tf.data.Dataset. 
        '''

        # creating the tf.data.Dataset from the iterator
        self.tf_data_set = tf.data.Dataset.from_generator(
                        self.create_generator,
                        (tf.float32, tf.float32),
                        output_shapes=(tf.TensorShape([self.len_of_samples]), \
                                       tf.TensorShape([self.len_of_samples])),
                        args=None
                        )

        
                


class DTLN_model():
    '''
    Class to create and train the DTLN model
    '''
    
    def __init__(self):
        '''
        Constructor
        '''

        # defining default cost function
        self.cost_function = self.snr_cost
        # empty property for the model
        self.model = []
        # defining default parameters
        self.fs = 16000
        self.batchsize = 32
        self.len_samples = 15
        self.activation = 'sigmoid'
        self.numUnits = 128
        self.numLayer = 2
        self.blockLen = 512
        self.block_shift = 128
        self.dropout = 0.25
        self.lr = 1e-3
        self.max_epochs = 200
        self.encoder_size = 256
        self.eps = 1e-7
        # reset all seeds to 42 to reduce invariance between training runs
        os.environ['PYTHONHASHSEED']=str(42)
        seed(42)
        np.random.seed(42)
        tf.random.set_seed(42)
        # some line to correctly find some libraries in TF 2.x
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, enable=True)
        

    @staticmethod
    def snr_cost(s_estimate, s_true):
        '''
        Static Method defining the cost function. 
        The negative signal to noise ratio is calculated here. The loss is 
        always calculated over the last dimension. 
        '''

        # calculating the SNR
        snr = tf.reduce_mean(tf.math.square(s_true), axis=-1, keepdims=True) / \
            (tf.reduce_mean(tf.math.square(s_true-s_estimate), axis=-1, keepdims=True)+1e-7)
        # using some more lines, because TF has no log10
        num = tf.math.log(snr) 
        denom = tf.math.log(tf.constant(10, dtype=num.dtype))
        loss = -10*(num / (denom))
        # returning the loss
        return loss
        

    def lossWrapper(self):
        '''
        A wrapper function which returns the loss function. This is done to
        to enable additional arguments to the loss function if necessary.
        '''
        def lossFunction(y_true,y_pred):
            # calculating loss and squeezing single dimensions away
            loss = tf.squeeze(self.cost_function(y_pred,y_true))
            # calculate mean over batches
            loss = tf.reduce_mean(loss)
            # return the loss
            return loss
        # returning the loss function as handle
        return lossFunction
    
    

    '''
    In the following some helper layers are defined.
    '''  
    
    def stftLayer(self, x):
        '''
        Method for an STFT helper layer used with a Lambda layer. The layer
        calculates the STFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # creating frames from the continuous waveform
        frames = tf.signal.frame(x, self.blockLen, self.block_shift)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frames)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]
    
    def fftLayer(self, x):
        '''
        Method for an fft helper layer used with a Lambda layer. The layer
        calculates the rFFT on the last dimension and returns the magnitude and
        phase of the STFT.
        '''
        
        # expanding dimensions
        frame = tf.expand_dims(x, axis=1)
        # calculating the fft over the time frames. rfft returns NFFT/2+1 bins.
        stft_dat = tf.signal.rfft(frame)
        # calculating magnitude and phase from the complex signal
        mag = tf.abs(stft_dat)
        phase = tf.math.angle(stft_dat)
        # returning magnitude and phase as list
        return [mag, phase]

 
        
    def ifftLayer(self, x):
        '''
        Method for an inverse FFT layer used with an Lambda layer. This layer
        calculates time domain frames from magnitude and phase information. 
        As input x a list with [mag,phase] is required.
        '''
        
        # calculating the complex representation
        s1_stft = (tf.cast(x[0], tf.complex64) * 
                    tf.exp( (1j * tf.cast(x[1], tf.complex64))))
        # returning the time domain frames
        return tf.signal.irfft(s1_stft)  
    
    
    def overlapAddLayer(self, x):
        '''
        Method for an overlap and add helper layer used with a Lambda layer.
        This layer reconstructs the waveform from a framed signal.
        '''

        # calculating and returning the reconstructed waveform
        return tf.signal.overlap_and_add(x, self.block_shift)
    
        

    def seperation_kernel(self, num_layer, mask_size, x, stateful=False):
        '''
        Method to create a separation kernel. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''

        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            x = LSTM(self.numUnits, return_sequences=True, stateful=stateful)(x)
            # using dropout between the LSTM layer for regularization 
            if idx<(num_layer-1):
                x = Dropout(self.dropout)(x)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        # returning the mask
        return mask
    
    def seperation_kernel_with_states(self, num_layer, mask_size, x, 
                                      in_states):
        '''
        Method to create a separation kernel, which returns the LSTM states. 
        !! Important !!: Do not use this layer with a Lambda layer. If used with
        a Lambda layer the gradients are updated correctly.

        Inputs:
            num_layer       Number of LSTM layers
            mask_size       Output size of the mask and size of the Dense layer
        '''
        
        states_h = []
        states_c = []
        # creating num_layer number of LSTM layers
        for idx in range(num_layer):
            in_state = [in_states[:,idx,:, 0], in_states[:,idx,:, 1]]
            x, h_state, c_state = LSTM(self.numUnits, return_sequences=True, 
                     unroll=True, return_state=True)(x, initial_state=in_state)
            # using dropout between the LSTM layer for regularization 
            if idx<(num_layer-1):
                x = Dropout(self.dropout)(x)
            states_h.append(h_state)
            states_c.append(c_state)
        # creating the mask with a Dense and an Activation layer
        mask = Dense(mask_size)(x)
        mask = Activation(self.activation)(mask)
        out_states_h = tf.reshape(tf.stack(states_h, axis=0), 
                                  [1,num_layer,self.numUnits])
        out_states_c = tf.reshape(tf.stack(states_c, axis=0), 
                                  [1,num_layer,self.numUnits])
        out_states = tf.stack([out_states_h, out_states_c], axis=-1)
        # returning the mask and states
        return mask, out_states

    def build_DTLN_model(self, norm_stft=False):
        '''
        Method to build and compile the DTLN model. The model takes time domain 
        batches of size (batchsize, len_in_samples) and returns enhanced clips 
        in the same dimensions. As optimizer for the Training process the Adam
        optimizer with a gradient norm clipping of 3 is used. 
        The model contains two separation cores. The first has an STFT signal 
        transformation and the second a learned transformation based on 1D-Conv 
        layer. 
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(None, None))
        # calculate STFT
        mag,angle = Lambda(self.stftLayer)(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen//2+1), mag_norm)
        # multiply mask with magnitude
        estimated_mag = Multiply()([mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frames_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frames = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        # create waveform with overlap and add procedure
        estimated_sig = Lambda(self.overlapAddLayer)(decoded_frames)

        
        # create the model
        self.model = Model(inputs=time_dat, outputs=estimated_sig)
        # show the model summary
        print(self.model.summary())
        
    def build_DTLN_model_stateful(self, norm_stft=False):
        '''
        Method to build stateful DTLN model for real time processing. The model 
        takes one time domain frame of size (1, blockLen) and one enhanced frame. 
         
        '''
        
        # input layer for time signal
        time_dat = Input(batch_shape=(1, self.blockLen))
        # calculate STFT
        mag,angle = Lambda(self.fftLayer)(time_dat)
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1 = self.seperation_kernel(self.numLayer, (self.blockLen//2+1), mag_norm, stateful=True)
        # multiply mask with magnitude
        estimated_mag = Multiply()([mag, mask_1])
        # transform frames back to time domain
        estimated_frames_1 = Lambda(self.ifftLayer)([estimated_mag,angle])
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,use_bias=False)(estimated_frames_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2 = self.seperation_kernel(self.numLayer, self.encoder_size, encoded_frames_norm, stateful=True)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frame = Conv1D(self.blockLen, 1, padding='causal',use_bias=False)(estimated)
        # create the model
        self.model = Model(inputs=time_dat, outputs=decoded_frame)
        # show the model summary
        print(self.model.summary())
        
    def compile_model(self):
        '''
        Method to compile the model for training

        '''
        
        # use the Adam optimizer with a clipnorm of 3
        optimizerAdam = keras.optimizers.Adam(lr=self.lr, clipnorm=3.0)
        # compile model with loss function
        self.model.compile(loss=self.lossWrapper(), optimizer=optimizerAdam)
        
    def create_saved_model(self, weights_file, target_name):
        '''
        Method to create a saved model folder from a weights file

        '''
        # check for type
        if weights_file.find('_norm_') != -1:
            norm_stft = True
        else:
            norm_stft = False
        # build model    
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        # load weights
        self.model.load_weights(weights_file)
        # save model
        tf.saved_model.save(self.model, target_name)
        
    def create_tf_lite_model(self, weights_file, target_name, use_dynamic_range_quant=False):
        '''
        Method to create a tf lite model folder from a weights file. 
        The conversion creates two models, one for each separation core. 
        Tf lite does not support complex numbers yet. Some processing must be 
        done outside the model.
        For further information and how real time processing can be 
        implemented see "real_time_processing_tf_lite.py".
        
        The conversion only works with TF 2.3.

        '''
        # check for type
        if weights_file.find('_norm_') != -1:
            norm_stft = True
            num_elements_first_core = 2 + self.numLayer * 3 + 2
        else:
            norm_stft = False
            num_elements_first_core = self.numLayer * 3 + 2
        # build model    
        self.build_DTLN_model_stateful(norm_stft=norm_stft)
        # load weights
        self.model.load_weights(weights_file)
        
        #### Model 1 ##########################
        mag = Input(batch_shape=(1, 1, (self.blockLen//2+1)))
        states_in_1 = Input(batch_shape=(1, self.numLayer, self.numUnits, 2))
        # normalizing log magnitude stfts to get more robust against level variations
        if norm_stft:
            mag_norm = InstantLayerNormalization()(tf.math.log(mag + 1e-7))
        else:
            # behaviour like in the paper
            mag_norm = mag
        # predicting mask with separation kernel  
        mask_1, states_out_1 = self.seperation_kernel_with_states(self.numLayer, 
                                                    (self.blockLen//2+1), 
                                                    mag_norm, states_in_1)
        
        model_1 = Model(inputs=[mag, states_in_1], outputs=[mask_1, states_out_1])
        
        #### Model 2 ###########################
        
        estimated_frame_1 = Input(batch_shape=(1, 1, (self.blockLen)))
        states_in_2 = Input(batch_shape=(1, self.numLayer, self.numUnits, 2))
        
        # encode time domain frames to feature domain
        encoded_frames = Conv1D(self.encoder_size,1,strides=1,
                                use_bias=False)(estimated_frame_1)
        # normalize the input to the separation kernel
        encoded_frames_norm = InstantLayerNormalization()(encoded_frames)
        # predict mask based on the normalized feature frames
        mask_2, states_out_2 = self.seperation_kernel_with_states(self.numLayer, 
                                                    self.encoder_size, 
                                                    encoded_frames_norm, 
                                                    states_in_2)
        # multiply encoded frames with the mask
        estimated = Multiply()([encoded_frames, mask_2]) 
        # decode the frames back to time domain
        decoded_frame = Conv1D(self.blockLen, 1, padding='causal',
                               use_bias=False)(estimated)
        
        model_2 = Model(inputs=[estimated_frame_1, states_in_2], 
                        outputs=[decoded_frame, states_out_2])
        
        # set weights to submodels
        weights = self.model.get_weights()
        model_1.set_weights(weights[:num_elements_first_core])
        model_2.set_weights(weights[num_elements_first_core:])
        # convert first model
        converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '_1.tflite', 'wb') as f:
              f.write(tflite_model)
        # convert second model    
        converter = tf.lite.TFLiteConverter.from_keras_model(model_2)
        if use_dynamic_range_quant:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with tf.io.gfile.GFile(target_name + '_2.tflite', 'wb') as f:
              f.write(tflite_model)
              
        print('TF lite conversion complete!')
        
    
    def train_model(self, runName, path_to_train_mix, path_to_train_speech, \
                    path_to_val_mix, path_to_val_speech):
        '''
        Method to train the DTLN model. 
        '''
        
        # create save path if not existent
        savePath = './models_'+ runName+'/' 
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        # create log file writer
        csv_logger = CSVLogger(savePath+ 'training_' +runName+ '.log')
        # create callback for the adaptive learning rate
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=3, min_lr=10**(-10), cooldown=1)
        # create callback for early stopping
        early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=10, verbose=0, mode='auto', baseline=None)
        # create model check pointer to save the best model
        checkpointer = ModelCheckpoint(savePath+runName+'.h5',
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       save_freq='epoch'
                                       )

        # calculate length of audio chunks in samples
        len_in_samples = int(np.fix(self.fs * self.len_samples / 
                                    self.block_shift)*self.block_shift)
        # create data generator for training data
        generator_input = audio_generator(path_to_train_mix, 
                                          path_to_train_speech, 
                                          len_in_samples, 
                                          self.fs, train_flag=True)
        dataset = generator_input.tf_data_set
        dataset = dataset.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of training steps in one epoch
        steps_train = generator_input.total_samples//self.batchsize
        # create data generator for validation data
        generator_val = audio_generator(path_to_val_mix,
                                        path_to_val_speech, 
                                        len_in_samples, self.fs)
        dataset_val = generator_val.tf_data_set
        dataset_val = dataset_val.batch(self.batchsize, drop_remainder=True).repeat()
        # calculate number of validation steps
        steps_val = generator_val.total_samples//self.batchsize
        # start the training of the model
        self.model.fit(
            x=dataset, 
            batch_size=None,
            steps_per_epoch=steps_train, 
            epochs=self.max_epochs,
            verbose=1,
            validation_data=dataset_val,
            validation_steps=steps_val, 
            callbacks=[checkpointer, reduce_lr, csv_logger, early_stopping],
            max_queue_size=50,
            workers=4,
            use_multiprocessing=True)
        # clear out garbage
        tf.keras.backend.clear_session()

    

class InstantLayerNormalization(Layer):
    '''
    Class implementing instant layer normalization. It can also be called 
    channel-wise layer normalization and was proposed by 
    Luo & Mesgarani (https://arxiv.org/abs/1809.07454v2) 
    '''

    def __init__(self, **kwargs):
        '''
            Constructor
        '''
        super(InstantLayerNormalization, self).__init__(**kwargs)
        self.epsilon = 1e-7 
        self.gamma = None
        self.beta = None

    def build(self, input_shape):
        '''
        Method to build the weights.
        '''
        shape = input_shape[-1:]
        # initialize gamma
        self.gamma = self.add_weight(shape=shape,
                             initializer='ones',
                             trainable=True,
                             name='gamma')
        # initialize beta
        self.beta = self.add_weight(shape=shape,
                             initializer='zeros',
                             trainable=True,
                             name='beta')
 

    def call(self, inputs):
        '''
        Method to call the Layer. All processing is done here.
        '''

        # calculate mean of each frame
        mean = tf.math.reduce_mean(inputs, axis=[-1], keepdims=True)
        # calculate variance of each frame
        variance = tf.math.reduce_mean(tf.math.square(inputs - mean), 
                                       axis=[-1], keepdims=True)
        # calculate standard deviation
        std = tf.math.sqrt(variance + self.epsilon)
        # normalize each frame independently 
        outputs = (inputs - mean) / std
        # scale with gamma
        outputs = outputs * self.gamma
        # add the bias beta
        outputs = outputs + self.beta
        # return output
        return outputs
    

    
