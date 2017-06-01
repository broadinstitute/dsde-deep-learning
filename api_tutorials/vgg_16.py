# -*- coding: utf-8 -*-
'''VGG16 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
from __future__ import print_function

import os
import cv2
import h5py
import numpy as np
from scipy.misc import imsave
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.layers import Flatten, Dense, Input
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.layers import Conv2D, ZeroPadding2D, MaxPooling2D


def vgg_16(num_labels=0, weights_path=None, input_tensor=None):
	'''Instantiate the VGG16 architecture,
	optionally loading weights pre-trained
	on ImageNet. Note that when using TensorFlow,
	for best performance you should set
	`image_dim_ordering="tf"` in your Keras config
	at ~/.keras/keras.json.

	The model and the weights are compatible with both
	TensorFlow and Theano. The dimension ordering
	convention used by the model is the one
	specified in your Keras config file.

	# Arguments
		num_labels: whether to include the 3 fully-connected
			layers at the top of the network.
		weights_path: `None` or a path to a weight file (.h5).
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.

	# Returns
		A Keras model instance.
	'''
	# Determine proper input shape
	if  K.image_data_format() == 'channels_first':
		if num_labels > 0:
			input_shape = (3, 224, 224)
		else:
			input_shape = (3, None, None)
	else:
		if num_labels > 0:
			input_shape = (224, 224, 3)
		else:
			input_shape = (None, None, 3)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor)
		else:
			img_input = input_tensor

	# Block 1
	x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
	x = Conv2D(64, (3, 3),  activation='relu', padding='same', name='block1_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

	# Block 2
	x = Conv2D(128, (3, 3),  activation='relu', padding='same', name='block2_conv1')(x)
	x = Conv2D(128, (3, 3),  activation='relu', padding='same', name='block2_conv2')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

	# Block 3
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

	# Block 4
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

	# Block 5
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

	if num_labels > 0:
		# Classification block
		x = Flatten(name='flatten')(x)
		x = Dense(4096, activation='relu', name='fc1')(x)
		x = Dense(4096, activation='relu', name='fc2')(x)
		if num_labels == 1000:
			x = Dense(num_labels, activation='softmax', name='predictions')(x)
		else:
			x = Dense(num_labels, activation='softmax', name='predictions_'+str(num_labels))(x)

	# Create model
	model = Model(img_input, x)

	# load weights
	if weights_path:
		print('K.image_data_format:', K.image_data_format())
		model.load_weights(weights_path, by_name=True)
		convert_all_kernels_in_model(model)

	model.summary()

	return model