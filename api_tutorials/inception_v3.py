from __future__ import print_function

import numpy as np
import warnings

from keras.models import Model
from keras import layers
from keras.layers import Activation, Flatten, Dense, Input, BatchNormalization, merge
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K


def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
	'''Utility function to apply conv + BN.
	'''
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	if K.image_data_format() == 'channels_first':
		bn_axis = 1
	else:
		bn_axis = 3
	x = Conv2D(filters, (num_row, num_col), strides=strides, padding=padding, use_bias=False, name=conv_name)(x)
	x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
	x = Activation('relu', name=name)(x)
	return x

def inception_v3(num_labels=0, weights_path=None, input_tensor=None):
	'''Instantiate the Inception v3 architecture,
	optionally loading weights pre-trained
	on ImageNet. Note that when using TensorFlow,
	for best performance you should set
	`image_dim_ordering="tf"` in your Keras config
	at ~/.keras/keras.json.

	The model and the weights are compatible with both
	TensorFlow and Theano. The dimension ordering
	convention used by the model is the one
	specified in your Keras config file.

	Note that the default input image size for this model is 299x299.

	# Arguments
		num_labels: classes at the top of the network.
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.

	# Returns
		A Keras model instance.num_labels
	'''

	# Determine proper input shape
	size = 299 
	if K.image_data_format()== 'channels_first':
		input_shape = (3, size, size)
	else:
		input_shape = (size, size, 3)

	if input_tensor is None:
		img_input = Input(shape=input_shape, name='input_image')
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape, name='input_image')
		else:
			img_input = input_tensor

	if K.image_dim_ordering() == 'th':
		channel_axis = 1
	else:
		channel_axis = 3

	x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
	x = conv2d_bn(x, 32, 3, 3, padding='valid')
	x = conv2d_bn(x, 64, 3, 3)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn(x, 80, 1, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, 3, padding='valid')
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# mixed 0, 1, 2: 35 x 35 x 256
	for i in range(3):
		branch1x1 = conv2d_bn(x, 64, 1, 1)

		branch5x5 = conv2d_bn(x, 48, 1, 1)
		branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

		branch3x3dbl = conv2d_bn(x, 64, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
		x = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool],
				  axis=channel_axis,
				  name='mixed' + str(i))

	# mixed 3: 17 x 17 x 768
	branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3,
							 strides=(2, 2), padding='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate([branch3x3, branch3x3dbl, branch_pool],
			  axis=channel_axis, name='mixed3')

	# mixed 4: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 128, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 128, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  axis=channel_axis, name='mixed4')

	# mixed 5, 6: 17 x 17 x 768
	for i in range(2):
		branch1x1 = conv2d_bn(x, 192, 1, 1)

		branch7x7 = conv2d_bn(x, 160, 1, 1)
		branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
		branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

		branch7x7dbl = conv2d_bn(x, 160, 1, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
				  axis=channel_axis, name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 192, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 160, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  axis=channel_axis, name='mixed7')

	# mixed 8: 8 x 8 x 1280
	branch3x3 = conv2d_bn(x, 192, 1, 1)
	branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
						  strides=(2, 2), padding='valid')

	branch7x7x3 = conv2d_bn(x, 192, 1, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3,
							strides=(2, 2), padding='valid')

	branch_pool = AveragePooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate([branch3x3, branch7x7x3, branch_pool],
			  axis=channel_axis, name='mixed8')

	# mixed 9: 8 x 8 x 2048
	for i in range(2):
		branch1x1 = conv2d_bn(x, 320, 1, 1)

		branch3x3 = conv2d_bn(x, 384, 1, 1)
		branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
		branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
		branch3x3 = layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

		branch3x3dbl = conv2d_bn(x, 448, 1, 1)
		branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
		branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
		branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
		branch3x3dbl = layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

		branch_pool = AveragePooling2D(
			(3, 3), strides=(1, 1), padding='same')(x)
		branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
		x = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
				  axis=channel_axis, name='mixed' + str(9 + i))

	if num_labels > 0:
		# Classification block
		# pool stride was 8
		x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(x)
		x = Flatten(name='flatten')(x)
		if num_labels != 1000:
			x = Dense(num_labels, activation='softmax', name=str(num_labels)+'_predictions')(x)
		else:
			x = Dense(num_labels, activation='softmax', name='predictions')(x)
						
	# Create model
	model = Model(img_input, x)


	# load weights
	if weights_path:
		model.load_weights(weights_path, by_name=True)
		convert_all_kernels_in_model(model)

	return model