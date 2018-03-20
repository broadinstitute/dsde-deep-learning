# models.py
#
# Neural Network Models for Variant Filtration in Keras
# This file contains model architectures in the top models section
# Training regimen code in the training section 
# (usually a flavor of stochastic gradient descent)
# Metrics to watch during training are defined in the metrics section
# Model evaluation code is at the end of the file
#
# December 2016
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import sys
import time
import math
import h5py
import json
import plots
import defines
import numpy as np
from collections import namedtuple

# Keras Imports
from keras import layers
from keras import metrics
import keras.backend as K
from keras.preprocessing import image
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import plot_model, to_categorical
from keras.utils.vis_utils import model_to_dot
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D, UpSampling1D, UpSampling2D, Conv2DTranspose
from keras.layers.convolutional import Convolution1D, Convolution2D, Convolution3D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers import Add, Input, Dense, Dropout, BatchNormalization, SpatialDropout2D, SpatialDropout1D, Activation, Flatten, Reshape, LSTM, merge, Permute, GlobalAveragePooling2D

ResidualLayer = namedtuple("ResidualLayer", "identity filters strides")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_sequential_model(args):
	""" Create sequential 1d Convolutional model with 3 layers for classifying variants.

	Three layers of convolution followed by two dense layers.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input
		args.input_symbols: Dict mapping input symbols to the index of each typically DNA (e.g. {'A':0, 'C':1, ...})
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	"""	
	in_shape = (args.window_size, len(args.input_symbols))
	model = Sequential()
	model.add(Convolution1D(input_shape=in_shape, input_length=args.window_size, nb_filter=320, filter_length=16, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=256, filter_length=16, activation="relu"))
	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=160, filter_length=16, activation="relu"))
	model.add(Dropout(0.2))	
	model.add(Flatten())

	model.add(Dense(output_dim=40, init='normal'))
	model.add(Activation('relu'))
	
	model.add( Dense(output_dim=len(args.labels), init='normal') )
	model.add( Activation('softmax'))
	
	sgd = SGD(lr=0.001, decay=1e-5, momentum=0.9, nesterov=True)
	model.compile(loss='binary_crossentropy', 
		optimizer=sgd, 
		metrics=get_metrics())

	model.summary()

	return model


def build_sequential_snp_indel_model(args):
	""" Build sequential 1d Convolutional model with 3 layers for classifying variants.

	Three layers of convolution followed by two dense layers.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input
		args.input_symbols: Dict mapping input symbols to the index of each typically DNA (e.g. {'A':0, 'C':1, ...})
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	"""	
	model = Sequential()
	model.add(Convolution1D(input_shape=(args.window_size, len(args.input_symbols)), 
		input_length=args.window_size, 
		nb_filter=320,
		filter_length=16, 
		activation="relu",
		init='normal'))

	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=256, filter_length=16, activation="relu", init='normal'))
	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=160, filter_length=16, activation="relu", init='normal'))
	model.add(Dropout(0.2))	
	model.add(Flatten())

	model.add(Dense(output_dim=40, init='normal'))
	model.add(Activation('relu'))
	
	model.add( Dense(output_dim=len(args.labels), init='normal') )
	model.add( Activation('softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(loss='mse', optimizer=adamo, metrics=get_metrics(args.labels))
	model.summary()

	return model


def build_reference_model(args):
	'''Build 1d Convolutional model with 3 layers for classifying variants.

	Three layers of convolution followed by two dense layers, uses the functional API.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''	
	channel_map = defines.get_tensor_channel_map_from_args(args)	
	reference = Input(shape=(args.window_size, len(channel_map)), name='reference')	

	conv_width = 9
	x = Conv1D(filters=216, kernel_size=conv_width, activation='relu')(reference)
	x = SpatialDropout1D(0.5)(x)
	x = Conv1D(filters=128, kernel_size=conv_width, activation='relu')(x)
	x = SpatialDropout1D(0.5)(x)
	x = Conv1D(filters=96, kernel_size=conv_width, activation='relu')(x)
	x = SpatialDropout1D(0.5)(x)	
	x = MaxPooling1D(3)(x)
	x = Flatten()(x)

	x = Dense(units=64, activation='relu')(x)
	x = Dropout(0.5)(x)	
	#x = Dense(units=32, activation='relu')(x)
	#x = Dropout(0.5)(x)	

	prob_output = Dense(units=len(args.labels), activation='softmax')(x)
	
	model = Model(inputs=[reference], outputs=[prob_output])
	
	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
		
	return model


def annotation_multilayer_perceptron_from_args(args,
											fc_layers = [128, 128, 128, 128],
											dropout = 0.3,
											initializer='glorot_normal',
											batch_normalize_input = False,
											batch_normalization = False,
											skip_connection = False):
	'''Build Multilayer perceptron for classifying variants.

	Four layers of dense connection, uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''
	annotations_in = Input(shape=(len(args.annotations),), name=args.annotation_set)
	
	if batch_normalize_input:
		x = annotations = BatchNormalization(axis=1)(annotations_in)
	else:
		x = annotations = annotations_in

	for l in fc_layers:
		
		if batch_normalization:
			x = Dense(units=l, activation='linear', kernel_initializer=initializer)(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)
		else:
			x = Dense(units=l, activation='relu', kernel_initializer=initializer)(x)
	
		if dropout > 0:
			x = Dropout(dropout)(x)
		if skip_connection:
			x = layers.concatenate([x, annotations], axis=1)	

	prob_output = Dense(units=len(args.labels), kernel_initializer=initializer, activation='softmax')(x)
	
	model = Model(inputs=[annotations_in], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)	
	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_annotation_multilayer_perceptron(args):
	'''Build Multilayer perceptron for classifying variants.

	Four layers of dense connection, uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''
	annotations = Input(shape=(len(args.annotations),), name=args.annotation_set)
	annotations_bn = BatchNormalization(axis=1)(annotations)
	
	x = Dense(units=216, kernel_initializer='glorot_normal', activation='relu')(annotations_bn)
	x = Dropout(0.3)(x)	
	x = Dense(units=160, kernel_initializer='glorot_normal', activation='relu')(x)
	x = Dropout(0.3)(x)	
	x = Dense(units=128, kernel_initializer='glorot_normal', activation='relu')(x)
	x = Dropout(0.3)(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)	
	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model	


def build_reference_1d_1layer_model(args):
	'''Build Reference plus bed tracks 1d CNN model for classifying variants.

	Single layer of convolution followed by dense connection, concatenated with annotations.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''	
	channel_map = defines.get_tensor_channel_map_from_args(args)	
	reference = Input(shape=(args.window_size, len(channel_map)), name=args.tensor_map)
	
	x = Conv1D(filters=320, kernel_size=9, activation="relu", kernel_initializer='glorot_normal')(reference)
	x = SpatialDropout1D(0.4)(x)
	x = Conv1D(filters=128, kernel_size=9, activation="relu", kernel_initializer='glorot_normal')(x)
	x = SpatialDropout1D(0.4)(x)	
	#x = MaxPooling1D(3)(x)
	x = Flatten()(x)

	x = Dense(units=32, activation='relu')(x)
	x = Dropout(0.4)(x)
	x = Dense(units=48, activation='relu')(x)
	x = Dropout(0.3)(x)

	prob_output = Dense(units=len(args.labels), name='softmax_predictions', activation='softmax')(x)
	
	model = Model(inputs=[reference], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
		
	return model


def build_reference_plus_model(args):
	'''Build Reference Plus bed tracks 1d CNN model for classifying variants.

	Convolutions followed by dense connection, concatenated with annotations.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''	
	channel_map = defines.get_tensor_channel_map_from_args(args)	
	reference = Input(shape=(args.window_size, len(channel_map)), name=args.tensor_map)
	conv_width = 11
	x = Conv1D(filters=256, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(reference)
	x = SpatialDropout1D(0.3)(x)
	x = Conv1D(filters=196, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(x)
	x = SpatialDropout1D(0.3)(x)
	x = Conv1D(filters=128, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(x)
	x = SpatialDropout1D(0.3)(x)
	x = Flatten()(x)

	annotations = Input(shape=(len(args.annotations),), name=args.annotation_set)
	annos_normed = BatchNormalization(axis=-1)(annotations)
	annos_normed_x = Dense(units=20, kernel_initializer='normal', activation='relu')(annos_normed)
	
	x = layers.concatenate([x, annos_normed_x], axis=-1)
	x = Dense(units=20, kernel_initializer='normal', activation='relu')(x)
	x = Dropout(0.2)(x)	
	x = layers.concatenate([x, annos_normed], axis=-1)

	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[reference, annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
		
	return model


def build_reference_annotation_skip_model(args):
	'''Build Reference 1d CNN model for classifying variants with skip connected annotations.

	Convolutions followed by dense connection, concatenated with annotations.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''
	if args.channels_last:
		channel_axis = -1
	else:
		channel_axis = 1

	channel_map = defines.get_tensor_channel_map_from_args(args)	
	reference = Input(shape=(args.window_size, len(channel_map)), name="reference")
	conv_width = 12
	conv_dropout = 0.1
	fc_dropout = 0.2
	x = Conv1D(filters=256, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(reference)
	x = Conv1D(filters=256, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(x)
	x = Dropout(conv_dropout)(x)
	x = Conv1D(filters=128, kernel_size=conv_width, activation="relu", kernel_initializer='he_normal')(x)
	x = Dropout(conv_dropout)(x)	
	x = Flatten()(x)

	annotations = Input(shape=(len(args.annotations),), name="annotations")
	annos_normed = BatchNormalization(axis=channel_axis)(annotations)
	annos_normed_x = Dense(units=40, kernel_initializer='normal', activation='relu')(annos_normed)

	x = layers.concatenate([x, annos_normed_x], axis=channel_axis)
	x = Dense(units=40, kernel_initializer='normal', activation='relu')(x)
	x = Dropout(fc_dropout)(x)	
	x = layers.concatenate([x, annos_normed], axis=channel_axis)

	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[reference, annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(optimizer=adamo, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
		
	return model


def build_reference_1d_model_from_args(args,
										conv_width = 6, 
										conv_layers = [128, 128, 128, 128],
										conv_dropout = 0.1,
										conv_batch_normalize = False,			
										spatial_dropout = True,
										max_pools = [3, 1],
										padding='valid',
										fc_layers = [64],
										fc_dropout = 0.3,
										fc_batch_normalize = False,
										fc_initializer='glorot_normal',
										kernel_initializer='glorot_normal'):
	'''Build Reference 1d CNN model for classifying variants.

	Architecture specified by parameters.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''	
	channel_map = defines.get_tensor_channel_map_1d()
	concat_axis = -1	
	x = reference = Input(shape=(args.window_size, len(channel_map)), name=args.tensor_map)

	max_pool_diff = len(conv_layers)-len(max_pools)	
	for  i,c in enumerate(conv_layers):

		if conv_batch_normalize:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation('relu')(x)
		else:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='relu', padding=padding, kernel_initializer=kernel_initializer)(x)

		if conv_dropout > 0 and spatial_dropout:
			x = SpatialDropout1D(conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling1D(max_pools[i-max_pool_diff])(x)

	x = Flatten()(x)

	for fc in fc_layers:
		if fc_batch_normalize:
			x = Dense(units=fc, activation='linear', kernel_initializer=fc_initializer)(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)			
		else:
			x = Dense(units=fc, activation='relu')(x)
		
		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)

	prob_output = Dense(units=len(args.labels), activation='softmax')(x)
	
	model = Model(inputs=[reference], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
	
	return model


def build_reference_annotation_1d_model_from_args(args,
													conv_width = 6, 
													conv_layers = [128, 128, 128, 128],
													conv_dropout = 0.0,
													conv_batch_normalize = False,		
													spatial_dropout = True,
													max_pools = [],
													padding='valid',
													annotation_units = 16,
													annotation_shortcut = False,
													annotation_batch_normalize = True,	
													fc_layers = [64],
													fc_dropout = 0.0,
													fc_batch_normalize = False,
													fc_initializer='glorot_normal',
													kernel_initializer='glorot_normal'
												):
	'''Build Reference 1d CNN model for classifying variants.

	Architecture specified by parameters.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.annotations: The variant annotations, perhaps from a HaplotypeCaller VCF.
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)

	Returns
		The keras model
	'''	
	channel_map = defines.get_tensor_channel_map_1d()
	concat_axis = -1	
	x = reference = Input(shape=(args.window_size, len(channel_map)), name=args.tensor_map)

	max_pool_diff = len(conv_layers)-len(max_pools)	
	for  i,c in enumerate(conv_layers):

		if conv_batch_normalize:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation('relu')(x)
		else:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='relu', padding=padding, kernel_initializer=kernel_initializer)(x)

		if conv_dropout > 0 and spatial_dropout:
			x = SpatialDropout1D(conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling1D(max_pools[i-max_pool_diff])(x)

	f = Flatten()(x)

	annotations = annotations_in = Input(shape=(len(args.annotations),), name=args.annotation_set)
	if annotation_batch_normalize:
		annotations_in = BatchNormalization(axis=concat_axis)(annotations_in)
	annotation_mlp = Dense(units=annotation_units, kernel_initializer=fc_initializer, activation='relu')(annotations_in)
	
	x = layers.concatenate([f, annotation_mlp], axis=1)
	for fc in fc_layers:
		if fc_batch_normalize:
			x = Dense(units=fc, activation='linear', kernel_initializer=fc_initializer)(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)		
		else:
			x = Dense(units=fc, activation='relu', kernel_initializer=fc_initializer)(x)
		
		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)
	
	if annotation_shortcut:
		x = layers.concatenate([x, annotations_in], axis=1)

	prob_output = Dense(units=len(args.labels), activation='softmax')(x)
	
	model = Model(inputs=[reference, annotations], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)
	
	return model


def read_tensor_2d_model_from_args(args, 
									conv_width = 6, 
									conv_height = 6,
									conv_layers = [128, 128, 128, 128],
									conv_dropout = 0.1,
									conv_batch_normalize = False,
									spatial_dropout = True,
									max_pools = [(3,1), (3,1)],
									padding='valid',
									fc_layers = [64],
									fc_dropout = 0.3,
									fc_batch_normalize = False,
									fc_initializer='glorot_normal',
									kernel_initializer='glorot_normal'
									):
	'''Builds Read Tensor 2d CNN model for classifying variants.

	Arguments specify widths and depths of each layer.
	2d Convolutions followed by dense connection mixed with annotation values.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.weights_hd5: An existing model file to load weights from
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag
		conv_layers: list of number of convolutional filters in each layer
		batch_normalization: Boolean whether to apply batch normalization or not
	Returns
		The keras model
	'''			
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		K.set_image_data_format('channels_last')
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	x = read_tensor_in = Input(shape=in_shape, name=args.tensor_map)

	# Add convolutional layers
	max_pool_diff = len(conv_layers)-len(max_pools)
	for i,f in enumerate(conv_layers):
		if i%2 == 0:
			cur_kernel = (conv_width, 1)
		else:
			cur_kernel = (1, conv_height)

		if conv_batch_normalize:
			x = Conv2D(f, cur_kernel, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation('relu')(x)
		else:
			x = Conv2D(f, cur_kernel, activation='relu', padding=padding, kernel_initializer=kernel_initializer)(x)

		if conv_dropout > 0 and spatial_dropout:
			x = SpatialDropout2D(conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling2D(max_pools[i-max_pool_diff])(x)

	x = Flatten()(x)

	# Fully connected layers
	for fc_units in fc_layers:
		if fc_batch_normalize:
			x = Dense(units=fc_units, kernel_initializer=fc_initializer, activation='linear')(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)
		else:
			x = Dense(units=fc_units, kernel_initializer=fc_initializer, activation='relu')(x)
		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)

	# Softmax output
	prob_output = Dense(units=len(args.labels), kernel_initializer=fc_initializer, activation='softmax')(x)
	
	# Map inputs to outputs
	model = Model(inputs=[read_tensor_in], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def read_tensor_2d_annotation_model_from_args(args, 
											conv_width = 6, 
											conv_height = 6,
											conv_layers = [128, 128, 128, 128],
											conv_dropout = 0.0,
											conv_batch_normalize = False,
											spatial_dropout = True,
											residual_layers = [],
											max_pools = [(3,1), (3,3)],
											padding='valid',
											annotation_units = 16,
											annotation_shortcut = False,
											annotation_batch_normalize = True,
											fc_layers = [64],
											fc_dropout = 0.0,
											fc_batch_normalize = False,
											kernel_initializer='glorot_normal',
											kernel_single_channel=True,
											freeze_bn=False,
											fc_initializer='glorot_normal'):
	'''Builds Read Tensor 2d CNN model with variant annotations mixed in for classifying variants.

	Arguments specify widths and depths of each layer.
	2d Convolutions followed by dense connection mixed with annotation values.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.weights_hd5: An existing model file to load weights from
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag
		conv_layers: list of number of convolutional filters in each layer
		batch_normalization: Boolean whether to apply batch normalization or not
	Returns
		The keras model
	'''			
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	x = read_tensor_in = Input(shape=in_shape, name=args.tensor_map)

	max_pool_diff = max(0, len(conv_layers)-len(max_pools))

	# Add convolutional layers
	for i,f in enumerate(conv_layers):	
		if kernel_single_channel and i%2 == 0:
			cur_kernel = (conv_width, 1)
		elif kernel_single_channel:
			cur_kernel = (1, conv_height)
		else:
			cur_kernel = (conv_width, conv_height)

		if conv_batch_normalize:
			x = Conv2D(f, cur_kernel, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation('relu')(x)
		else:
			x = Conv2D(f, cur_kernel, activation='relu', padding=padding, kernel_initializer=kernel_initializer)(x)

		if conv_dropout > 0 and spatial_dropout:
			x = SpatialDropout2D(conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling2D(max_pools[i-max_pool_diff])(x)

	for i,r in enumerate(residual_layers):
		if kernel_single_channel and i%2 == 0:
			cur_kernel = (conv_width, 1)
		elif kernel_single_channel:
			cur_kernel = (1, conv_height)
		else:
			cur_kernel = (conv_width, conv_height)

		y = Conv2D(r.filters[0], (1, 1), strides=r.strides)(x)
		y = BatchNormalization(axis=concat_axis)(y)
		y = Activation('relu')(y)

		y = Conv2D(r.filters[1], cur_kernel, padding='same')(y)
		y = BatchNormalization(axis=concat_axis)(y)
		y = Activation('relu')(y)

		y = Conv2D(r.filters[2], (1, 1))(y)
		y = BatchNormalization(axis=concat_axis)(y)

		if r.identity:
			x = layers.add([y, x])
		else:
			shortcut = Conv2D(r.filters[2], (1, 1), strides=r.strides)(x)
			shortcut = BatchNormalization(axis=concat_axis)(shortcut)
			x = layers.add([y, shortcut])
		
		x = Activation('relu')(x)

	x = Flatten()(x)

	# Mix the variant annotations in
	annotations = annotations_in = Input(shape=(len(args.annotations),), name=args.annotation_set)
	if annotation_batch_normalize:
		annotations_in = BatchNormalization(axis=-1)(annotations)

	annotations_mlp = Dense(units=annotation_units, kernel_initializer=fc_initializer, activation='relu')(annotations_in)
	x = layers.concatenate([x, annotations_mlp], axis=concat_axis)

	# Fully connected layers
	for fc_units in fc_layers:
		
		if fc_batch_normalize:
			x = Dense(units=fc_units, kernel_initializer=fc_initializer, activation='linear')(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)
		else:
			x = Dense(units=fc_units, kernel_initializer=fc_initializer, activation='relu')(x)		

		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)

	if annotation_shortcut:
		x = layers.concatenate([x, annotations_in], axis=concat_axis)

	# Softmax output
	prob_output = Dense(units=len(args.labels), kernel_initializer=fc_initializer, activation='softmax')(x)
	
	# Map inputs to outputs
	model = Model(inputs=[read_tensor_in, annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_read_tensor_2d_model(args):
	'''Build Read Tensor 2d CNN model for classifying variants.

	2d Convolutions followed by dense connection.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	'''		
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)

	read_tensor = Input(shape=in_shape, name="read_tensor")
	read_conv_width = 16
	conv_dropout = 0.2
	fc_dropout = 0.3

	x = Conv2D(216, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(read_tensor)
	#x = MaxPooling2D((2,1))(x)
	x = Conv2D(160, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	#x = Dropout(conv_dropout)(x)
	x = Conv2D(128, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)
	#x = Dropout(conv_dropout)(x)
	x = Conv2D(96, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = Dropout(conv_dropout)(x)
	x = MaxPooling2D((2,1))(x)
	x = Conv2D(64, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = Dropout(conv_dropout)(x)
	x = MaxPooling2D((3,1))(x)						

	x = Flatten()(x)
	x = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(x)
	x = Dropout(fc_dropout)(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_read_tensor_2d_and_annotations_model(args):
	'''Build Read Tensor 2d CNN model with variant annotations mixed in for classifying variants.

	2d Convolutions followed by dense connection mixed with annotation values.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	'''		
	#print('IN MODEL K.image_data_format:', K.image_data_format())
	#K.set_image_data_format('channels_first')
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	read_tensor = Input(shape=in_shape, name=args.tensor_map)

	read_conv_width = 16
	conv_dropout = 0.2
	fc_dropout = 0.3			
	x = Conv2D(216, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(read_tensor)
	x = Conv2D(160, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = Conv2D(128, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)
	x = Conv2D(96, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)
	x = Dropout(conv_dropout)(x)
	x = Conv2D(64, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)						
	x = Dropout(conv_dropout)(x)

	x = Flatten()(x)

	# Mix the variant annotations in
	annotations = Input(shape=(len(args.annotations),), name=args.annotation_set)
	annotations_bn = BatchNormalization(axis=1)(annotations)
	alt_input_mlp = Dense(units=16, kernel_initializer='glorot_normal', activation='relu')(annotations_bn)
	x = layers.concatenate([x, alt_input_mlp], axis=concat_axis)

	x = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(x)
	x = layers.concatenate([x, annotations_bn], axis=concat_axis)
	x = Dropout(fc_dropout)(x)

	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor, annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_pileup_filter(args):
	'''Build Pileup Tensor 1d CNN for filtering variants'''
	channels = defines.get_reference_and_read_channels(args)
	in_shape = (args.window_size, channels)

	pileup_tensor = Input(shape=in_shape, name="pileup_tensor")

	read_conv_width = 8
	conv_dropout = 0.1
	fc_dropout = 0.2
	
	conv1 = Conv1D(256, read_conv_width, activation="relu", padding='same')(pileup_tensor)
	conv1 = Conv1D(256, read_conv_width, activation='relu', padding='same')(conv1)
	pool1 = MaxPooling1D(pool_size=2)(conv1)
	
	conv2 = Conv1D(128, read_conv_width, activation='relu', padding='same')(pool1)
	conv2 = Conv1D(128, read_conv_width, activation='relu', padding='same')(conv2)
	pool2 = MaxPooling1D(pool_size=2)(conv2)

	conv3 = Conv1D(64, read_conv_width, activation='relu', padding='same')(pool2)
	conv3 = Conv1D(64, read_conv_width, activation='relu', padding='same')(conv3)
	pool3 = MaxPooling1D(pool_size=2)(conv3)
	
	x = Dropout(conv_dropout)(pool3)
	x = Flatten()(x)

	x = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(x)
	x = Dropout(fc_dropout)(x)
	
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[pileup_tensor], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_read_tensor_2d_dilated_model(args):
	'''Build Read Tensor 2d CNN model with Dilations  for classifying variants.

	2d Convolutions followed by dilated convolutions followed by dense connection.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	'''		
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)

	read_tensor = Input(shape=in_shape, name="read_tensor")
	read_conv_width = 16
	conv_dropout = 0.4
	fc_dropout = 0.3

	x = Conv2D(180, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(read_tensor)
	x = Conv2D(128, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	#x = Dropout(conv_dropout)(x)
	x = MaxPooling2D((2,1))(x)
	x = Conv2D(128, (1, read_conv_width), dilation_rate=(1,2), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	#x = Dropout(conv_dropout)(x)
	x = Conv2D(96, (1, read_conv_width),  dilation_rate=(1,4), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	#x = Dropout(conv_dropout)(x)
	x = MaxPooling2D((2,1))(x)						
	x = Conv2D(64, (1, read_conv_width),  dilation_rate=(1,8), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)						
	x = Conv2D(48, (1, read_conv_width),  dilation_rate=(1,16), padding='valid', activation="relu", kernel_initializer="he_normal")(x)

	x = Flatten()(x)
	x = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(x)
	#x = Dropout(fc_dropout)(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=get_metrics(args.labels))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_1d_cnn_calling_segmentation_1d(args):
	'''Build Read Tensor 1d CNN for calling variants as 1d genotyped segmentation'''
	channels = defines.get_reference_and_read_channels(args)
	in_shape = (args.window_size, channels)

	pileup_tensor = Input(shape=in_shape, name="pileup_tensor")

	read_conv_width = 8 # was 8
	conv_dropout = 0.1
	fc_dropout = 0.2
	p_mode = 'same'

	conv1 = Conv1D(64, read_conv_width, activation="relu", padding=p_mode)(pileup_tensor)
	conv1 = Conv1D(64, read_conv_width, activation='relu', padding=p_mode)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(conv1)
	
	conv2 = Conv1D(128, read_conv_width, activation='relu', padding=p_mode)(pool1)
	conv2 = Conv1D(128, read_conv_width, activation='relu', padding=p_mode)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(conv2)

	conv4 = Conv1D(256, read_conv_width, activation='relu', padding=p_mode)(pool2)
	conv4 = Conv1D(256, read_conv_width, activation='relu', padding=p_mode)(conv4)

	up5 = layers.concatenate([UpSampling1D(2)(conv4), conv2], axis=-1)
	conv5 = Conv1D(128, read_conv_width, activation='relu', padding=p_mode)(up5)
	conv5 = Conv1D(128, read_conv_width, activation='relu', padding=p_mode)(conv5)	

	up9 = layers.concatenate([UpSampling1D(2)(conv5), conv1], axis=-1)
	conv9 = Conv1D(64, read_conv_width, activation='relu', padding=p_mode)(up9)
	conv9 = Conv1D(64, read_conv_width, activation='relu', padding=p_mode)(conv9)
	
	conv_label = Conv1D(len(args.labels), 1, activation='relu', padding=p_mode)(conv9)

	conv_out = Activation('softmax', name='site_labels')(conv_label)

	model = Model(inputs=pileup_tensor, outputs=conv_out)

	weights = np.array([0.1,3,3,6,4,6,4])
	#weights = np.array([0.1,5,5,5,5,5,5])

	weighted_loss = weighted_categorical_crossentropy(weights)

	model.compile(optimizer=Adam(lr=1e-4), loss=weighted_loss, metrics=get_metrics(args.labels, dim=3))
	#model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=get_metrics(args.labels, dim=3))

	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5, '\nLoss weights are:', weights)

	return model


def build_2d_cnn_calling_segmentation_1d(args):
	'''Build Read Tensor 2d CNN for calling variants as 1d genotyped segmentation'''
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	read_tensor = Input(shape=in_shape, name="read_tensor")

	read_conv_width = 9
	read_conv_height = 35	

	pileup_filters = 64
	padding_mode = 'same'
	
	
	x = Conv2D(96, (read_conv_height, 1), padding='valid', activation="relu")(read_tensor)
	x = Conv2D(pileup_filters, (args.read_limit-read_conv_height+1, 1), padding='valid', activation="relu")(x)

	x = Reshape((pileup_filters, args.window_size))(x)
	piled_up = Permute((2, 1))(x)

	conv1 = Conv1D(128, read_conv_width, activation="relu", padding=padding_mode)(piled_up)
	conv1 = Conv1D(128, read_conv_width, activation='relu', padding=padding_mode)(conv1)

	up9 = layers.concatenate([conv1, piled_up], axis=-1)
	conv9 = Conv1D(128, read_conv_width, activation='relu', padding=padding_mode)(up9)
	conv9 = Conv1D(128, read_conv_width, activation='relu', padding=padding_mode)(conv9)

	up10 = layers.concatenate([conv9, piled_up], axis=-1)
	conv10 = Conv1D(128, read_conv_width, activation='relu', padding=padding_mode)(up10)
	conv10 = Conv1D(128, read_conv_width, activation='relu', padding=padding_mode)(conv10)
	conv10 = Conv1D(128, 3, activation='relu', padding=padding_mode)(conv10)
	conv11 = layers.concatenate([conv10, piled_up], axis=-1)

	conv_label = Conv1D(len(args.labels), 1, activation="linear", padding=padding_mode)(conv11)
	conv_out = Activation('softmax')(conv_label)

	model = Model(inputs=read_tensor, outputs=conv_out)
	weights = np.array([0.5,3,2,1,1,1,1])
	weighted_loss = weighted_categorical_crossentropy(weights)
	adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

	model.compile(optimizer=adam, loss=weighted_loss, metrics=get_metrics(args.labels, dim=3))
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5, '\nLoss weights are:', weights)

	return model


def build_2d_cnn_calling_segmentation_full_2d(args):
	'''Build Read Tensor 2d CNN for calling variants as 1d genotyped segmentation'''
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	conv_width = 7
	conv_height = 7 
	pileup_filters = 32
	read_tensor = Input(shape=in_shape, name="read_tensor")

	conv3 = Conv2D(96, (1, conv_width), activation='relu', padding='same')(read_tensor)
	conv3 = Conv2D(96, (conv_height, 1), activation='relu', padding='same')(conv3)
	#conv3 = Conv2D(128, (1, conv_width), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv5 = Conv2D(64, (1, conv_width), activation='relu', padding='same')(pool3)
	conv5 = Conv2D(64, (conv_height, 1), activation='relu', padding='same')(conv5)

	up7 = layers.concatenate([UpSampling2D((2, 2))(conv5), conv3], axis=concat_axis)
	conv7 = Conv2D(96, (1, conv_width), activation='relu', padding='same')(up7)
	conv7 = Conv2D(96, (conv_height, 1), activation='relu', padding='same')(conv7)

	up8 = layers.concatenate([conv7, read_tensor], axis=concat_axis)

	x = Conv2D(pileup_filters, (args.read_limit, 1), padding='valid', activation='linear')(up8)

	x = Reshape((pileup_filters, args.window_size))(x)
	x = Permute((2, 1))(x)

	x = Conv1D(128, 9, activation='relu', padding='same')(x)

	conv_label = Conv1D(len(args.labels), 1, activation="relu", padding='same')(x)
	conv_out = Activation('softmax')(conv_label)
	#conv_out = Activation('softmax')(x)

	model = Model(inputs=read_tensor, outputs=conv_out)
	weights = np.array([0.1,5,5,12,7,12,6])
	weighted_loss = weighted_categorical_crossentropy(weights)
	model.compile(optimizer=Adam(lr=1e-4), loss=weighted_loss, metrics=get_metrics(args.labels, dim=3))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5, '\nLoss weights are:', weights)

	return model



def get_unet():
	inputs = Input((img_rows, img_cols, 1))
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
	conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
	conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
	pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
	conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
	pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
	conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
	pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
	conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

	up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
	conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

	up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
	conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

	up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
	conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

	up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
	conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)


	conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
	conv_label = Conv1D(len(args.labels), 1, activation='relu',border_mode='same')(conv10)



	conv_out = core.Activation('softmax')(conv_label)

	model = Model(input=inputs, output=conv_out)

	if not optimizer is None:
		model.compile(loss="categorical_crossentropy", optimizer= optimizer , metrics=['accuracy'] )

	model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
	model.summary()

	return model


def build_read_tensor_2d_annotations_exome_model(args):
	'''Build Read Tensor 2d CNN model with variant annotations mixed in for classifying variants.

	2d Convolutions followed by dense connection mixed with annotation values.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	'''			
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		concat_axis = -1
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		concat_axis = 1

	read_tensor = Input(shape=in_shape, name="read_tensor")

	read_conv_width = 8
	read_conv_height = 16	
	conv_dropout = 0.1
	fc_dropout = 0.2			
	x = Conv2D(208, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(read_tensor)
	x = Dropout(conv_dropout)(x)	
	x = Conv2D(160, (1, read_conv_height), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = Dropout(conv_dropout)(x)	
	x = Conv2D(128, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = Dropout(conv_dropout)(x)	
	x = Conv2D(108, (1, read_conv_height), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)
	x = Dropout(conv_dropout)(x)	
	x = Conv2D(96, (1, read_conv_height), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((2,1))(x)		
	x = Dropout(conv_dropout)(x)
	x = Conv2D(64, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((3,2))(x)
	x = Dropout(conv_dropout)(x)
	x = Flatten()(x)

	# Mix the variant annotations in
	annotations = Input(shape=(len(args.annotations),), name="annotations")
	alt_input_mlp = Dense(units=8, kernel_initializer='glorot_normal', activation='relu')(annotations)
	x = layers.concatenate([x, alt_input_mlp], axis=concat_axis)

	x = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(x)
	x = Dropout(fc_dropout)(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor, annotations], outputs=[prob_output])
	
	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model




def identity_block(input_tensor, kernel_size, filters, stage, block):
	"""The identity block is the block that has no conv layer at shortcut.
	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size,
			   padding='same', name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	x = layers.add([x, input_tensor])
	x = Activation('relu')(x)
	return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
	"""conv_block is the block that has a conv layer at shortcut
	# Arguments
		input_tensor: input tensor
		kernel_size: defualt 3, the kernel size of middle conv layer at main path
		filters: list of integers, the filterss of 3 conv layer at main path
		stage: integer, current stage label, used for generating layer names
		block: 'a','b'..., current block label, used for generating layer names
	# Returns
		Output tensor for the block.
	Note that from stage 3, the first conv layer at main path is with strides=(2,2)
	And the shortcut should have strides=(2,2) as well
	"""
	filters1, filters2, filters3 = filters
	if K.image_data_format() == 'channels_last':
		bn_axis = 3
	else:
		bn_axis = 1
	conv_name_base = 'res' + str(stage) + block + '_branch'
	bn_name_base = 'bn' + str(stage) + block + '_branch'

	x = Conv2D(filters1, (1, 1), strides=strides,
			   name=conv_name_base + '2a')(input_tensor)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters2, kernel_size, padding='same',
			   name=conv_name_base + '2b')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
	x = Activation('relu')(x)

	x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

	shortcut = Conv2D(filters3, (1, 1), strides=strides,
					  name=conv_name_base + '1')(input_tensor)
	shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

	x = layers.add([x, shortcut])
	x = Activation('relu')(x)
	return x


def conv2d_bn(x,
			  filters,
			  num_row,
			  num_col,
			  padding='same',
			  strides=(1, 1),
			  name=None):
	"""Utility function to apply conv + BN.
	# Arguments
		x: input tensor.
		filters: filters in `Conv2D`.
		num_row: height of the convolution kernel.
		num_col: width of the convolution kernel.
		padding: padding mode in `Conv2D`.
		strides: strides in `Conv2D`.
		name: name of the ops; will become `name + '_conv'`
			for the convolution and `name + '_bn'` for the
			batch norm layer.
	# Returns
		Output tensor after applying `Conv2D` and `BatchNormalization`.
	"""
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
	x = Conv2D(
		filters, (num_row, num_col),
		strides=strides,
		padding=padding,
		use_bias=False,
		name=conv_name)(x)
	x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
	x = Activation('relu', name=name)(x)
	return x


def build_read_tensor_2d_residual_model(args):
	"""Build Read Tensor 2d Residual Network model for classifying variants.

	Repeated 2d Convolutions followed by addition of the input, followed by dense connection.
	See: https://arxiv.org/abs/1512.03385
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	"""			
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		channel_axis = 3
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		channel_axis = 1

	read_tensor = Input(shape=in_shape, name="read_tensor")
	read_conv_width = 12
	conv_dropout = 0.0
	num_filters = 128

	x = Conv2D(316, (read_conv_width, 1), padding='valid', kernel_initializer="he_normal")(read_tensor)
	x = BatchNormalization(axis=channel_axis, name='bn_conv1')(x)
	x = Dropout(conv_dropout)(x)
	x = Activation('relu')(x)
	x = Conv2D(160, (1, read_conv_width), padding='valid', kernel_initializer="he_normal")(x)
	x = BatchNormalization(axis=channel_axis, name='bn_conv2')(x)
	x = Dropout(conv_dropout)(x)
	x = Activation('relu')(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

	x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
	x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

	x = conv_block(x, 3, [256, 256, 512], stage=4, block='a')
	x = identity_block(x, 3, [256, 256, 512], stage=4, block='b')
	x = identity_block(x, 3, [256, 256, 512], stage=4, block='c')
	x = identity_block(x, 3, [256, 256, 512], stage=4, block='d')
	x = identity_block(x, 3, [256, 256, 512], stage=4, block='e')
	x = identity_block(x, 3, [256, 256, 512], stage=4, block='f')

	x = AveragePooling2D((2, 2), name='avg_pool')(x)

	x = Flatten()(x)
	x = Dense(units=len(args.labels), activation='softmax', name='fully_connected')(x)

	model = Model(read_tensor, x, name='resnet50')

	adamo = Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)#, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_read_tensor_2d_inception_model(args):
	"""Build Read Tensor 2d Inception Network model for classifying variants.

	Repeated 2d Convolutions followed by addition of the input, followed by dense connection.
	See: https://arxiv.org/abs/1512.03385
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	"""			
	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
		channel_axis = 3
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
		channel_axis = 1

	read_tensor = Input(shape=in_shape, name="read_tensor")

	x = conv2d_bn(read_tensor, 32, 3, 3, strides=(2, 2), padding='valid')
	x = conv2d_bn(x, 32, 3, 3, padding='valid')
	x = conv2d_bn(x, 64, 3, 3)
	#x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn(x, 80, 1, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, 3, padding='valid')
	#x = MaxPooling2D((3, 3), strides=(2, 2))(x)
	# mixed 0, 1, 2: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
	x = layers.concatenate(
		[branch1x1, branch5x5, branch3x3dbl, branch_pool],
		axis=channel_axis,
		name='mixed0')

	# mixed 1: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
	x = layers.concatenate(
		[branch1x1, branch5x5, branch3x3dbl, branch_pool],
		axis=channel_axis,
		name='mixed1')

	# mixed 2: 35 x 35 x 256
	branch1x1 = conv2d_bn(x, 64, 1, 1)

	branch5x5 = conv2d_bn(x, 48, 1, 1)
	branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
	x = layers.concatenate(
		[branch1x1, branch5x5, branch3x3dbl, branch_pool],
		axis=channel_axis,
		name='mixed2')

	# mixed 3: 17 x 17 x 768
	branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

	branch3x3dbl = conv2d_bn(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn(
		branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate(
		[branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name='mixed3')

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
	x = layers.concatenate(
		[branch1x1, branch7x7, branch7x7dbl, branch_pool],
		axis=channel_axis,
		name='mixed4')

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
		x = layers.concatenate(
			[branch1x1, branch7x7, branch7x7dbl, branch_pool],
			axis=channel_axis,
			name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	branch1x1 = conv2d_bn(x, 192, 1, 1)

	branch7x7 = conv2d_bn(x, 192, 1, 1)
	branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
	branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn(x, 192, 1, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

	branch_pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	x = layers.concatenate(
		[branch1x1, branch7x7, branch7x7dbl, branch_pool],
		axis=channel_axis,
		name='mixed7')

	# mixed 8: 8 x 8 x 1280
	branch3x3 = conv2d_bn(x, 192, 1, 1)
	branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
						  strides=(2, 2), padding='valid')

	branch7x7x3 = conv2d_bn(x, 192, 1, 1)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
	branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
	branch7x7x3 = conv2d_bn(
		branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = layers.concatenate(
		[branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name='mixed8')

	# mixed 9: 8 x 8 x 2048
	# for i in range(2):
	#     branch1x1 = conv2d_bn(x, 320, 1, 1)

	#     branch3x3 = conv2d_bn(x, 384, 1, 1)
	#     branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
	#     branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
	#     branch3x3 = layers.concatenate(
	#         [branch3x3_1, branch3x3_2], axis=channel_axis, name='mixed9_' + str(i))

	#     branch3x3dbl = conv2d_bn(x, 448, 1, 1)
	#     branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
	#     branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
	#     branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
	#     branch3x3dbl = layers.concatenate(
	#         [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

	#     branch_pool = AveragePooling2D(
	#         (3, 3), strides=(1, 1), padding='same')(x)
	#     branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
	#     x = layers.concatenate(
	#         [branch1x1, branch3x3, branch3x3dbl, branch_pool],
	#         axis=channel_axis,
	#		name='mixed' + str(9 + i))

	# Classification block
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	x = Dense(len(args.labels), activation='softmax', name='predictions')(x)

	model = Model(read_tensor, x, name='inception_v3')

	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def inception_node(x, filters, channel_axis):
	""" Build an inception node.

	Parallel 1x1 convolution (channel-wise) followed by 2d Convolutions (spatial) followed by concatenation.
	See: https://arxiv.org/abs/1409.4842

	Arguments
		x: The input.	
		filters: The number of filters to use (the width of the inception node).
		channel_axis: The tensor axis for the channels (1 for theano, 3 for tensorflow) 
	Returns
		The output activations
	"""		
	tower_1 = Convolution2D(filters, 1, 1, border_mode='same', activation='relu')(x)
	tower_1 = Convolution2D(filters, 3, 3, border_mode='same', activation='relu')(tower_1)

	tower_2 = Convolution2D(filters, 1, 1, border_mode='same', activation='relu')(x)
	tower_2 = Convolution2D(filters, 5, 5, border_mode='same', activation='relu')(tower_2)

	tower_3 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
	tower_3 = Convolution2D(filters, 1, 1, border_mode='same', activation='relu')(tower_3)

	return merge([tower_1, tower_2, tower_3], mode='concat', concat_axis=channel_axis)


def build_bqsr_model(args):
	'''Build bqsr sequential 1d Convolutional model with 3 layers for classifying read bases.

	Three layers of convolution followed by two dense layers.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input
		args.input_symbols: Dict mapping input symbols to the index of each typically DNA (e.g. {'A':0, 'C':1, ...})
		args.labels: The dict of output labels (e.g. {'GOOD_BASE':0, 'BAD_BASE':1} )

	Returns
		The keras model
	'''	
	model = Sequential()
	model.add(Conv1D(name="read_tensor", 
		input_shape=(args.window_size, len(args.input_symbols)), 
		filters=320,
		kernel_size=3, 
		activation="relu",
		kernel_initializer='glorot_normal'))

	model.add(Dropout(0.2))
	model.add(Conv1D(filters=256, kernel_size=3, activation="relu", kernel_initializer='glorot_normal'))
	model.add(Dropout(0.2))
	model.add(Conv1D(filters=160, kernel_size=3, activation="relu", kernel_initializer='glorot_normal'))
	model.add(Dropout(0.2))	
	model.add(Flatten())

	model.add(Dense(units=40, kernel_initializer='glorot_normal'))
	model.add(Activation('relu'))
	
	model.add( Dense(units=len(args.labels), kernel_initializer='glorot_normal') )
	model.add( Activation('softmax'))

	adamo = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(loss='binary_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	model.summary()

	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def build_bqsr_annotation_model(args):
	'''Build bqsr sequential 1d Convolutional model with 3 layers for classifying read bases.

	Three layers of convolution followed by two dense layers.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input
		args.input_symbols: Dict mapping input symbols to the index of each typically DNA (e.g. {'A':0, 'C':1, ...})
		args.labels: The dict of output labels (e.g. {'GOOD_BASE':0, 'BAD_BASE':1} )

	Returns
		The keras model
	'''	
	read_tensor = Input(shape=(args.window_size, len(args.input_symbols)), name='read_tensor')
	x = Conv1D(filters=320, kernel_size=3, activation="relu", kernel_initializer='glorot_normal')(read_tensor)
	x = Dropout(0.2)(x)
	x = Conv1D(filters=256, kernel_size=3, activation="relu", kernel_initializer='glorot_normal')(x)
	x = Dropout(0.2)(x)
	x = Conv1D(filters=160, kernel_size=3, activation="relu", kernel_initializer='glorot_normal')(x)
	x = Dropout(0.2)(x)
	x = Flatten()(x)

	x = Dense(units=40, activation="relu", kernel_initializer='glorot_normal')(x)
	
	# Mix the read annotations in
	annotations = Input(shape=(len(args.annotations),), name="annotations")
	alt_input_mlp = Dense(units=32, kernel_initializer='glorot_normal', activation='relu')(annotations)
	x = layers.concatenate([x, alt_input_mlp], axis=1)

	x = Dense(units=48, kernel_initializer='glorot_normal', activation='relu')(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='glorot_normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor, annotations], outputs=[prob_output])

	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)

	model.compile(loss='binary_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))
	model.summary()

	return model


def build_bqsr_lstm_model(args):
	model = Sequential()
	model.add(LSTM(128, name="read_tensor", input_shape=(args.window_size, len(args.input_symbols))))
	model.add(LSTM(128, name="read_tensor", input_shape=(args.window_size, len(args.input_symbols))))
	model.add(Dense(len(args.labels), activation='softmax'))

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='binary_crossentropy', optimizer=optimizer)
	model.summary()
	return model


def conv2d_bn_old(x, nb_filter, nb_row, nb_col,
			  border_mode='same', subsample=(1, 1),
			  name=None):
	'''Utility function to apply conv + BN.
	'''
	if name is not None:
		bn_name = name + '_bn'
		conv_name = name + '_conv'
	else:
		bn_name = None
		conv_name = None
	if K.image_dim_ordering() == 'th':
		bn_axis = 1
	else:
		bn_axis = 3
	x = Convolution2D(nb_filter, nb_row, nb_col,
					  subsample=subsample,
					  activation='relu',
					  border_mode=border_mode,
					  name=conv_name)(x)
	x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
	return x


def inception_v3_max(args, input_tensor=None, architecture=''):
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
		args.labels: classes at the top of the network.
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.

	# Returns
		A Keras model instance.
	'''

	# Determine proper input shape
	size = 299 
	if K.image_dim_ordering() == 'th':
		input_shape = (3, size, size)

	else:
		input_shape = (size, size, 3)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor)
		else:
			img_input = input_tensor

	if K.image_dim_ordering() == 'th':
		channel_axis = 1
	else:
		channel_axis = 3

	x = conv2d_bn_old(img_input, 32, 3, 3, subsample=(2, 2), border_mode='valid')
	x = conv2d_bn_old(x, 32, 3, 3, border_mode='valid')
	x = conv2d_bn_old(x, 64, 3, 3)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv2d_bn_old(x, 80, 1, 1, border_mode='valid')
	x = conv2d_bn_old(x, 192, 3, 3, border_mode='valid')
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	# mixed 0, 1, 2: 35 x 35 x 256
	for i in range(3):
		branch1x1 = conv2d_bn_old(x, 64, 1, 1)

		branch5x5 = conv2d_bn_old(x, 48, 1, 1)
		branch5x5 = conv2d_bn_old(branch5x5, 64, 5, 5)

		branch3x3dbl = conv2d_bn_old(x, 64, 1, 1)
		branch3x3dbl = conv2d_bn_old(branch3x3dbl, 96, 3, 3)
		branch3x3dbl = conv2d_bn_old(branch3x3dbl, 96, 3, 3)

		branch_pool = MaxPooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn_old(branch_pool, 32, 1, 1)
		x = merge([branch1x1, branch5x5, branch3x3dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(i))

	# mixed 3: 17 x 17 x 768
	branch3x3 = conv2d_bn_old(x, 384, 3, 3, subsample=(2, 2), border_mode='valid')

	branch3x3dbl = conv2d_bn_old(x, 64, 1, 1)
	branch3x3dbl = conv2d_bn_old(branch3x3dbl, 96, 3, 3)
	branch3x3dbl = conv2d_bn_old(branch3x3dbl, 96, 3, 3,
							 subsample=(2, 2), border_mode='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = merge([branch3x3, branch3x3dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed3')

	# mixed 4: 17 x 17 x 768
	branch1x1 = conv2d_bn_old(x, 192, 1, 1)

	branch7x7 = conv2d_bn_old(x, 128, 1, 1)
	branch7x7 = conv2d_bn_old(branch7x7, 128, 1, 7)
	branch7x7 = conv2d_bn_old(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn_old(x, 128, 1, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 128, 1, 7)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 128, 7, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 1, 7)

	branch_pool = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
	branch_pool = conv2d_bn_old(branch_pool, 192, 1, 1)
	x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed4')

	# mixed 5, 6: 17 x 17 x 768
	for i in range(2):
		branch1x1 = conv2d_bn_old(x, 192, 1, 1)

		branch7x7 = conv2d_bn_old(x, 160, 1, 1)
		branch7x7 = conv2d_bn_old(branch7x7, 160, 1, 7)
		branch7x7 = conv2d_bn_old(branch7x7, 192, 7, 1)

		branch7x7dbl = conv2d_bn_old(x, 160, 1, 1)
		branch7x7dbl = conv2d_bn_old(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn_old(branch7x7dbl, 160, 1, 7)
		branch7x7dbl = conv2d_bn_old(branch7x7dbl, 160, 7, 1)
		branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 1, 7)

		branch_pool = MaxPooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn_old(branch_pool, 192, 1, 1)
		x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(5 + i))

	# mixed 7: 17 x 17 x 768
	branch1x1 = conv2d_bn_old(x, 192, 1, 1)

	branch7x7 = conv2d_bn_old(x, 192, 1, 1)
	branch7x7 = conv2d_bn_old(branch7x7, 192, 1, 7)
	branch7x7 = conv2d_bn_old(branch7x7, 192, 7, 1)

	branch7x7dbl = conv2d_bn_old(x, 160, 1, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 1, 7)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 7, 1)
	branch7x7dbl = conv2d_bn_old(branch7x7dbl, 192, 1, 7)

	branch_pool = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(x)
	branch_pool = conv2d_bn_old(branch_pool, 192, 1, 1)
	x = merge([branch1x1, branch7x7, branch7x7dbl, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed7')

	# mixed 8: 8 x 8 x 1280
	branch3x3 = conv2d_bn_old(x, 192, 1, 1)
	branch3x3 = conv2d_bn_old(branch3x3, 320, 3, 3,
						  subsample=(2, 2), border_mode='valid')

	branch7x7x3 = conv2d_bn_old(x, 192, 1, 1)
	branch7x7x3 = conv2d_bn_old(branch7x7x3, 192, 1, 7)
	branch7x7x3 = conv2d_bn_old(branch7x7x3, 192, 7, 1)
	branch7x7x3 = conv2d_bn_old(branch7x7x3, 192, 3, 3,
							subsample=(2, 2), border_mode='valid')

	branch_pool = MaxPooling2D((3, 3), strides=(2, 2))(x)
	x = merge([branch3x3, branch7x7x3, branch_pool],
			  mode='concat', concat_axis=channel_axis,
			  name='mixed8')

	# mixed 9: 8 x 8 x 2048
	for i in range(2):
		branch1x1 = conv2d_bn_old(x, 320, 1, 1)

		branch3x3 = conv2d_bn_old(x, 384, 1, 1)
		branch3x3_1 = conv2d_bn_old(branch3x3, 384, 1, 3)
		branch3x3_2 = conv2d_bn_old(branch3x3, 384, 3, 1)
		branch3x3 = merge([branch3x3_1, branch3x3_2],
						  mode='concat', concat_axis=channel_axis,
						  name='mixed9_' + str(i))

		branch3x3dbl = conv2d_bn_old(x, 448, 1, 1)
		branch3x3dbl = conv2d_bn_old(branch3x3dbl, 384, 3, 3)
		branch3x3dbl_1 = conv2d_bn_old(branch3x3dbl, 384, 1, 3)
		branch3x3dbl_2 = conv2d_bn_old(branch3x3dbl, 384, 3, 1)
		branch3x3dbl = merge([branch3x3dbl_1, branch3x3dbl_2],
							 mode='concat', concat_axis=channel_axis)

		branch_pool = MaxPooling2D(
			(3, 3), strides=(1, 1), border_mode='same')(x)
		branch_pool = conv2d_bn_old(branch_pool, 192, 1, 1)
		x = merge([branch1x1, branch3x3, branch3x3dbl, branch_pool],
				  mode='concat', concat_axis=channel_axis,
				  name='mixed' + str(9 + i))


	# Classification block
	# pool stride was 8
	x = MaxPooling2D((3, 3), strides=(3, 3), name='avg_pool')(x)
	x = Flatten(name='flatten')(x)
	x = Dense(len(args.labels), activation='softmax', name='predictions')(x)

	# Create model
	model = Model(img_input, x)
	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=get_metrics(args.labels))

	model.summary()

	if os.path.exists(architecture):
		model.load_weights(architecture, by_name=True)
		print('Loaded model weights from:', architecture)

	return model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Serialization ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def serialize_model_semantics(args, architecture_hd5):
	'''Save a json file specifying model semantics, I/O contract.

	Arguments
		args.tensor_map: String which indicates tensor map to use (from defines.py) or None
		args.window_size: sites included in the tensor map
		args.read_limit: Maximum reads included in the tensor map
		args.annotations: List of annotations or None
		args.id: the id of the run will be the name of the semantics file
		architecture_hd5: Keras model and weights hd5 file (created with save_model())
	'''
	semantics = {
					'id' : args.id, 
					'output_labels' : args.labels, 
					'architecture' : os.path.basename(architecture_hd5),
					'input_symbols' : args.input_symbols,
				}

	if args.tensor_map:
		semantics['input_tensor_map_name'] = args.tensor_map
		semantics['input_tensor_map'] = defines.get_tensor_channel_map_from_args(args)
		semantics['window_size'] = args.window_size
		semantics['read_limit'] = args.read_limit

	if args.annotation_set and args.annotation_set != '_':
		semantics['input_annotations'] = args.annotations
		semantics['input_annotation_set'] = args.annotation_set

	if args.data_dir:
		semantics['data_dir'] = args.data_dir

	semantics['channels_last'] = args.channels_last

	json_file_name = args.output_dir + args.id + '.json'
	with open(json_file_name, 'w') as outfile:
		json.dump(semantics, outfile)

	print('Saved model semantics at:', json_file_name)


def set_args_and_get_model_from_semantics(args, semantics_json):
	'''Recreate a model from a json file specifying model semantics.

	Update the args namespace from the semantics file values.
	Assert that the serialized tensor map and the recreated one are the same.

	Arguments:
		args.tensor_map: String which indicates tensor map to use (from defines.py) or None
		args.window_size: sites included in the tensor map
		args.read_limit: Maximum reads included in the tensor map
		args.annotations: List of annotations or None
		semantics_json: Semantics json file (created with serialize_model_semantics())

	Returns:
		The Keras model
	'''
	with open(semantics_json, 'r') as infile:
		semantics = json.load(infile)

	if 'input_tensor_map' in semantics:
		args.tensor_map = semantics['input_tensor_map_name']
		args.window_size = semantics['window_size']
		args.read_limit = semantics['read_limit']
		tm = defines.get_tensor_channel_map_from_args(args)
		assert(len(tm) == len(semantics['input_tensor_map']))
		for key in tm:
			assert(tm[key] == semantics['input_tensor_map'][key])

	if 'input_annotations' in semantics:
		args.annotations = semantics['input_annotations']

	if 'channels_last' in semantics:
		args.channels_last = semantics['channels_last']
		if args.channels_last:
			K.set_image_data_format('channels_last')
		else:
			K.set_image_data_format('channels_first')
				
	args.input_symbols = semantics['input_symbols']
	args.labels = semantics['output_labels']

	weight_path_hd5 = os.path.join(os.path.dirname(semantics_json),semantics['architecture'])
	model = load_model(weight_path_hd5, custom_objects=get_metric_dict(args.labels))
	model.summary()
	return model



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Inspections ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def inspect_model(args, model, generate_train, generate_valid, image_path=None):
	'''Collect statistics on model inference and training times.

	Arguments
		args.samples: number of optimization steps to take
		args.batch_size: size of the mini-batches
		model: the model to inspect
		generate_train: training data generator function	
		generate_valid: Validation data generator function

	Returns
		The slightly optimized keras model
	'''
	if image_path:
		plot_dot_model_in_color(model_to_dot(model, show_shapes=True), image_path)

	t0 = time.time()
	history = model.fit_generator(generate_train, steps_per_epoch=args.training_steps, epochs=1, verbose=1, validation_steps=5, validation_data=generate_valid)
	t1 = time.time()
	train_speed = (t1-t0)/(args.batch_size*args.training_steps)
	print('Spent: ', t1-t0, ' seconds training, batch_size:', args.batch_size, 'steps:', args.training_steps, ' Per example training speed:', train_speed)

	t0 = time.time()
	predictions = model.predict_generator(generate_valid, steps=args.training_steps, verbose=1)
	t1 = time.time()
	inference_speed = (t1-t0)/(args.batch_size*args.training_steps)
	print('Spent: ', t1-t0, ' seconds predicting. Per tensor inference speed:', inference_speed)
	
	return model


def plot_dot_model_in_color(dot, image_path):
	for n in dot.get_nodes():
		if n.get_label():
			if 'Conv1' in n.get_label():
				n.set_fillcolor("cyan")
			elif 'Conv2' in n.get_label():
				n.set_fillcolor("deepskyblue1")				
			elif 'BatchNormalization' in n.get_label():
				n.set_fillcolor("goldenrod1")		
			elif 'Activation' in n.get_label():
				n.set_fillcolor("yellow")	
			elif 'MaxPooling' in n.get_label():
				n.set_fillcolor("aquamarine")
			elif 'softmax' in n.get_label():
				n.set_fillcolor("darkolivegreen4")										
			elif 'Dense' in n.get_label():
				n.set_fillcolor("gold")
			elif 'Flatten' in n.get_label():
				n.set_fillcolor("coral3")
			elif 'Input' in n.get_label():
				n.set_fillcolor("darkolivegreen1")
			elif 'Concatenate' in n.get_label():
				n.set_fillcolor("darkorange")
			elif 'Dropout' in n.get_label():
				n.set_fillcolor("tomato")
		n.set_style("filled")
	print('Saving architecture diagram to:',image_path)
	dot.write_png(image_path)


def iterate_neuron(model, layer_dict, neuron, layer_name='conv2d_2'):
	print(layer_name, 'output shape', layer_dict[layer_name].output_shape)
	
	if K.image_data_format() == 'channels_first':
		x = layer_dict[layer_name].output[:,neuron,:,:]
	else:
		x = layer_dict[layer_name].output[:,:,:,neuron]

	loss = K.variable(0.)
	loss_weight_activity = 1.0
	loss_weight_l2 = 0.0

	loss += loss_weight_activity*K.sum(K.square(x))
	
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	loss += loss_weight_l2 * (K.sum(K.square(model.input[0])))
	
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, model.input[0])[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function(model.input+[K.learning_phase()], [loss, grads])
	return iterate


def write_filters_2d(args, model):

	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, in_channels)
	else:
		in_shape = (in_channels, args.read_limit, args.window_size)
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	start_path = '/dsde/data/deep/vqsr/tensors/g94982_ref_read_some_annotations/valid/SNP/recalibrated_g94982_nist_na12878_minimal-SNP-10_100524469.h5'
	with h5py.File(start_path,'r') as hf:
		start_tensor = np.array(hf.get('read_tensor'))

	start_tensor = np.random.random(in_shape)
	annos = np.random.random((1, len(args.annotations)))
	input_tensor = np.zeros((1,)+in_shape)
	for filter_index in range(2, 25, 4):
		exclude = ['read_tensor', 'dropout', 'concatenate', 'dense', 'annotations', 'flatten']
		for layer in model.layers:
			if any([ex in layer.name for ex in exclude]):
				continue
			
			print(" layer name:", layer.name, "filter index:", filter_index)
			iterate = iterate_neuron(model, layer_dict, filter_index, layer.name)
			

			learning_rate = 0.001
			input_tensor[0] = start_tensor
			# run gradient ascent
			for i in range(args.iterations):
				loss_value, grads_value = iterate([input_tensor, annos, 1])
				input_tensor += learning_rate*grads_value
				if i % 2 == 0:
					print("Iteration:", i, "loss:", loss_value, "layer:", layer.name, "index:", filter_index)
			
			tensor_path = './generated_2d/excite_%s/filter_%d.hd5' % (layer.name, filter_index)
			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				hf.create_dataset('read_tensor', data=input_tensor[0])


def iterate_neuron_1d(model, layer_dict, neuron, layer_name='conv1d_2'):
	print(layer_name, 'output shape', layer_dict[layer_name].output_shape)
	
	x = layer_dict[layer_name].output[:,:,neuron]

	loss = K.sum(K.square(x))
	
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, model.input)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([model.input, K.learning_phase()], [loss, grads])
	
	return iterate
		

def write_filters_1d(args, model):
	channel_map = defines.get_tensor_channel_map_from_args(args)	
	in_shape = (1, args.window_size, len(channel_map))
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	start_path = '/dsde/data/deep/vqsr/tensors/g94982_1d_dna_only/valid/SNP/recalibrated_g94982_nist_na12878_minimal-SNP-21_15592343.h5'
	with h5py.File(start_path, 'r') as hf:
		start_tensor = np.array(hf.get('reference'))

	print(start_tensor)
	#start_tensor = np.random.random(in_shape)
	input_tensor = np.zeros(in_shape)
	for filter_index in range(2, 25, 4):
		exclude = ['reference', 'dropout', 'concatenate', 'dense', 'annotations', 'flatten']
		for layer in model.layers:
			if any([ex in layer.name for ex in exclude]):
				continue
			
			print(" layer name:", layer.name, "filter index:", filter_index)
			iterate = iterate_neuron_1d(model, layer_dict, filter_index, layer.name)

			jitter_size = 0.001
			learning_rate = 0.1
			input_tensor[0] = start_tensor
			
			# run gradient ascent
			for i in range(args.iterations):
				random_jitter = jitter_size * (np.random.random(in_shape) - 0.5)
				loss_value, grads_value = iterate([input_tensor, 1])
				input_tensor += random_jitter
				input_tensor += learning_rate*grads_value
				if i % 2 == 0:
					print("Iteration:", i, "loss:", loss_value, "layer:", layer.name, "index:", filter_index)
				input_tensor -= random_jitter
				
			tensor_path = './generated_1d/excite_%s/filter_%d.hd5' % (layer.name, filter_index)
			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				hf.create_dataset('reference', data=input_tensor[0])



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_model_from_generators(args, model, generate_train, generate_valid, save_weight_hd5):
	'''Train an image model for classifying variants.

	Training data lives on disk, it will be loaded by generator functions.
	Plots the metric history after training. Creates a directory to save weights at if necessary.

	Arguments
		args.batch_size: size of the mini-batches
		args.patience: Maximum number of epochs to run without validation loss improvement
		args.epochs: Maximum number of epochs to run regardless of Early Stopping
		args.training_steps: Number of mini-batches in each so-called epoch
		args.validation_steps: Number of validation mini-batches to examine after each epoch.
		model: the model to optimize
		generate_train: training data generator function	
		valid_tuple: Validation data data generator function
		save_weight_hd5: path to save the model weights at

	Returns
		The now optimized keras model
	'''
	if not os.path.exists(os.path.dirname(save_weight_hd5)):
		os.makedirs(os.path.dirname(save_weight_hd5))	
	
	history = model.fit_generator(generate_train, 
		steps_per_epoch=args.training_steps, epochs=args.epochs, verbose=1, 
		validation_steps=args.validation_steps, validation_data=generate_valid,
		callbacks=get_callbacks(args, save_weight_hd5))

	plots.plot_metric_history(history, plots.weight_path_to_title(save_weight_hd5))
	serialize_model_semantics(args, save_weight_hd5)
	print('Model weights saved at: %s' % save_weight_hd5)
	
	return model


def get_callbacks(args, save_weight_hd5):
	callbacks = []
	
	callbacks.append(ModelCheckpoint(filepath=save_weight_hd5, verbose=1, save_best_only=True))
	callbacks.append(EarlyStopping(monitor='val_loss', patience=args.patience*4, verbose=1))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', patience=args.patience, verbose=1))
	
	# if args.channels_last:
	# 	callbacks.append(TensorBoard())
	
	return callbacks
		


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def coeff_determination(y_true, y_pred):
	SS_res =  K.sum(K.square( y_true-y_pred )) 
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
	return ( 1 - SS_res/(SS_tot + K.epsilon()) )


def precision(y_true, y_pred):
	'''Calculates the precision, a metric for multi-label classification of
	how many selected items are relevant.
	'''
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def recall(y_true, y_pred):
	'''Calculates the recall, a metric for multi-label classification of
	how many relevant items are selected.
	'''
	true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def precision_3d(y_true, y_pred):
	'''Calculates the precision, a metric for multi-label classification of
	how many selected items are relevant.
	'''
	true_positives = K.sum(K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0), axis=0)
	predicted_positives = K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0), axis=0)
	precision = true_positives[6] / (predicted_positives[6] + K.epsilon())
	return precision


def recall_3d(y_true, y_pred):
	'''Calculates the recall, a metric for multi-label classification of
	how many relevant items are selected.
	'''
	true_positives = K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0)
	possible_positives = K.sum(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0), axis=0)
	recall = true_positives[1] / (possible_positives[1] + K.epsilon())
	return recall


def per_class_recall(labels):
	recall_fxns = []
	
	for label_key in labels:
		label_idx = labels[label_key]
		string_fxn = 'def '+ label_key + '_recall(y_true, y_pred):\n'
		string_fxn += '\ttrue_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)\n'
		string_fxn += '\tpossible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)\n'
		string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'
		
		exec(string_fxn)
		recall_fxn = eval(label_key + '_recall')
		recall_fxns.append(recall_fxn)
	
	return recall_fxns


def per_class_precision(labels):
	precision_fxns = []
	
	for label_key in labels:
		label_idx = labels[label_key]
		string_fxn = 'def '+ label_key + '_precision(y_true, y_pred):\n'
		string_fxn += '\ttrue_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)\n'
		string_fxn += '\tpredicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)\n'
		string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'
		
		exec(string_fxn)
		precision_fxn = eval(label_key + '_precision')
		precision_fxns.append(precision_fxn)
	
	return precision_fxns


def per_class_recall_3d(labels):
	recall_fxns = []
	
	for label_key in labels:
		label_idx = labels[label_key]
		string_fxn = 'def '+ label_key + '_recall(y_true, y_pred):\n'
		string_fxn += '\ttrue_positives = K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0)\n'
		string_fxn += '\tpossible_positives = K.sum(K.sum(K.round(K.clip(y_true, 0, 1)), axis=0), axis=0)\n'
		string_fxn += '\treturn true_positives['+str(label_idx)+'] / (possible_positives['+str(label_idx)+'] + K.epsilon())\n'
		
		exec(string_fxn)
		recall_fxn = eval(label_key + '_recall')
		recall_fxns.append(recall_fxn)
	
	return recall_fxns


def per_class_precision_3d(labels):
	precision_fxns = []
	
	for label_key in labels:
		label_idx = labels[label_key]
		string_fxn = 'def '+ label_key + '_precision(y_true, y_pred):\n'
		string_fxn += '\ttrue_positives = K.sum(K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0), axis=0)\n'
		string_fxn += '\tpredicted_positives = K.sum(K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0), axis=0)\n'
		string_fxn += '\treturn true_positives['+str(label_idx)+'] / (predicted_positives['+str(label_idx)+'] + K.epsilon())\n'
		
		exec(string_fxn)
		precision_fxn = eval(label_key + '_precision')
		precision_fxns.append(precision_fxn)
	
	return precision_fxns


def get_metrics(classes=None, dim=2):
	if classes and dim == 2:
		return [metrics.categorical_accuracy] + per_class_precision(classes) + per_class_recall(classes)
	elif classes and dim == 3:
		return [metrics.categorical_accuracy] + per_class_precision_3d(classes) + per_class_recall_3d(classes)
	else:
		return [metrics.categorical_accuracy, precision, recall]


def get_metric_dict(labels):
	metrics = {'precision':precision, 'recall':recall}
	precision_fxns = per_class_precision(labels)
	recall_fxns = per_class_recall(labels)
	for i,label_key in enumerate(labels.keys()):
		metrics[label_key+'_precision'] = precision_fxns[i]
		metrics[label_key+'_recall'] = recall_fxns[i]
	return metrics

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Evaluation ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def eval_model(model, test_tuple):
	tresults = model.evaluate(test_tuple[0], test_tuple[1])
	print('Test results:', tresults)


def predictions_to_snp_scores(args, predictions, positions):
	eps = 1e-7	
	snp = predictions[:, args.labels['SNP']]
	not_snp = predictions[:, args.labels['NOT_SNP']]
	snp_scores = np.log(eps + snp / (not_snp + eps))
	return dict(zip(positions, snp_scores))


def predictions_to_indel_scores(args, predictions, positions):
	eps = 1e-7
	indel = predictions[:, args.labels['INDEL']]
	not_indel = predictions[:, args.labels['NOT_INDEL']]
	indel_scores = np.log(eps + indel / (not_indel + eps))
	return dict(zip(positions, indel_scores))


def predictions_to_snp_indel_scores(args, predictions, positions):
	snp_dict = predictions_to_snp_scores(args, predictions, positions)
	indel_dict = predictions_to_indel_scores(args, predictions, positions)
	return snp_dict, indel_dict


def weighted_categorical_crossentropy(weights):
	"""
	A weighted version of keras.objectives.categorical_crossentropy

	Variables:
	weights: numpy array of shape (C,) where C is the number of classes

	Usage:
	weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
	loss = weighted_categorical_crossentropy(weights)
	model.compile(loss=loss,optimizer='adam')
	"""

	weights = K.variable(weights)

	def loss(y_true, y_pred):
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		# calc
		loss = y_true * K.log(y_pred) * weights
		loss = -K.sum(loss, -1)
		return loss

	return loss




