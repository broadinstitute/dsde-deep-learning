#!/usr/bin/env python
# activity.py
#
# Activity functions with keras.py
# DSDE Deep Learning
# June 2018
# Sam Friedman 
# sam@broadinstitute.org

from __future__ import print_function

import os
import sys
import math
import h5py
import gzip
import pickle
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
from scipy import interp
from keras import metrics
import keras.backend as K
from random import shuffle
from Bio import Seq, SeqIO
from itertools import cycle
import matplotlib.pyplot as plt
from collections import Counter

from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers import Input, Dense, Dropout, BatchNormalization, SpatialDropout1D, Activation, Flatten

data_path = '/dsde/data/deep/'
reference_fasta = data_path + 'Homo_sapiens_assembly19.fasta'
chrom_hmm_bed_file = data_path + 'wgEncodeAwgSegmentationCombinedGm12878.bed'
breakpoint_bed_file =  '/dsde/data/deep/vqsr/beds/icgc_bkp_sorted.bed'

amiguity_codes = {'K':[0,0,0.5,0.5], 'M':[0.5,0.5,0,0], 'R':[0.5,0,0,0.5], 'Y':[0,0.5,0.5,0], 'S':[0,0.5,0,0.5], 
				  'W':[0.5,0,0.5,0], 'B':[0,0.333,0.333,0.334], 'V':[0.333,0.333,0,0.334],'H':[0.333,0.333,0.334,0],
				  'D':[0.333,0,0.333,0.334], 'N':[0.25,0.25,0.25,0.25]}

label_sets = {
	'chrom_hmm'  : {'TSS':0, 'PF':1, 'E':2, 'WE':3, 'CTCF':4, 'T':5, 'R':6 },
	'breakpoint' : {'WT':0, 'BREAKPOINT':1}
}


def run():
	args = parse_args()
	multilayer_perceptron_on_mnist(args)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='bp')
	parser.add_argument('--activation', default='relu', 
		help="The neuronal activation type (keras string like 'relu' of make your own...")
	parser.add_argument('--window_size', default=64, type=int)
	parser.add_argument('--samples', default=1000, type=int)
	parser.add_argument('--activity', default=0, type=int)
	parser.add_argument('--reference_fasta', default=reference_fasta)
	parser.add_argument('--bed_file',default=breakpoint_bed_file)	
	parser.add_argument('--inputs', default={'A':0, 'C':1, 'T':2, 'G':3})
	parser.add_argument('--epochs', default=15, type=int, help='Training epochs.')
	parser.add_argument('--batch_size', default=32, type=int, help='Training mini batch size.')
	parser.add_argument('--label_set', default='breakpoint', choices=label_sets.keys())
	parser.add_argument('--labels', default={})
	parser.add_argument('--conv_width', default=5, type=int, help='Width of 1D convolutional kernels.')
	parser.add_argument('--conv_dropout', default=0.0, type=float, 
		help='Dropout rate in convolutional layers.')
	parser.add_argument('--conv_batch_normalize', default=False, action='store_true',
		help='Batch normalize convolutional layers.')
	parser.add_argument('--conv_layers', nargs='+', default=[128, 96, 64, 48], type=int,
		help='List of sizes for each convolutional filter layer')
	parser.add_argument('--kernel_initializer', default='he_normal', 
		help='convolution kernel initializer.')
	parser.add_argument('--same_padding', default=False, action='store_true',
		help='Valid or same border padding on the convolutional layers.')	
	parser.add_argument('--spatial_dropout', default=False, action='store_true',
		help='Spatial dropout on the convolutional layers.')	
	parser.add_argument('--max_pools', nargs='+', default=[], type=int,
		help='List of maxpooling layers.')	
	parser.add_argument('--fc_layers', nargs='+', default=[32], type=int,
		help='List of sizes for each fully connected layer')
	parser.add_argument('--fc_dropout', default=0.0, type=float, 
		help='Dropout rate in fully connected  layers.')
	parser.add_argument('--fc_batch_normalize', default=False, action='store_true',
		help='Batch normalize fully connected layers.')
	parser.add_argument('--fc_initializer', default='he_normal', 
		help='fully connected layer initializer')

	args = parser.parse_args()
	args.labels = label_sets[args.label_set]
	print('Arguments are', args)
	return args


def activity_1d_model_from_args(args):
	'''Build Reference 1d CNN model for classification.

	Architecture specified by parameters.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.labels: The output labels (e.g. BREAKPOINT, WT)

	Returns
		The keras model
	'''	
	concat_axis = -1	
	x = reference = Input(shape=(args.window_size, len(args.inputs)), name='reference')

	max_pool_diff = len(args.conv_layers)-len(args.max_pools)	
	for  i,c in enumerate(args.conv_layers):

		x = Conv1D(filters=c, kernel_size=args.conv_width, activation='linear', padding=args.padding, kernel_initializer=args.kernel_initializer)(x)
		if args.conv_batch_normalize:
			x = BatchNormalization(axis=concat_axis)(x)
		x = Activation(args.activation)(x)

		if args.conv_dropout > 0 and args.spatial_dropout:
			x = SpatialDropout1D(args.conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(args.conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling1D(args.max_pools[i-max_pool_diff])(x)

	x = Flatten()(x)

	for fc in args.fc_layers:
		if args.fc_batch_normalize:
			x = Dense(units=fc, activation='linear', kernel_initializer=args.fc_initializer)(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)			
		else:
			x = Dense(units=fc, activation=args.activation)(x)
		
		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)

	prob_output = Dense(units=len(args.labels), activation='softmax')(x)
	
	model = Model(inputs=[reference], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	my_metrics = [metrics.binary_accuracy]

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=my_metrics)
	model.summary()
	
	return model


def multilayer_perceptron_on_mnist(args):
	train, test, valid = load_data('mnist.pkl.gz')

	num_labels = 10
	train_y = make_one_hot(train[1], num_labels)
	valid_y = make_one_hot(valid[1], num_labels)
	test_y = make_one_hot(test[1], num_labels)

	mlp_model = Sequential()
	for fc in args.fc_layers:
		mlp_model.add(Dense(fc, activation=args.activation, input_dim=784))
		
	mlp_model.add(Dense(num_labels, activation='softmax'))
	mlp_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	mlp_model.summary()

	print('Train multilayer perceptron on MNIST. Test set loss and accuracy:', mlp_model.evaluate(test[0], test_y))
	mlp_model.fit(train[0], train_y, validation_data=(valid[0],valid_y), batch_size=32, epochs=args.epochs)
	print('Multilayer Perceptron trained. Test set loss and accuracy:', mlp_model.evaluate(test[0], test_y))


def make_one_hot(y, num_labels):
	ohy = np.zeros((len(y), num_labels))
	for i in range(0, len(y)):
		ohy[i][y[i]] = 1.0
	return ohy


def load_data(dataset):
	''' Loads the dataset
	:type dataset: string
	:param dataset: the path to the dataset (here MNIST)
	'''

	#############
	# LOAD DATA #
	#############

	# Download the MNIST dataset if it is not present
	data_dir, data_file = os.path.split(dataset)
	if data_dir == "" and not os.path.isfile(dataset):
		# Check if dataset is in the data directory.
		new_path = os.path.join(
			os.path.split(__file__)[0],
			"data",
			dataset
		)
		if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
			dataset = new_path

	if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
		origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
		print('Downloading data from %s' % origin)
		if not os.path.exists(os.path.dirname(dataset)):
			os.makedirs(os.path.dirname(dataset))	
		if (sys.version_info > (3, 0)):
			from urllib.request import urlretrieve
			urlretrieve(origin, dataset)
		else:
			import urllib
			urllib.urlretrieve(origin, dataset)

	print('loading data...')

	# Load the dataset
	f = gzip.open(dataset, 'rb')
	if sys.version_info[0] == 3:
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		train_set, valid_set, test_set = u.load()
	else:
		train_set, valid_set, test_set = pickle.load(f)

	f.close()
	#train_set, valid_set, test_set format: tuple(input, target)
	#input is an numpy.ndarray of 2 dimensions (a matrix)
	#which row's correspond to an example. target is a
	#numpy.ndarray of 1 dimensions (vector)) that have the same length as
	#the number of rows in the input. It should give the target
	#target to the example with the same index in the input.

	return train_set, valid_set, test_set


if '__main__'==__name__:
	run() 
