#!/usr/bin/python
# keras_example.py


# Basic example for learning small keras model's on simulated data and MNIST
# sam@broadinstitute.org
# December 2016


# Try to be Python 2 / 3 Compatible
from __future__ import print_function

# Import:
import os
import sys
import gzip
import pickle
import random
import numpy as np
from keras import metrics
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense


def run():
	#linear_regression()
	logistic_regression()
	#multilayer_perceptron()


def linear_regression():
	samples = 40
	real_weight = 2.0
	real_bias = 0.5
	x = np.linspace(-1, 1, samples)
	y = real_weight*x + real_bias + (np.random.randn(*x.shape) * 0.1)

	linear_model = Sequential()
	linear_model.add(Dense(1, input_dim=1))
	linear_model.compile(loss='mse', optimizer='sgd')
	linear_model.fit(x, y, batch_size=1, epochs=10)

	learned_slope = linear_model.get_weights()[0][0][0]
	learned_bias = linear_model.get_weights()[1][0]
	print('Learned slope:',  learned_slope, 'real slope:', real_weight, 'learned bias:', learned_bias, 'real bias:', real_bias)
	plt.plot(x, y)
	plt.plot([-1,1], [-learned_slope+learned_bias, learned_slope+learned_bias], 'r')
	plt.show()


def logistic_regression():
	train, test, valid = load_data('mnist.pkl.gz')

	epochs = 3200
	num_labels = 10
	train_y = make_one_hot(train[1], num_labels)
	valid_y = make_one_hot(valid[1], num_labels)
	test_y = make_one_hot(test[1], num_labels)

	logistic_model = Sequential()
	logistic_model.add(Dense(10, activation='softmax', input_dim=784, name='mnist_templates'))
	logistic_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	logistic_model.summary()
	templates = logistic_model.layers[0].get_weights()[0]
	plot_templates(templates, 0)
	print('weights shape:', templates.shape)

	for e in range(epochs):
		trainidx = random.sample(range(0, train[0].shape[0]), 8192)
		x_batch = train[0][trainidx,:]
		y_batch = train_y[trainidx]
		logistic_model.train_on_batch(x_batch, y_batch)
		if e % 5 == 0:
			plot_templates(logistic_model.layers[0].get_weights()[0], e)

	print('Test set loss and accuracy:', logistic_model.evaluate(test[0], test_y))


def plot_templates(templates, epoch):
	n = 10
	templates = templates.reshape((28,28,n))
	plt.figure(figsize=(16, 8))
	for i in range(n):
		ax = plt.subplot(2, 5, i+1)		
		plt.imshow(templates[:, :, i])
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

	plot_name = "./frames/mnist/regression/templates_"+str(epoch)+".png"
	if not os.path.exists(os.path.dirname(plot_name)):
		os.makedirs(os.path.dirname(plot_name))		
	plt.savefig(plot_name)


def multilayer_perceptron():
	train, test, valid = load_data('mnist.pkl.gz')

	num_labels = 10
	train_y = make_one_hot(train[1], num_labels)
	valid_y = make_one_hot(valid[1], num_labels)
	test_y = make_one_hot(test[1], num_labels)

	mlp_model = Sequential()
	mlp_model.add(Dense(300, activation='relu', input_dim=784))
	mlp_model.add(Dense(10, activation='softmax'))
	mlp_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
	mlp_model.fit(train[0], train_y, validation_data=(valid[0],valid_y), batch_size=32, epochs=10)
	print('Test set loss and accuracy:', mlp_model.evaluate(test[0], test_y))


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
		import urllib
		origin = (
			'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
		)
		print('Downloading data from %s' % origin)
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