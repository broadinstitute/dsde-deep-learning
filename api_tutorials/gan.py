#gan.py
#sam@broadinstitute.org

# Keras GAN Implementation
# See: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import cv2
import time
import random
import argparse
import numpy as np
import pickle, random, sys, keras

import matplotlib
matplotlib.use('Agg') # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt # First import matplotlib, then use Agg, then import plt

import keras.models as models
from keras import backend as K
from keras.models import Model, save_model, load_model
from keras.optimizers import *
from keras.activations import *
from keras.regularizers import *
from keras.utils import np_utils
from keras.utils import generic_utils
from keras.layers import Input,merge
from keras.layers.recurrent import LSTM
from keras.layers.normalization import *
from keras.datasets import mnist, cifar10
from keras.layers.noise import GaussianNoise
from keras.layers.wrappers import TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import Reshape, Dense, Dropout, Activation, Flatten, SpatialDropout2D, ActivityRegularization
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, AveragePooling2D


def run():
	'''Parse arguments, create a model and dispatch on mode'''
	args = parse_args()
	if 'imagenet' == args.mode:
		gan_on_imagenet(args)
	elif 'cifar' == args.mode:
		gan_on_cifar(args)
	elif 'mnist' == args.mode:
		gan_on_mnist(args)
	elif 'faces' == args.mode:
		gan_on_imagenet(args)
	else:
		raise ValueError('Unknown adversarial mode:', args.mode)


def parse_args():
	parser = argparse.ArgumentParser()

	# Required mode argument: What type of Adversary are you?
	parser.add_argument('mode')

	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--epochs', default=5000, type=int)
	parser.add_argument('--save_path', default='./frames/gan/')
	parser.add_argument('--fps', default=12, type=int)
	parser.add_argument('--plot_examples', default=16, type=int)
	parser.add_argument('--seeds', default=200, type=int)
	parser.add_argument('--dropout', default=0.2, type=float)
	parser.add_argument('--in_shape', default=(28,28,1))
	parser.add_argument('--pretrain', default=0, type=int)
	parser.add_argument('--l2', default=0.0, type=float)
	parser.add_argument('--l1', default=0.0, type=float)
	parser.add_argument('--activity_weight', default=1.0, type=float)
	parser.add_argument('--total_variation', default=1e-5, type=float)
	parser.add_argument('--continuity_loss', default=0, type=float)

	parser.add_argument('-bn',  '--batch_normalize', default=False, action='store_true')
	parser.add_argument('-gl',  '--generator_loops', default=5, type=int)
	parser.add_argument('-glr', '--generator_learning_rate', default=1e-4, type=float)	
	parser.add_argument('-dl',  '--discriminator_loops', default=6, type=int)
	parser.add_argument('-dlr', '--discriminator_learning_rate', default=1e-4, type=float)
	parser.add_argument('-lrd', '--learning_rate_decay', default=0, type=int)


	args = parser.parse_args()
	print('Arguments are', args)	
	return args


def gan_on_imagenet(args):
	args.in_shape = (256,256,3)
	generator = build_imagenet_generative_model(args)
	discriminator = build_imagenet_discriminative(args)
	gan = build_stacked_gan_imagenet(args, generator, discriminator)	
	
	train_imagenet_gan(args, generator, discriminator, gan)	


def gan_on_cifar(args):
	args.in_shape = (32,32, 3)
	generator = build_generative_model(args)
	discriminator = build_discriminative(args)
	gan = build_stacked_gan(args, generator, discriminator)	

	# gan = load_model('cifar10_gan.hd5')
	# generator = load_model('cifar10_generator.hd5') 
	# discriminator = load_model('cifar10_discriminator.hd5')
	train_cifar_gan(args, generator, discriminator, gan)


def gan_on_mnist(args):
	args.in_shape = (28,28, 1)

	generator = build_mnist_generative_model(args)
	discriminator = build_mnist_discriminative(args)
	gan = build_stacked_gan(args, generator, discriminator)

	# gan = load_model('mnist_gan.hd5')
	# generator = load_model('mnist_generator.hd5') 
	# discriminator = load_model('mnist_discriminator.hd5')

	train_mnist_gan(args, generator, discriminator, gan)	


def load_mnist():
	img_rows, img_cols = 28, 28

	# the data, shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('bounds:', np.min(x_train), np.max(x_train))

	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train, y_train), (x_test, y_test)


def load_cifar():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')	
	return (x_train, y_train), (x_test, y_test)


def load_imagenet(num_labels=24,shape=(128,128)):

	imagenet_path = '/home/sam/big_data/imagenet/ILSVRC2014_DET_train/'
	train_paths = [ imagenet_path + tp for tp in sorted(os.listdir(imagenet_path)) if os.path.isdir(imagenet_path + tp)  ]
	(x_train, y_train), (x_test, y_test) = load_images_from_class_dirs(train_paths, num_labels, shape=shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	#x_train /= 255
	#x_test /= 255

	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train, y_train), (x_test, y_test)


def load_faces(num_labels=14000, shape=(128,128)):
	imagenet_path = '/home/sam/big_data/faces/lfw/'
	train_paths = [ imagenet_path + tp for tp in sorted(os.listdir(imagenet_path)) if os.path.isdir(imagenet_path + tp)  ]
	(x_train, y_train), (x_test, y_test) = load_images_from_class_dirs(train_paths, num_labels, shape=shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	#x_train /= 255
	#x_test /= 255

	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train, y_train), (x_test, y_test)


def load_images_from_class_dirs(train_paths, num_labels, shape=(224,224), per_class_max=5000):
	count = 0
	train_set = []
	t_labels = []
	valid_set = []
	v_labels = []

	image_exts =['.png', '.jpeg', '.jpg', 'tif']
	print("Got :", len(train_paths), "train paths. will use:", num_labels)
	
	for label, tp in enumerate(train_paths):
		if count == num_labels:
			print('Got more image dirs than labels. bailing out with:', count)
			break
		imgs = os.listdir(tp)
		count += 1
		this_t = 0
		for im in imgs:		
			fn, file_extension = os.path.splitext(im)
			if file_extension.lower() == '.gif':
				print("Got a gif gonna skip it.", fn, "ext:", file_extension)
				continue

			if file_extension.lower() not in image_exts:
				continue

			y_vector = np.zeros(num_labels) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			
			img = image_as_matrix(tp+'/'+im, shape=shape)

			this_t += 1
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			if (random.random() > 0.1):
				train_set.append(img)
				t_labels.append(y_vector)
			else:
				valid_set.append(img)
				v_labels.append(y_vector)
				
	return (np.asarray(train_set), np.asarray(t_labels)), (np.asarray(valid_set), np.asarray(v_labels))


def image_as_matrix(image_path, expand_dims=False, shape=(224,224)):
	img = cv2.resize(cv2.imread(image_path), shape).astype(np.float32)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	img /= 255.0
	#img -= 0.5

	if K.image_data_format() == 'channels_first':
		img = img.transpose((2,0,1))
	return img


def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val

def wasserstein(y_true, y_pred):
	return K.mean(y_true * y_pred)

def binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)

def categorical_crossentropy(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true)

def negative_categorical_crossentropy(y_true, y_pred):
    return -1*K.categorical_crossentropy(y_pred, y_true)

def generator_loss(args, pre_logit):
	"""Get a custom loss function for a generative model

	Variables:
		args: command line arguments object
		pre_logit: the generated image before going through the final sigmoid
	"""
	def loss(y_true, y_pred):
		# scale predictions so that the class probas of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		# calc
		loss = y_true * K.log(y_pred)
		loss = -K.sum(loss, -1)

		loss -= args.total_variation * total_variation_norm(pre_logit)
		loss -= args.continuity_loss * continuity_loss(pre_logit)

		loss -= args.l2 * K.sum(K.square(pre_logit)) / np.prod(args.in_shape)

		return loss

	return loss


##############################
###### Loss Regularizers #####
##############################

# continuity loss util function
def continuity_loss(x):
	assert K.ndim(x) == 4
	if K.image_data_format()== 'channels_first':
		a = K.square(x[:, :, :-2, :-2] -
					 x[:, :, 2:, :-2])
		b = K.square(x[:, :, :-2, :-2] -
					 x[:, :, :-2, 2:])
	else:
		a = K.square(x[:, :-2, :-2, :] -
					 x[:, 2: , :-2, :])
		b = K.square(x[:, :-2, :-2, :] -
					 x[:, :-2, 2: , :])
	return K.sum(K.pow(a + b, 1.25))


def alpha_norm(x, alpha=6, lambdaa=0.05):
	x -= K.mean(x)
	return lambdaa * K.pow(K.sum(x), alpha)


def total_variation_norm(x):
	x -= K.mean(x)

	if K.image_data_format()== 'channels_first':
		a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
		b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
	else:
		a = K.square(x[:, 1:, :-1, :] - x[:, :-1, :-1, :])
		b = K.square(x[:, :-1, 1:, :] - x[:, :-1, :-1, :])

	tv = K.sum(K.pow(a + b, 1.25))

	return tv

def build_imagenet_generative_model(args):
	# Build Generative model ...
	nch = 50
	dense_channels = 32
	inner_dim = 16
	channel_axis = -1
	g_input = Input(shape=[args.seeds])
	H = Dense(dense_channels*inner_dim*inner_dim, kernel_initializer='glorot_normal')(g_input)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Activation('relu')(H)
	H = Reshape( [inner_dim, inner_dim, dense_channels] )(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Activation('relu')(H)	
	pre_logit = Conv2D(3, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
	generation = Activation('sigmoid')(pre_logit)
	generator = Model(g_input, generation)
	opt = RMSprop(lr=args.generator_learning_rate)		

	gloss = generator_loss(args, pre_logit)	
	generator.compile(loss=gloss, optimizer=opt)
	generator.summary()
	return generator


def batch_normalize_or_not(args, x, channel_axis):
	if args.batch_normalize:
		return BatchNormalization(scale=False, axis=channel_axis)(x)
	return x


def build_imagenet_discriminative(args):
	# Build Discriminative model ...
	channel_axis = -1
	d_input = Input(shape=args.in_shape)
	H = Conv2D(216, (5, 5), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform')(d_input)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Dropout(args.dropout)(H)
	H = Activation('relu')(H)
	H = Conv2D(256,  (3, 3), strides=(2, 2), padding='valid', kernel_initializer='glorot_uniform')(d_input)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Dropout(args.dropout)(H)
	H = Activation('relu')(H)
	H = Conv2D(128,  (3, 3), strides=(2, 2), padding='valid', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Dropout(args.dropout)(H)
	H = Activation('relu')(H)
	H = Conv2D(64,  (3, 3), strides=(2, 2),  padding='valid', kernel_initializer='glorot_uniform')(H)
	H = batch_normalize_or_not(args, H, channel_axis)
	H = Dropout(args.dropout)(H)
	H = Activation('relu')(H)
	H = Flatten()(H)
	H = Dense(44)(H)
	H = Dropout(args.dropout)(H)
	probability_out = Dense(2, activation='softmax')(H)
	discriminator = Model(d_input, probability_out)
	dopt = RMSprop(lr=args.discriminator_learning_rate)	
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_stacked_gan_imagenet(args, generator, discriminator):
	# Build stacked GAN model
	gan_input = Input(shape=[args.seeds])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)
	opt = RMSprop(lr=args.generator_learning_rate)	
	GAN.compile(loss='binary_crossentropy', optimizer=opt)
	GAN.summary()
	return GAN


def build_generative_model(args):
	# Build Generative model ...
	nch = 44
	inner_dim = 8
	bn_axis = -1
	g_input = Input(shape=[args.seeds])
	H = Dense(nch*inner_dim*inner_dim, kernel_initializer='glorot_normal')(g_input)
	#H = BatchNormalization(axis=bn_axis, scale=False,)(H)
	H = Activation('relu')(H)
	if K.image_data_format() == 'channels_first':
		H = Reshape( [nch, inner_dim, inner_dim] )(H)
	else:
		H = Reshape( [inner_dim, inner_dim, nch] )(H)
	
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*6, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	#H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*5, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	#H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = Conv2D(nch*6, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = Dropout(0.2)(H)
	#H = BatchNormalization(axis=bn_axis, scale=False)(H)

	H = Activation('relu')(H)
	H = Conv2D(3, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = Activation('sigmoid')(H)

	generator = Model(g_input, H)
	opt = RMSprop(lr=args.generator_learning_rate) #RMSprop(lr=1e-5)
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
	return generator


def build_discriminative(args):
	# Build Discriminative model ...
	d_input = Input(shape=args.in_shape)
	H = Conv2D(316, (5, 5), strides=(2,2), padding='same')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Conv2D(216, (3, 3), strides=(2,2), padding='same')(H)
	H = Dropout(args.dropout)(H)
	H = LeakyReLU(0.2)(H)
	H = Conv2D(128, (3, 3), padding='same')(H)
	H = Dropout(args.dropout)(H)
	H = LeakyReLU(0.2)(H)
	H = Flatten()(H)
	H = Dense(108)(H)
	H = Dropout(args.dropout)(H)
	H = LeakyReLU(0.2)(H)
	d_V = Dense(2, activation='softmax')(H)
	discriminator = Model(d_input, d_V)
	dopt = RMSprop(lr=args.discriminator_learning_rate) #RMSprop(lr=1e-5)	
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_mnist_generative_model(args):
	# Build Generative model ...
	nch = 24
	bn_axis = -1
	g_input = Input(shape=[args.seeds])
	H = Dense(nch*7*7, kernel_initializer='glorot_normal')(g_input)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = Reshape( [7, 7, nch] )(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*5, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = Conv2D(nch*6, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)	
	H = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
	g_V = Activation('sigmoid')(H)
	generator = Model(g_input,g_V)
	opt = RMSprop(lr=args.generator_learning_rate)	
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
	return generator


def build_mnist_discriminative(args):
	# Build Discriminative model ...
	d_input = Input(shape=args.in_shape)
	H = Conv2D(128, (5, 5), strides=(2, 2), padding='same', activation='relu')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Dropout(args.dropout)(H)
	H = Conv2D(108, (3, 3), strides=(2, 2), padding='same', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(args.dropout)(H)
	H = Flatten()(H)
	H = Dense(64)(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(args.dropout)(H)
	d_V = Dense(2, activation='softmax')(H)
	discriminator = Model(d_input,d_V)
	dopt = RMSprop(lr=args.discriminator_learning_rate)	
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_stacked_gan(args, generator, discriminator):
	# Build stacked GAN model
	gan_input = Input(shape=[args.seeds])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)
	#opt = RMSprop(lr=2e-5)	
	opt = RMSprop(lr=args.generator_learning_rate)		
	GAN.compile(loss='binary_crossentropy', optimizer=opt)
	GAN.summary()
	return GAN


def train_imagenet_gan(args, generator, discriminator, gan):
	imagenet_data = load_faces(shape=(args.in_shape[0], args.in_shape[1]))
	(x_train, y_train), (x_test, y_test) = imagenet_data
	#plot_color(x_train)

	if args.pretrain > 0:	# Pre-train the discriminator network ...
		ntrain = min(args.pretrain, x_train.shape[0])-1
		trainidx = random.sample(range(0, x_train.shape[0]), ntrain)
		xt = x_train[trainidx,:,:,:]	
		noise_gen = np.random.uniform(0,1,size=[xt.shape[0], args.seeds])
		generated_images = generator.predict(noise_gen, batch_size=32, verbose=1)
		x = np.concatenate((xt, generated_images))
		n = xt.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1
		print('np sum 1s:', np.sum(y[:,0]), 'x shape:', x.shape)
		discriminator.fit(x,y, epochs=1, batch_size=32, validation_split=0.1, shuffle=True)
		y_hat = discriminator.predict(x, batch_size=32, verbose=1)

		# Measure accuracy of pre-trained discriminator network
		y_hat_idx = np.argmax(y_hat,axis=1)
		y_idx = np.argmax(y,axis=1)
		diff = y_idx-y_hat_idx
		n_tot = y.shape[0]
		n_rig = (diff==0).sum()
		acc = n_rig*100.0/n_tot
		print("Pretraining Accuracy: %0.02f pct (%d of %d) right" % (acc, n_rig, n_tot))

	# Plot some generated images from our GAN before training
	# plot_gen_color(generator, 25,(5,5),(34,34))
	train_for_n(args, imagenet_data, generator, discriminator, gan)
	save_model(gan, 'face3_gan.hd5')
	save_model(generator, 'face3_generator.hd5') 
	save_model(discriminator, 'face3_discriminator.hd5')
	# Plot some generated images from our GAN after training
	plot_gen_color(generator, 25, (5,5), (34,34))


def train_cifar_gan(args, generator, discriminator, gan):
	cifar_data = load_cifar()
	(x_train, y_train), (x_test, y_test) = cifar_data

	if args.pretrain > 0: # Pre-train the discriminator network ...
		trainidx = random.sample(range(0, x_train.shape[0]), args.pretrain)
		xt = x_train[trainidx,:,:,:]
		noise_gen = np.random.uniform(0, 1, size=[xt.shape[0], args.seeds])
		generated_images = generator.predict(noise_gen)
		x = np.concatenate((xt, generated_images))
		n = xt.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1
		print('np sum 1s:', np.sum(y[:,0]), 'x shape:', x.shape)
		discriminator.fit(x,y, epochs=1, batch_size=32, validation_split=0.1, shuffle=True)
		y_hat = discriminator.predict(x)

		# Measure accuracy of pre-trained discriminator network
		y_hat_idx = np.argmax(y_hat,axis=1)
		y_idx = np.argmax(y,axis=1)
		diff = y_idx-y_hat_idx
		n_tot = y.shape[0]
		n_rig = (diff==0).sum()
		acc = n_rig*100.0/n_tot
		print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
		
	# Train for 6000 epochs at original learning rates
	train_for_n(args, cifar_data, generator, discriminator, gan)
	save_model(gan, 'cifar10_gan2.hd5')
	save_model(generator, 'cifar10_generator2.hd5') 
	save_model(discriminator, 'cifar10_discriminator2.hd5')
	# Plot some generated images from our GAN
	plot_gen_color(generator, 25,(5,5),(12,12))


def train_mnist_gan(args, generator, discriminator, gan):
	mnist_data = load_mnist()
	(x_train, y_train), (x_test, y_test) = mnist_data

	if args.pretrain:	# Pre-train the discriminator network ...
		trainidx = random.sample(range(0,x_train.shape[0]), args.pretrain)
		XT = x_train[trainidx,:,:,:]	
		noise_gen = np.random.uniform(0,1,size=[XT.shape[0], args.seeds])
		generated_images = generator.predict(noise_gen, batch_size=128, verbose=1)
		X = np.concatenate((XT, generated_images))
		n = XT.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1

		discriminator.fit(X,y, epochs=1, batch_size=128, validation_split=0.1, shuffle=True)
		y_hat = discriminator.predict(X, batch_size=128, verbose=1)

		# Measure accuracy of pre-trained discriminator network
		y_hat_idx = np.argmax(y_hat,axis=1)
		y_idx = np.argmax(y,axis=1)
		diff = y_idx-y_hat_idx
		n_tot = y.shape[0]
		n_rig = (diff==0).sum()
		acc = n_rig*100.0/n_tot
		print("\nAccuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
			
	train_for_n(args, mnist_data, generator, discriminator, gan)
	# Plot some generated images from our GAN
	plot_gen(generator, 25,(5,5),(12,12))
	# Plot real MNIST images for comparison
	save_model(gan, 'mnist_gan.hd5')
	save_model(generator, 'mnist_generator.hd5') 
	save_model(discriminator, 'mnist_discriminator.hd5')
	plot_real(x_train)


def train_for_n(args, data, generator, discriminator, gan):
	# set up loss storage vector
	losses = {"d":[], "g":[]}
	(x_train, y_train), (x_test, y_test) = data
	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	samples_seeds = np.random.uniform(0,1,size=[args.plot_examples, args.seeds])
	opt = RMSprop(lr=args.discriminator_learning_rate)
	for e in range(args.epochs):
		make_trainable(discriminator, False)
		make_trainable(generator, True)
		gan.compile(loss='binary_crossentropy', optimizer=opt)
		for _ in range(args.generator_loops):	
			# train Generator-Discriminator stack on input noise to non-generated output class
			noise_tr = np.random.uniform(0, 1, size=[args.batch_size, args.seeds])
			y2 = np.zeros([args.batch_size,2])
			# Tell the model that random is correct
			y2[:, 0] = 1
			g_loss = gan.train_on_batch(noise_tr, y2)
			losses["g"].append(g_loss)

		make_trainable(discriminator, True)
		make_trainable(generator, False)
		gan.compile(loss='binary_crossentropy', optimizer=opt)
		for _ in range(args.discriminator_loops):
			# Make generative images
			image_batch = x_train[np.random.randint(0,x_train.shape[0], size=args.batch_size),:,:,:]    
			noise_gen = np.random.uniform(0,1,size=[args.batch_size, args.seeds])
			generated_images = generator.predict(noise_gen)
			
			# Train discriminator on generated images
			X = np.concatenate((image_batch, generated_images))
			y = np.zeros([2*args.batch_size,2])
			y[:args.batch_size, 0] = 1
			y[args.batch_size:, 1] = 1
			d_loss  = discriminator.train_on_batch(X,y)
			losses["d"].append(d_loss)
		
		if args.learning_rate_decay and (e+1)%args.learning_rate_decay == 0:
			args.discriminator_learning_rate /= 2
			args.generator_learning_rate /= 2
			print('Learning rates decayed, dlr:', args.discriminator_learning_rate, 'glr:', args.generator_learning_rate)

		# Save images during optimization 
		if e%args.fps == args.fps-1:
			save_path = args.save_path + '/' + args.mode + '/epoch_' + str(e) + '.jpg'
			print('iteration:',e, 'of:', args.epochs, 'generator loss:', g_loss, 'discriminator loss:', d_loss, '\nsave image at:', save_path)
			dim = np.sqrt(args.plot_examples)

			if K.image_data_format() == 'channels_first':
				channel_idx = 1
			else:
				channel_idx = -1
			if x_train.shape[channel_idx] == 1:
				plot_gen(args, generator, n_ex=args.plot_examples, dim=(dim, dim), random_seeds=samples_seeds, save_path=save_path)
			elif x_train.shape[channel_idx] == 3 or x_train.shape[channel_idx] == 4:
				plot_gen_color(args, generator, n_ex=args.plot_examples, dim=(dim, dim), random_seeds=samples_seeds, save_path=save_path)
		

def plot_real(x_train, n_ex=16, dim=(4,4), figsize=(10,10)):	
	idx = np.random.randint(0,x_train.shape[0],n_ex)
	generated_images = x_train[idx,:,:,:]

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = generated_images[i,0,:,:]
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	plot_name = "./figures/plot_real.png"
	if not os.path.exists(os.path.dirname(plot_name)):
		os.makedirs(os.path.dirname(plot_name))		
	plt.savefig(plot_name)


def plot_color(x_train, n_ex=16,dim=(4,4), figsize=(10,10) ):	
	idx = np.random.randint(0,x_train.shape[0],n_ex)
	generated_images = x_train[idx,:,:,:]

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = np.rollaxis(generated_images[i], 0, 3)
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	plot_name = "./figures/plot_color.png"
	if not os.path.exists(os.path.dirname(plot_name)):
		os.makedirs(os.path.dirname(plot_name))		
	plt.savefig(plot_name)


def plot_gen_color(args, generator, n_ex=16, dim=(4,4), figsize=(24,24), random_seeds=None, save_path=None):
	if random_seeds is None:
		random_seeds = np.random.uniform(0,1,size=[n_ex,args.seeds])
	generated_images = generator.predict(random_seeds)

	fig = plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		#img = np.rollaxis(generated_images[i], 0, 3)
		img = generated_images[i]
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	if save_path:
		if not os.path.exists(os.path.dirname(save_path)):
			os.makedirs(os.path.dirname(save_path))
		plt.savefig(save_path)
		plt.close(fig)
	else:
		plt.show()


def plot_loss(losses):
#        display.clear_output(wait=True)
#        display.display(plt.gcf())
		plt.figure(figsize=(10,8))
		plt.plot(losses["d"], label='discriminitive loss')
		plt.plot(losses["g"], label='generative loss')
		plt.legend()
		plt.show()

def plot_gen(args, generator, n_ex=16, dim=(4,4), figsize=(10,10), random_seeds=None, save_path=None):
	if random_seeds is None:
		random_seeds = np.random.uniform(0,1,size=[n_ex, args.seeds])
	generated_images = generator.predict(random_seeds)

	fig = plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = generated_images[i,:,:,0]
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	if save_path:
		if not os.path.exists(os.path.dirname(save_path)):
			os.makedirs(os.path.dirname(save_path))
		plt.savefig(save_path)
		plt.close(fig)
	else:
		plt.show()


if __name__=='__main__':
	run()