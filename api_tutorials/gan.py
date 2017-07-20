#gan.py
#sam@broadinstitute.org

# Keras GAN Implementation
# See: https://oshearesearch.com/index.php/2016/07/01/mnist-generative-adversarial-model-in-keras/

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import cv2
import time
import os,random
import numpy as np
import theano as th
import theano.tensor as T
import matplotlib.pyplot as plt
import pickle, random, sys, keras

import keras.models as models
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

dropout_rate = 0.15
seeds = 300


def run():
	gan_on_imagenet()
	#gan_on_cifar()
	#gan_on_mnist()


def gan_on_imagenet():
	in_shape = (3,128,128)
	generator = build_imagenet_generative_model()
	discriminator = build_imagenet_discriminative(in_shape)
	gan = build_stacked_gan_imagenet(generator, discriminator)	
	
	train_imagenet_gan(generator, discriminator, gan, pretrain=True)	


def gan_on_cifar():
	in_shape = (3,32,32)
	generator = build_generative_model()
	discriminator = build_discriminative(in_shape)
	gan = build_stacked_gan(generator, discriminator)	

	# gan = load_model('cifar10_gan.hd5')
	# generator = load_model('cifar10_generator.hd5') 
	# discriminator = load_model('cifar10_discriminator.hd5')
	data = load_cifar()
	train_cifar_gan(generator, discriminator, gan, pretrain=True)


def gan_on_mnist():
	in_shape = (1,28,28)

	generator = build_mnist_generative_model()
	discriminator = build_mnist_discriminative(in_shape)
	gan = build_stacked_gan(generator, discriminator)

	# gan = load_model('mnist_gan.hd5')
	# generator = load_model('mnist_generator.hd5') 
	# discriminator = load_model('mnist_discriminator.hd5')

	train_mnist_gan(generator, discriminator, gan, pretrain=True)	


def load_mnist():
	img_rows, img_cols = 28, 28

	# the data, shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
	x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
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


def load_imagenet(num_labels=24):

	imagenet_path = '/home/sam/big_data/imagenet/ILSVRC2014_DET_train/'
	train_paths = [ imagenet_path + tp for tp in sorted(os.listdir(imagenet_path)) if os.path.isdir(imagenet_path + tp)  ]
	(x_train, y_train), (x_test, y_test) = load_images_from_class_dirs(train_paths, num_labels, shape=(128,128))

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

def build_imagenet_generative_model():
	# Build Generative model ...
	nch = 316
	dense_channels = 60
	inner_dim = 8
	channel_axis = 1
	g_input = Input(shape=[seeds])
	H = Dense(dense_channels*inner_dim*inner_dim, kernel_initializer='glorot_normal')(g_input)
	#H = BatchNormalization(scale=False, axis=channel_axis)(H)
	H = Activation('relu')(H)
	H = Reshape( [dense_channels, inner_dim, inner_dim] )(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	#H = BatchNormalization(scale=False, axis=channel_axis)(H)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False, axis=channel_axis)(H)
	H = Activation('relu')(H)	
	H = UpSampling2D(size=(2, 2))(H)	
	H = Conv2D(nch, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False, axis=channel_axis)(H)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)	
	H = Conv2D(nch, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False, axis=channel_axis)(H)
	H = Activation('relu')(H)	
	H = Conv2D(3, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
	#H = ActivityRegularization(l1=0, l2=1e-8)(H)
	H = Activation('sigmoid')(H)
	generator = Model(g_input, H)
	opt = Adam(lr=1e-6)		
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
	return generator


def build_imagenet_discriminative(in_shape, out_labels=2):
	# Build Discriminative model ...
	d_input = Input(shape=in_shape)
	H = Conv2D(512, (5, 5), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform', activation='relu')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)	
	H = Conv2D(316,  (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform', activation='relu')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	H = Conv2D(256,  (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = Conv2D(128,  (3, 3), strides=(2, 2), padding='same', kernel_initializer='glorot_uniform', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	H = Flatten()(H)
	H = Dense(64)(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	probability_out = Dense(out_labels, activation='softmax')(H)
	discriminator = Model(d_input, probability_out)
	dopt = Adam(lr=1e-5)	
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_stacked_gan_imagenet(generator, discriminator):
	# Build stacked GAN model
	gan_input = Input(shape=[seeds])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)
	opt = Adam(lr=1e-6)	
	GAN.compile(loss='binary_crossentropy', optimizer=opt)
	GAN.summary()
	return GAN

def build_generative_model():
	# Build Generative model ...
	nch = 40
	inner_dim = 8
	g_input = Input(shape=[seeds])
	H = Dense(nch*inner_dim*inner_dim, kernel_initializer='glorot_normal')(g_input)
	H = BatchNormalization(scale=False,)(H)
	H = LeakyReLU(0.2)(H)
	H = Reshape( [nch, inner_dim, inner_dim] )(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*5, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False,)(H)
	H = LeakyReLU(0.2)(H)
	H = SpatialDropout2D(0.25)(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False,)(H)
	H = LeakyReLU(0.2)(H)
	H = SpatialDropout2D(0.25)(H)
	H = Conv2D(nch*2, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(scale=False,)(H)
	H = Activation('relu')(H)
	H = SpatialDropout2D(0.25)(H)	
	H = Conv2D(3, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	#H = ActivityRegularization(l1=0, l2=0.00001)(H)

	H = Activation('sigmoid')(H)
	generator = Model(g_input, H)
	opt = Adam(lr=1e-5) #RMSprop(lr=1e-5)
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
	return generator


def build_discriminative(in_shape, out_labels=2):
	# Build Discriminative model ...
	d_input = Input(shape=in_shape)
	H = Conv2D(512, (3, 3), strides=(2,2), padding='same', activation='relu')(d_input)
	H = LeakyReLU(0.2)(H)
	H = SpatialDropout2D(dropout_rate)(H)
	H = Conv2D(396, (3, 3), strides=(2,2), padding='same', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = Conv2D(256, (3, 3), strides=(2,2), padding='same', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = SpatialDropout2D(dropout_rate)(H)
	H = Flatten()(H)
	H = Dense(180)(H)
	H = LeakyReLU(0.2)(H)
	d_V = Dense(out_labels, activation='softmax')(H)
	discriminator = Model(d_input, d_V)
	dopt = Adam(lr=1e-4) #RMSprop(lr=1e-5)	
	discriminator.compile(loss=binary_crossentropy, optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_mnist_generative_model():
	# Build Generative model ...
	nch = 80
	bn_axis = 1
	g_input = Input(shape=[seeds])
	H = Dense(nch*7*7, kernel_initializer='glorot_normal')(g_input)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = Reshape( [nch, 7, 7] )(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*6, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = UpSampling2D(size=(2, 2))(H)
	H = Conv2D(nch*4, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)
	H = Conv2D(nch*2, (3, 3), padding='same', kernel_initializer='glorot_uniform')(H)
	H = BatchNormalization(axis=bn_axis, scale=False)(H)
	H = Activation('relu')(H)	
	H = Conv2D(1, (1, 1), padding='same', kernel_initializer='glorot_uniform')(H)
	H = ActivityRegularization(l1=0.000001, l2=0)(H)
	g_V = Activation('sigmoid')(H)
	generator = Model(g_input,g_V)
	opt = RMSprop(lr=1e-5)	
	generator.compile(loss='binary_crossentropy', optimizer=opt)
	generator.summary()
	return generator


def build_mnist_discriminative(in_shape):
	# Build Discriminative model ...
	d_input = Input(shape=in_shape)
	H = Conv2D(512, (5, 5), strides=(2, 2), padding='same', activation='relu')(d_input)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	H = Conv2D(316, (3, 3), strides=(2, 2), padding='same', activation='relu')(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	H = Flatten()(H)
	H = Dense(140)(H)
	H = LeakyReLU(0.2)(H)
	H = Dropout(dropout_rate)(H)
	d_V = Dense(2, activation='softmax')(H)
	discriminator = Model(d_input,d_V)
	dopt = RMSprop(lr=1e-4)	
	discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
	discriminator.summary()
	return discriminator


def build_stacked_gan(generator, discriminator):
	# Build stacked GAN model
	gan_input = Input(shape=[seeds])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)
	#opt = RMSprop(lr=2e-5)	
	opt = RMSprop(lr=1e-4)		
	GAN.compile(loss='binary_crossentropy', optimizer=opt)
	GAN.summary()
	return GAN


def train_imagenet_gan(generator, discriminator, gan, pretrain=True):

	imagenet_data = load_faces()
	(x_train, y_train), (x_test, y_test) = imagenet_data
	#plot_color(x_train)

	ntrain = min(6000, x_train.shape[0])-1
	trainidx = random.sample(range(0, x_train.shape[0]), ntrain)
	xt = x_train[trainidx,:,:,:]

	if pretrain:	# Pre-train the discriminator network ...
		noise_gen = np.random.uniform(0,1,size=[xt.shape[0],seeds])
		generated_images = generator.predict(noise_gen)
		x = np.concatenate((xt, generated_images))
		n = xt.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1
		print('np sum 1s:', np.sum(y[:,0]), 'x shape:', x.shape)
		make_trainable(discriminator,True)
		discriminator.fit(x,y, epochs=1, batch_size=32, shuffle=True)
		y_hat = discriminator.predict(x)

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
	train_for_n_memory(imagenet_data, generator, discriminator, gan, epochs=35000, plt_frq=31, batch_size=32, title='faces5')
	save_model(gan, 'face3_gan.hd5')
	save_model(generator, 'face3_generator.hd5') 
	save_model(discriminator, 'face3_discriminator.hd5')
	# Plot some generated images from our GAN after training
	plot_gen_color(generator, 25, (5,5), (34,34))


def train_cifar_gan(generator, discriminator, gan, pretrain=True):

	cifar_data = load_cifar()
	(x_train, y_train), (x_test, y_test) = cifar_data


	if pretrain: # Pre-train the discriminator network ...
		ntrain = 20000
		trainidx = random.sample(range(0, x_train.shape[0]), ntrain)
		xt = x_train[trainidx,:,:,:]
		noise_gen = np.random.uniform(0, 1, size=[xt.shape[0],seeds])
		generated_images = generator.predict(noise_gen)
		x = np.concatenate((xt, generated_images))
		n = xt.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1
		print('np sum 1s:', np.sum(y[:,0]), 'x shape:', x.shape)
		make_trainable(discriminator,True)
		discriminator.fit(x,y, epochs=1, batch_size=128, shuffle=True)
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
	train_for_n(cifar_data, generator, discriminator, gan, epochs=40000, plt_frq=25, batch_size=32, title='cifar1')
	save_model(gan, 'cifar10_gan2.hd5')
	save_model(generator, 'cifar10_generator2.hd5') 
	save_model(discriminator, 'cifar10_discriminator2.hd5')
	# Plot some generated images from our GAN
	plot_gen_color(generator, 25,(5,5),(12,12))


def train_mnist_gan(generator, discriminator, gan, pretrain=False):
	ntrain = 10000
	mnist_data = load_mnist()
	(x_train, y_train), (x_test, y_test) = mnist_data
	trainidx = random.sample(range(0,x_train.shape[0]), ntrain)
	XT = x_train[trainidx,:,:,:]

	if pretrain:	# Pre-train the discriminator network ...
		noise_gen = np.random.uniform(0,1,size=[XT.shape[0],seeds])
		generated_images = generator.predict(noise_gen)
		X = np.concatenate((XT, generated_images))
		n = XT.shape[0]
		y = np.zeros([2*n,2])
		y[:n, 0] = 1
		y[n:, 1] = 1

		make_trainable(discriminator,True)
		discriminator.fit(X,y, epochs=1, batch_size=128)
		y_hat = discriminator.predict(X)

		# Measure accuracy of pre-trained discriminator network
		y_hat_idx = np.argmax(y_hat,axis=1)
		y_idx = np.argmax(y,axis=1)
		diff = y_idx-y_hat_idx
		n_tot = y.shape[0]
		n_rig = (diff==0).sum()
		acc = n_rig*100.0/n_tot
		print("Accuracy: %0.02f pct (%d of %d) right"%(acc, n_rig, n_tot))
			
	# Plot some generated images from our GAN
	# plot_gen(generator, 25,(5,5),(12,12))
	# Train for 6000 epochs at original learning rates
	train_for_n(mnist_data, generator, discriminator, gan, epochs=6000, plt_frq=50, batch_size=32, title='mnist8')

	# Plot some generated images from our GAN
	plot_gen(generator, 25,(5,5),(12,12))
	# Plot real MNIST images for comparison
	save_model(gan, 'mnist_gan.hd5')
	save_model(generator, 'mnist_generator.hd5') 
	save_model(discriminator, 'mnist_discriminator.hd5')
	plot_real(x_train)


# Set up our main training loop
def train_for_n_memory(data, generator, discriminator, gan, epochs=5000, plt_frq=25, batch_size=32, title=None):
	# set up loss storage vector
	losses = {"d":[], "g":[]}
	(x_train, y_train), (x_test, y_test) = data
	print('bounds:', np.min(x_train), np.max(x_train), 'x_train shape:', x_train.shape, 'test shape', x_test.shape)
	plot_examples = 16 # should be perfect square
	samples_seeds = np.random.uniform(0,1,size=[plot_examples, seeds])
	
	# Keep a running list of generated sample
	noise_gen = np.random.uniform(0,1,size=[batch_size, seeds])
	generated_images = generator.predict(noise_gen)
	generated_idx = 0

	for e in range(epochs):  
		
		make_trainable(discriminator, True)
		make_trainable(generator, False)
		# Make generative images
		image_batch = x_train[np.random.randint(0,x_train.shape[0], size=batch_size),:,:,:]    
		noise_gen = np.random.uniform(0,1,size=[1, seeds])
		generated_images[generated_idx] = generator.predict(noise_gen)[0]
		generated_idx = (generated_idx + 1) % batch_size

		# Train discriminator on generated images
		X = np.concatenate((image_batch, generated_images))
		y = np.zeros([2*batch_size,2])
		y[:batch_size, 0] = 1
		y[batch_size:, 1] = 1
		d_loss  = discriminator.train_on_batch(X,y)
		losses["d"].append(d_loss)
	


		make_trainable(discriminator, False)
		make_trainable(generator, True)
		# train Generator-Discriminator stack on input noise to non-generated output class
		noise_tr = np.random.uniform(0, 1, size=[batch_size,seeds])
		y2 = np.zeros([batch_size,2])
		# Tell the model that random is correct
		y2[:, 0] = 1
		g_loss = gan.train_on_batch(noise_tr, y2)
		losses["g"].append(g_loss)
		
		# Updates 
		if e%plt_frq==plt_frq-1:
			print('iteration:',e, 'of:', epochs, 'generator loss:', g_loss, 'discriminator loss:', d_loss)
			if x_train.shape[1] == 1:
				if title:
					save_path = './frames/gan/' + title + '/epoch_' + str(e) + '.jpg'
					plot_gen(generator, n_ex=plot_examples, random_seeds=samples_seeds, save_path=save_path)
				else:
					plot_gen(generator, n_ex=plot_examples, random_seeds=samples_seeds)
					plot_loss(losses)
			elif  x_train.shape[1] == 3 or  x_train.shape[1] == 4:
				if title:
					save_path = './frames/gan/' + title + '/samples_at_epoch_' + str(e) + '.jpg'				
					plot_gen_color(generator, n_ex=plot_examples, random_seeds=samples_seeds, save_path=save_path)
				else:
					plot_gen_color(generator, n_ex=plot_examples, random_seeds=samples_seeds)
					plot_color(x_train)


# Set up our main training loop
def train_for_n(data, generator, discriminator, gan, epochs=5000, plt_frq=25, batch_size=32, title=None):
	# set up loss storage vector
	losses = {"d":[], "g":[]}
	(x_train, y_train), (x_test, y_test) = data
	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	plot_examples = 16 # should be perfect square
	samples_seeds = np.random.uniform(0,1,size=[plot_examples, seeds])
	for e in range(epochs):  
		
		make_trainable(discriminator, True)	
		make_trainable(generator, False)	
		for _ in range(2):
			# Make generative images
			image_batch = x_train[np.random.randint(0,x_train.shape[0], size=batch_size),:,:,:]    
			noise_gen = np.random.uniform(0,1,size=[batch_size, seeds])
			generated_images = generator.predict(noise_gen)
			
			# Train discriminator on generated images
			X = np.concatenate((image_batch, generated_images))
			y = np.zeros([2*batch_size,2])
			y[:batch_size, 0] = 1
			y[batch_size:, 1] = 1
			d_loss  = discriminator.train_on_batch(X,y)
			losses["d"].append(d_loss)
	

		make_trainable(discriminator, False)
		make_trainable(generator, True)
		for _ in range(2):
			# train Generator-Discriminator stack on input noise to non-generated output class
			noise_tr = np.random.uniform(0, 1, size=[batch_size,seeds])
			y2 = np.zeros([batch_size,2])
			# Tell the model that random is correct
			y2[:, 0] = 1
			g_loss = gan.train_on_batch(noise_tr, y2)
			losses["g"].append(g_loss)
		
		# Updates 
		if e%plt_frq==plt_frq-1:
			print('iteration:',e, 'of:', epochs, 'generator loss:', g_loss, 'discriminator loss:', d_loss)
			if x_train.shape[1] == 1:
				if title:
					save_path = './frames/gan/' + title + '/epoch_' + str(e) + '.jpg'
					plot_gen(generator, n_ex=plot_examples, random_seeds=samples_seeds, save_path=save_path)
				else:
					plot_gen(generator, n_ex=plot_examples, random_seeds=samples_seeds)
					plot_loss(losses)
			elif  x_train.shape[1] == 3 or  x_train.shape[1] == 4:
				if title:
					save_path = './frames/gan/' + title + '/samples_at_epoch_' + str(e) + '.jpg'				
					plot_gen_color(generator, n_ex=plot_examples, random_seeds=samples_seeds, save_path=save_path)
				else:
					plot_gen_color(generator, n_ex=plot_examples, random_seeds=samples_seeds)
					plot_color(x_train)
		

def plot_real(x_train, n_ex=16,dim=(4,4), figsize=(10,10) ):	
	idx = np.random.randint(0,x_train.shape[0],n_ex)
	generated_images = x_train[idx,:,:,:]

	plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = generated_images[i,0,:,:]
		plt.imshow(img)
		plt.axis('off')
	plt.tight_layout()
	plt.show()


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
	plt.show()


def plot_gen_color(generator, n_ex=16, dim=(4,4), figsize=(24,24), random_seeds=None, save_path=None):
	if random_seeds is None:
		random_seeds = np.random.uniform(0,1,size=[n_ex,seeds])
	generated_images = generator.predict(random_seeds)

	fig = plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = np.rollaxis(generated_images[i], 0, 3)
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

def plot_gen(generator, n_ex=16, dim=(4,4), figsize=(10,10), random_seeds=None, save_path=None):
	if random_seeds is None:
		random_seeds = np.random.uniform(0,1,size=[n_ex,seeds])
	generated_images = generator.predict(random_seeds)

	fig = plt.figure(figsize=figsize)
	for i in range(generated_images.shape[0]):
		plt.subplot(dim[0],dim[1],i+1)
		img = generated_images[i,0,:,:]
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