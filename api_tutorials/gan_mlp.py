import os,random
import numpy as np
import matplotlib.pyplot as plt

from keras.optimizers import *
from keras.activations import *
from keras.regularizers import *
from keras.layers import Input, merge
from keras.layers.normalization import *
from keras.layers.noise import GaussianNoise
from keras.models import Model, save_model, load_model
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense


seeds = 20
num_annotations = 28
gan_on_annotations()


def gan_on_annotations():

	generator = build_mnist_generative_model()
	discriminator = build_mnist_discriminative()
	gan = build_stacked_gan(generator, discriminator)

	train_for_n(data, generator, discriminator, gan, nb_epoch=1000, plt_frq=25, batch_size=32)


def annotation_gan():
	in_shape = (1,28,28)

	generator = build_generative_model()
	discriminator = build_discriminative(in_shape)
	gan = build_stacked_gan(generator, discriminator)

	make_trainable(discriminator, False)	

	train_mnist_gan(generator, discriminator, gan)	


def build_generative_model():
	# Build Generative model ...
	nch = 200
	g_input = Input(shape=[seeds])
	H = Dense(nch, init='glorot_normal')(g_input)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)
	H = Dense(num_annotations, init='glorot_normal')(H)
	H = BatchNormalization(mode=2)(H)
	output = Activation('relu')(H)

	generator = Model(g_input, output)
	opt = Adam(lr=1e-4)

	generator.compile(loss='mse', optimizer=opt)
	generator.summary()

	return generator


def build_discriminative():
	# Build Discriminative model ...
	d_input = Input(shape=num_annotations)
	H = Dense(nch, init='glorot_normal')(d_input)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)
	H = Dense(nch, init='glorot_normal')(d_input)
	H = BatchNormalization(mode=2)(H)
	H = Activation('relu')(H)	
	H = Dropout(0.4)(H)

	d_V = Dense(2, activation='softmax')(H)
	discriminator = Model(d_input, d_V)
	dopt = Adam(lr=1e-4)

	discriminator.compile(loss='mse', optimizer=dopt)
	discriminator.summary()
	
	return discriminator


def build_stacked_gan(generator, discriminator):
	# Build stacked GAN model
	gan_input = Input(shape=[seeds])
	H = generator(gan_input)
	gan_V = discriminator(H)
	GAN = Model(gan_input, gan_V)

	opt = Adam(lr=1e-4)		
	GAN.compile(loss='mse', optimizer=opt)
	GAN.summary()
	return GAN


def make_trainable(net, val):
	net.trainable = val
	for l in net.layers:
		l.trainable = val


def train_for_n(data, generator, discriminator, gan, nb_epoch=5000, plt_frq=25, batch_size=32):
	# set up loss storage vector
	losses = {"d":[], "g":[]}

	for e in range(nb_epoch):  
		
		# Make generative images
		image_batch = x_train[np.random.randint(0, data.shape[0], size=batch_size)]    
		noise_gen = np.random.uniform(0,1,size=[batch_size, seeds])
		generated_images = generator.predict(noise_gen)
		
		# Train discriminator on generated images
		X = np.concatenate((image_batch, generated_images))
		y = np.zeros([2*batch_size,2])
		y[0:batch_size,1] = 1
		y[batch_size:,0] = 1

		make_trainable(discriminator, True)
		make_trainable(generator, False)
		d_loss  = discriminator.train_on_batch(X,y)
		losses["d"].append(d_loss)
	
		# train Generator-Discriminator stack on input noise to non-generated output class
		noise_tr = np.random.uniform(0,1,size=[batch_size,seeds])
		y2 = np.zeros([batch_size,2])
		y2[:,1] = 1

		make_trainable(discriminator, False)
		make_trainable(generator, True)		
		g_loss = gan.train_on_batch(noise_tr, y2 )
		losses["g"].append(g_loss)
		
		# Updates 
		if e%50==49:
			print 'iteration:',e, 'of:', nb_epoch, 'generator loss:', g_loss, 'discriminator loss:', d_loss



