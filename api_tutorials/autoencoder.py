# See https://blog.keras.io/building-autoencoders-in-keras.html
# July 2017
# Sam Friedman
# sam@broadinstitute.org


# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


# Imports
import os
import cv2
import random
import argparse
import numpy as np
from Bio import Seq, SeqIO
from keras import regularizers
from keras import backend as K
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Input, Dense
from keras.callbacks import TensorBoard
from keras.datasets import mnist, cifar10
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, LSTM, RepeatVector


def run():
	#autoencode_faces()
	#conv_autoencode_cifar()
	#conv_autoencode_mnist()
	autoencode_mnist()
	#autoencode_cifar()


def autoencode_mnist():
	(x_train, y_train), (x_test, y_test) = load_mnist()
	encoder, decoder, autoencoder = build_simple_autoencoder(l1_penalty=1e-7)
	autoencoder.fit(x_train, x_train, epochs=50, batch_size=256,shuffle=True, validation_data=(x_test, x_test))

	# encode and decode some digits
	# note that we take them from the *test* set
	encoded_imgs = encoder.predict(x_test)
	decoded_imgs = decoder.predict(encoded_imgs)
	plot_imgs_and_reconstructions(x_test, decoded_imgs, n=15)


def conv_autoencode_mnist():
	(x_train, y_train), (x_test, y_test) = load_mnist(flatten=False)
	autoencoder = build_conv_autoencoder()
	autoencoder.summary()
	autoencoder.fit(x_train, x_train,
		epochs=55,
		batch_size=128,
		shuffle=True,
		validation_data=(x_test, x_test),
		callbacks=[TensorBoard(log_dir='./tmp/autoencoder')])	
	
	decoded_imgs = autoencoder.predict(x_test)
	plot_imgs_and_reconstructions(x_test, decoded_imgs, n=10)


def autoencode_cifar():
	(x_train, y_train), (x_test, y_test) = load_cifar()
	encoder, decoder, autoencoder = build_simple_autoencoder(input_dim=3072, encoding_dim=48)
	autoencoder.fit(x_train, x_train, epochs=100, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

	# encode and decode some digits
	# note that we take them from the *test* set
	encoded_imgs = encoder.predict(x_test)
	decoded_imgs = decoder.predict(encoded_imgs)
	plot_imgs_and_reconstructions(x_test, decoded_imgs, n=10, shape=(32,32,3))


def autoencode_faces():
	(x_train, y_train), (x_test, y_test) = load_faces()
	autoencoder = build_conv_autoencoder(input_dim=(64,64,3))
	autoencoder.fit(x_train, x_train, epochs=20, batch_size=64, shuffle=True, validation_data=(x_test, x_test))

	# encode and decode some digits
	# note that we take them from the *test* set
	decoded_imgs = autoencoder.predict(x_test)
	plot_imgs_and_reconstructions(x_test, decoded_imgs, n=10, shape=(64,64,3))



def conv_autoencode_cifar():
	(x_train, y_train), (x_test, y_test) = load_cifar(flatten=False)
	autoencoder = build_conv_autoencoder(input_dim=(32,32,3))
	autoencoder.summary()

	autoencoder.fit(x_train, x_train,
		epochs=25,
		batch_size=64,
		shuffle=True,
		validation_data=(x_test, x_test),
		callbacks=[TensorBoard(log_dir='./tmp/autoencoder')])	
	
	decoded_imgs = autoencoder.predict(x_test)
	plot_imgs_and_reconstructions(x_test, decoded_imgs, n=10, shape=(32,32,3))


def build_simple_autoencoder(input_dim=784, encoding_dim=32, l1_penalty=0.):
	# this is the size of our encoded representations
	# 32 floats -> compression of factor 24.5, assuming the input is 784 floats
	# this is our input placeholder
	input_img = Input(shape=(input_dim,))
	
	# "encoded" is the encoded representation of the input
	encoded = Dense(encoding_dim, activation='relu',  activity_regularizer=regularizers.l1(l1_penalty))(input_img)
	
	# "decoded" is the lossy reconstruction of the input
	decoded = Dense(input_dim, activation='sigmoid')(encoded)

	# this model maps an input to its reconstruction
	autoencoder = Model(input_img, decoded)

	# this model maps an input to its encoded representation
	encoder = Model(input_img, encoded)

	# create a placeholder for an encoded (32-dimensional) input
	encoded_input = Input(shape=(encoding_dim,))
	# retrieve the last layer of the autoencoder model
	decoder_layer = autoencoder.layers[-1]
	# create the decoder model
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return encoder, decoder, autoencoder


def build_conv_autoencoder(input_dim=(28, 28, 1)):
	input_img = Input(shape=input_dim)  # adapt this if using `channels_first` image data format

	x = Conv2D(512, (3, 3), activation='relu', padding='same')(input_img)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = MaxPooling2D((2, 2), padding='same')(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	encoded = MaxPooling2D((2, 2), padding='same')(x)

	# at this point the representation is (4, 4, 8) i.e. 128-dimensional

	x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
	x = UpSampling2D((2, 2))(x)
	x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)
	x = UpSampling2D((2, 2))(x)
	if input_dim[0] == 28:
		x = Conv2D(512, (3, 3), activation='relu')(x)
	else:
		x = Conv2D(512, (3, 3), activation='relu', padding='same')(x)

	x = UpSampling2D((2, 2))(x)
	decoded = Conv2D(input_dim[2], (3, 3), activation='sigmoid', padding='same')(x)

	autoencoder = Model(input_img, decoded)
	autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
	return autoencoder


def build_lstm_autoencoder(timesteps, input_dim)
	inputs = Input(shape=(timesteps, input_dim))
	encoded = LSTM(latent_dim)(inputs)

	decoded = RepeatVector(timesteps)(encoded)
	decoded = LSTM(input_dim, return_sequences=True)(decoded)

	sequence_autoencoder = Model(inputs, decoded)
	encoder = Model(inputs, encoded)
	return encoder, sequence_autoencoder


def load_mnist(flatten=True):
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	x_train = x_train.astype('float32') / 255.
	x_test = x_test.astype('float32') / 255.
	
	if flatten:
		x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
		x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	else:
		x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))  # adapt this if using `channels_first` image data format
		x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))  # adapt this if using `channels_first` image data format
	
	print(x_train.shape)
	print(x_test.shape)

	return (x_train, y_train), (x_test, y_test)


def load_cifar(flatten=True):
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255.
	x_test /= 255.
	
	if flatten:
		x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
		x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	else:
		x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))  # adapt this if using `channels_first` image data format
		x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))  # adapt this if using `channels_first` image data format
	
	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')	
	return (x_train, y_train), (x_test, y_test)


def load_faces(num_labels=14000, shape=(64,64)):
	imagenet_path = '/home/sam/big_data/faces/lfw/'
	train_paths = [ imagenet_path + tp for tp in sorted(os.listdir(imagenet_path)) if os.path.isdir(imagenet_path + tp)  ]
	(x_train, y_train), (x_test, y_test) = load_images_from_class_dirs(train_paths, num_labels, shape=shape)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255.
	x_test /= 255.

	print('bounds:', np.min(x_train), np.max(x_train))
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train, y_train), (x_test, y_test)


def image_as_matrix(image_path, expand_dims=False, shape=(224,224)):
	img = cv2.resize(cv2.imread(image_path), shape).astype(np.float32)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	return img


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


def build_dna_dataset(k, vocabulary_size):
	dna_words = []
	ref_string = dna_from_reference()
	for i in range(vocabulary_size):
		while 'N' in ref_string[i*k : (i+1)*k]:
			i += 1 
		dna_words.append(ref_string[i*k : (i+1)*k])
	print('DNA words:', dna_words[:5])
	return dna_words


def dna_from_reference(chrom='9'):
	#reference_hg19 = '/dsde/data/deep/vqsr/Homo_sapiens_assembly19.fasta'
	reference_hg19 = '/Users/sam/vqsr_data/Homo_sapiens_assembly19.fasta'

	record_dict = SeqIO.to_dict(SeqIO.parse(reference_hg19, "fasta"))
	dna = str(record_dict[chrom].seq[10000000:75000000])
	return dna


def plot_imgs_and_reconstructions(imgs, reconstructions, n=10, shape=(28,28)):
	plt.figure(figsize=(20, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(imgs[i].reshape(shape))
		if len(shape) == 2:
			plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(reconstructions[i].reshape(shape))
		if len(shape) == 2:
			plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.show()


# Back to the top!
if "__main__" == __name__:
	run()