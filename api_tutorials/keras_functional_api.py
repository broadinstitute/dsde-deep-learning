# keras_functional_api.py
# DSDE Deep Learning
#
# Several model architectures using the functional api
# models can have many type of inputs and outputs
#
# February 2017
# Sam Friedman 
# sam@broadinstitute.org

import os
#import cv2
import h5py
import time
import gzip
import cPickle
import argparse
import numpy as np
#from vgg_16 import vgg_16
#from scipy.misc import imsave
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist
from inception_v3 import inception_v3
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Convolution2D, Input, ZeroPadding2D, MaxPooling2D
from keras.layers.convolutional import Convolution1D, Convolution2D, Convolution3D, MaxPooling1D, MaxPooling2D, AveragePooling2D
from keras.layers import Input, Dense, Dropout, BatchNormalization, SpatialDropout2D, SpatialDropout3D, Activation, Flatten, Reshape, merge


#data_root = '/Users/sam/Dropbox'
data_root = '/home/sam/Dropbox/'
data_path = data_root+'Code/python/cnn/saved_networks/'

inception_weights = data_path + 'inception_v3_weights_th_dim_ordering_th_kernels.h5'
vgg_weights = data_path + 'vgg16_weights_th_dim_ordering_th_kernels.h5'


def run():
	args = parse_args()

	model = model_from_args(args)

	if 'mnist_cnn' == args.mode:
		train_cnn_on_mnist(args, model)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--weights', default='')
	parser.add_argument('--model', default='mnist_cnn')
	parser.add_argument('--num_labels', default=10, type=int)
	parser.add_argument('--width', default=28, type=int)
	parser.add_argument('--height', default=28, type=int)
	parser.add_argument('--channels', default=1, type=int)
	parser.add_argument('--mode', default='mnist_cnn')
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--iterations', default=20, type=int)
	parser.add_argument('--image_path', default='/Users/sam/Dropbox/Photos/dog.jpg')	
	parser.add_argument('--save_path', default='/Users/sam/Dropbox/Photos/activations/')
	parser.add_argument('--data', default='./mnist.pkl.gz')

	args = parser.parse_args()
	print 'Arguments are', args	
	return args


def model_from_args(args):
	input_image = Input(shape=(args.channels, args.height, args.width), name='input_image')

	if 'inception' == args.model:
		model = inception_v3(args.num_labels, inception_weights, input_image)
	elif 'vgg' == args.model:
		model = vgg_16(args.num_labels, vgg_weights, input_image)
	elif 'mnist_cnn' == args.model:
		model = mnist_cnn(args, input_image)
	else:
		print '\n\nError: unknown model architecture:', args.model	

	return model
	

def mnist_cnn(args, input_image):
	shape = (args.channels, args.height, args.width)
	x = Convolution2D(32, 5, 5, 
		activation='relu', 
		border_mode='valid', 
		input_shape=shape)(input_image)
	x = MaxPooling2D((2,2))(x)			
	x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
	x = Dropout(0.2)(x)
	x = MaxPooling2D((2,2))(x)	
	x = Flatten()(x)
	x = Dense(128, activation='relu')(x)
	x = Dense(64, activation='relu')(x)

	predictions = Dense(args.num_labels, activation='softmax')(x)

	# this creates a model that includes
	# the Input layer and three Dense layers
	model = Model(input=input_image, output=predictions)
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	model.summary()
	return model


def train_cnn_on_mnist(args, model):
	#train, test, valid = load_data('/home/sam/Dropbox/Code/python/cnn/data/mnist.pkl.gz')
	#train, test, valid = load_data('/dsde/data/deep/mnist.pkl.gz')
	train, test, valid = load_data(args.data)

	input_images = train[0]
	input_images = input_images.reshape(-1, 1, 28, 28)
	train_labels = make_one_hot(train[1])
	valid_images = valid[0]
	valid_images = valid_images.reshape(-1, 1, 28, 28)
	valid_labels = make_one_hot(valid[1])

	model.fit(input_images, train_labels,
		validation_data=(valid_images, valid_labels), 
		batch_size=32, nb_epoch=10)  # starts training



def gtf_labels_from_cnn(args):
	bed_dict, bed_labels = td.gtf_to_bed_with_labels(defines.encode_gtf_file)
	d1 = td.load_dna_and_gtf_label(args, bed_dict, bed_labels, ['gene', 'protein_coding'])
	labels2 = ['exon', 'transcript', 'processed_transcript', 'pseudogene', 'lincRNA', 'antisense']
	d2 = td.load_dna_and_gtf_label(args, bed_dict, bed_labels, labels2)
	labels3 = ['start_codon', 'UTR', 'stop_codon', 'CDS', 'antisense']
	d3 = td.load_dna_and_gtf_label(args, bed_dict, bed_labels, labels3)
	labels4 = ['miRNA', 'misc_RNA', 'snRNA', 'snoRNA', 'sense_intronic', 'polymorphic_pseudogene', 'processed_transcript']
	d4 = td.load_dna_and_gtf_label(args, bed_dict, bed_labels, labels4)

	d5 = td.concat_and_shuffle_all(d1, d2)
	d6 = td.concat_and_shuffle_all(d3, d4)
	all_data = td.concat_and_shuffle_all(d5, d6)
	train, valid, test = td.split_data(all_data)
	
	weight_path = weight_path_from_args(args)
	model = models.build_multi_label_dna_cnn(args)
	model = models.train_dna_multi_labeller(model, train, valid, weight_path)
	y_pred = model.predict(test[0], batch_size=32, verbose=0)	

	title = plots.weight_path_to_title(weight_path)
	plots.plot_roc_pred(y_pred[0], test[1], defines.feature_type_labels, title)
	plots.plot_roc_per_class_pred(y_pred[0], test[1], defines.feature_type_labels, title)
	
	title = 'gene_label_' + title
	plots.plot_roc_pred(y_pred[1], test[2], defines.bio_type_labels, title)
	plots.plot_roc_per_class_pred(y_pred[1], test[2], defines.bio_type_labels, title)
	

def chrom_labels_from_cnn(args):
	train_data1 = td.load_dna_and_chrom_label(args, ['T', 'R'])
	args.samples *= 2
	train_data2 = td.load_dna_and_chrom_label(args, ['PF', 'E', 'TSS', 'CTCF', 'WE'])
	train_data = td.concat_and_shuffle(train_data1, train_data2)

	train, valid, test = td.split_data(train_data)

	weight_path = weight_path_from_args(args)
	model = models.build_sequential_chrom_label(args)
	model = models.train_chrom_labeller(model, train, valid, weight_path)

	title = plots.weight_path_to_title(weight_path)
	plots.plot_roc(model, test[0], test[1], defines.chrom_hmm_labels, title)
	plots.plot_roc_per_class(model, test[0], test[1], defines.chrom_hmm_labels, title)


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
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print 'loading data...'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #which row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    return train_set, valid_set, test_set


def make_one_hot(y):
	ohy = np.zeros((len(y), 10))
	for i in range(0, len(y)):
		ohy[i][y[i]] = 1.0
	return ohy


if __name__ == '__main__':
	run()



