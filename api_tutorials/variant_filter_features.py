# variant_filter_features.py
# February 2018
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import json
import h5py
import time
import scipy
import argparse
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from keras import metrics
from keras import backend as K
from keras.applications import inception_v3
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Convolution2D, Input, ZeroPadding2D, MaxPooling2D

data_root = '/home/sam/Dropbox/'
#data_root = '/Users/sam/Dropbox/'

DNA_SYMBOLS = {'A':0, 'C':1, 'G':2, 'T':3}
DNA_INDEL_SYMBOLS = {'A':0, 'C':1, 'G':2, 'T':3, '*':4}
INPUT_SYMBOLS = {
	'dna' : DNA_SYMBOLS,
	'dna_indel' : DNA_INDEL_SYMBOLS,
}

# Base calling ambiguities, See https://www.bioinformatics.org/sms/iupac.html
AMBIGUITY_CODES = {'K':[0, 0, 0.5, 0.5], 'M':[0.5, 0.5, 0, 0], 'R':[0.5, 0, 0, 0.5], 'Y':[0, 0.5, 0.5, 0], 'S':[0, 0.5, 0, 0.5], 'W':[0.5, 0, 0.5, 0],
				   'B':[0,0.333,0.333,0.334], 'V':[0.333,0.333,0,0.334],'H':[0.333,0.333,0.334,0],'D':[0.333,0,0.333,0.334],
				   'X':[0.25,0.25,0.25,0.25], 'N':[0.25,0.25,0.25,0.25]}


# Annotation sets
ANNOTATIONS = {
				'_' : [], # Allow command line to unset annotations
				'gatk_w_qual' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'QUAL', 'ReadPosRankSum'],
				'best_practices' : ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum'],
				'm2':['AF', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS'],
				'no_het0':['MQ', 'DP', 'SOR', 'QD', 'AF', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS' ],
				'mix':['DP', 'SOR', 'QD', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS' ],
				'mix_no0':['DP', 'SOR', 'QD', 'AD_1', 'MBQ', 'MFRL_1', 'MMQ', 'MPOS' ],
				'combine': ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum', 'AF', 'AD_0', 'AD_1', 'MBQ', 'MFRL_0', 'MFRL_1', 'MMQ', 'MPOS'],
				'gnomad': ['MQ', 'DP', 'SOR', 'FS', 'QD', 'MQRankSum', 'ReadPosRankSum', 'DP_MEDIAN', 'DREF_MEDIAN', 'GQ_MEDIAN', 'AB_MEDIAN'],
			  }

SNP_INDEL_LABELS = {'NOT_SNP':0, 'NOT_INDEL':1, 'SNP':2, 'INDEL':3}

def run():
	'''Parse arguments, create a model and dispatch on mode'''
	args = parse_args()
	model = set_args_and_get_model_from_semantics(args, args.semantics_json)

	if 'write_filters' == args.mode:
		write_filters(args, model)
	elif 'excite_neuron' == args.mode:
		excite_neuron(args, model)
	elif 'excite_layer' == args.mode:
		excite_layer(args, model)
	elif 'excite_softmax' == args.mode:
		excite_softmax(args, model)			
	elif 'deep_dream' == args.mode:
		deep_dream(args, model)		
	elif 'recover' == args.mode:
		recover_image(args, model)			
	elif 'draw' == args.mode:
		draw_loop(args, model)
	elif 'journey' == args.mode:
		image_journey(args, model)	
	elif 'saliency' == args.mode:
		write_saliency(args, model)		
	else:
		raise ValueError('unknown variant visualize mode:', args.mode)


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--maxfun', default=9, type=int)
	parser.add_argument('--fps', default=1, type=int)
	parser.add_argument('--learning_rate', default=0.01, type=float)
	parser.add_argument('--jitter', default=0.0, type=float)
	parser.add_argument('--l2', default=0.0, type=float)
	parser.add_argument('--l1', default=0.0, type=float)
	parser.add_argument('--activity_weight', default=1.0, type=float)
	parser.add_argument('--total_variation', default=0.00001, type=float)
	parser.add_argument('--neuron', default=58, type=int)


	# Tensor defining arguments
	parser.add_argument('--labels', default=SNP_INDEL_LABELS, help='Dict mapping label names to their index within label tensors.')
	parser.add_argument('--input_symbol_set', default='dna_indel', choices=INPUT_SYMBOLS.keys(), help='Key which maps to an input symbol to index mapping.')
	parser.add_argument('--input_symbols', help='Dict mapping input symbols to their index within input tensors, initialised via input_symbols_set argument')
	parser.add_argument('--batch_size', default=32, type=int, help='Mini batch size for stochastic gradient descent algorithms.')
	parser.add_argument('--read_limit', default=128, type=int, help='Maximum number of reads to load.')
	parser.add_argument('--window_size', default=128, type=int, help='Size of sequence window to use as input, typically centered at a variant.')
	parser.add_argument('--channels_last', default=False, dest='channels_last', action='store_true', help='Store the channels in the last axis of tensors, tensorflow->true, theano->false')


	# Annotation arguments
	parser.add_argument('--annotations', help='Array of annotation names, initialised via annotation_set argument')
	parser.add_argument('--annotation_set', default='best_practices', choices=ANNOTATIONS.keys(), help='Key which maps to an annotations list (or None for architectures that do not take annotations).')


	# I/O files and directories: vcfs, bams, beds, hd5, fasta
	parser.add_argument('--semantics_json', default='')
	parser.add_argument('--tensor_name', default='read_tensor', help='Key which looks up the map from tensor channels to their meaning.')
	parser.add_argument('--weights_hd5', default='', help='A hd5 file of weights to initialize a model, will use all layers with names that match.')
	parser.add_argument('--tensor_example', default='', help='A tensor to start visualizations from.')
	parser.add_argument('--data_dir', help='Directory of tensors, must be split into test/valid/train sets with directories for each label within.')
	parser.add_argument('--output_dir', default='./weights/', help='Directory to write models or other data out.')


	# Training and optimization related arguments
	parser.add_argument('--epochs', default=25, type=int, help='Number of epochs, typically passes through the entire dataset, not always well-defined.')
	parser.add_argument('--iterations', default=5, type=int, help='Generic iteration limit for hyperparameter optimization, animation, and other counts.')
	parser.add_argument('--layers', nargs='+', default=['conv', 'dense'], type=str,
		help='List of layer name to investigate.')

	# Run specific arguments
	parser.add_argument('--mode', help='High level recipe: write tensors, train, test or evaluate models.')
	parser.add_argument('--id', default='no_id', help='Identifier for this run, user-defined string to keep experiments organized.')
	parser.add_argument('--random_seed', default=12878, type=int, help='Random seed to use throughout run.  Always use np.random.')


	args = parser.parse_args()
	args.annotations = ANNOTATIONS[args.annotation_set]
	args.input_symbols = INPUT_SYMBOLS[args.input_symbol_set]
	np.random.seed(args.random_seed)
	print('Arguments are', args)
	K.set_learning_phase(0)
	
	return args


def set_args_and_get_model_from_semantics(args, semantics_json):
	'''Recreate a model from a json file specifying model semantics.

	Update the args namespace from the semantics file values.
	Assert that the serialized tensor map and the recreated one are the same.

	Arguments:
		args.tensor_name: String which indicates tensor map to use or None
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
		args.tensor_name = semantics['input_tensor_map_name']
		args.window_size = semantics['window_size']
		args.read_limit = semantics['read_limit']
		tm = get_tensor_channel_map_from_args(args)
		assert(len(tm) == len(semantics['input_tensor_map']))
		for key in tm:
			assert(tm[key] == semantics['input_tensor_map'][key])

	if 'input_annotations' in semantics:
		args.annotations = semantics['input_annotations']

	args.input_symbols = semantics['input_symbols']
	args.labels = semantics['output_labels']

	if 'data_dir' in semantics:
		args.data_dir = semantics['data_dir']

	weight_path_hd5 = os.path.join(os.path.dirname(semantics_json), semantics['architecture'])
	print('Loading keras weight file from:', weight_path_hd5)
	model = load_model(weight_path_hd5, custom_objects=get_metric_dict(args.labels))
	model.summary()
	return model


################################################
###### High-Level Image-Making Functions #######
################################################

def write_filters(args, model):
	#
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	exclude = [args.annotation_set, 'dropout', 'flatten', 'activation', 'batch_normalization', 'concatenate']

	for layer in model.layers:
		if not any([l in layer.name for l in args.layers]):
			continue
		
		for filter_index in range(0, num_layer_channels(layer), args.fps):
			print("Layer name:", layer.name, "filter index:", filter_index)
			if 'dense' in layer.name or 'softmax' in layer.name:
				iterate = iterate_softmax(args, model, layer_dict, layer.name, filter_index)
			else:
				iterate = iterate_channel(args, model, layer_dict, layer.name, filter_index)
	
			expand_dim_shape = (1,)+tensor_shape_from_args(args)
			read_tensor = np.random.random(expand_dim_shape)
			annos = np.random.random((1, len(args.annotations)))
			
			if os.path.exists(args.tensor_example):
				with h5py.File(args.tensor_example, 'r') as hf:
					read_tensor[0] = np.array(hf.get(args.tensor_name))
					annos[0] = np.array(hf.get(args.annotation_set))
				out_file = args.output_dir + '%s/%s/write_%s_filter_%d.hd5' % (plain_name(args.semantics_json), plain_name(args.tensor_example), layer.name, filter_index)
			else:
				out_file = args.output_dir + '%s/random/write_filters/%s_filter_%d.hd5' % (plain_name(args.semantics_json), layer.name, filter_index)

			# run gradient ascent
			for i in range(args.iterations):
				#print("predictions:", model.predict([read_tensor, annos])[0])
				random_jitter = args.jitter * (np.random.random(expand_dim_shape) - 0.5)
				read_tensor += random_jitter
				loss_value, grads_value = iterate([read_tensor, annos])
				read_tensor -= random_jitter

				read_tensor += args.learning_rate*grads_value
				if i % (max(args.iterations,4)//4) == 0:
					print("After iteration:", i, "of:", args.iterations, "loss is:", loss_value," layer name:", layer.name, "filter index:", filter_index)
			
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			with h5py.File(out_file, 'w') as hf:
				hf.create_dataset(args.tensor_name, data=read_tensor[0], compression='gzip')
			print('Wrote tensor to:', out_file)


#def write_saliency(args, model):
			


def excite_neuron(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	
	target_layer_name = 'conv2d_1'
	neuron = [15, 15]

	iterate = iterate_neuron(args, model, layer_dict, neuron, target_layer_name)

	expand_dim_shape = (1,)+tensor_shape_from_args(args)
	read_tensor = np.random.random(expand_dim_shape) 
	if os.path.exists(args.tensor_example):
		with h5py.File(args.tensor_example, 'r') as hf:
			read_tensor[0] = np.array(hf.get(args.tensor_name))
		out_file = args.output_dir + '%s/%s/write_%s.hd5' % (plain_name(args.weights_hd5), plain_name(args.tensor_example), target_layer_name)
	else:
		out_file = args.output_dir + '%s/random/layer_%s.hd5' % (plain_name(args.weights_hd5), target_layer_name)

	for i in range(args.iterations):
		random_jitter = args.jitter * (np.random.random(expand_dim_shape) - 0.5)
		read_tensor += random_jitter
		loss_value, grads_value = iterate([read_tensor])
		read_tensor -= random_jitter

		read_tensor += args.learning_rate*grads_value
		if i % 4 == 0:
			print("After iteration:", i, "of:", args.iterations, "loss is:", loss_value," layer name:", target_layer_name)
	
	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))
	with h5py.File(out_file, 'w') as hf:
		hf.create_dataset(args.tensor_name, data=read_tensor[0], compression='gzip')
	print('Wrote tensor to:', out_file)
					

########################################
###### Gradient & Loss Functions #######
########################################

def iterate_channel(args, model, layer_dict, layer_name='conv5_1', channel=0):
	input_tensor = model.input[0]
	if K.image_data_format()== 'channels_first':
		x = layer_dict[layer_name].output[:,channel,:,:]
	else:
		x = layer_dict[layer_name].output[:,:,:,channel]

	shape = layer_dict[layer_name].output_shape
	w = x.shape[1]
	h = x.shape[2]

	objective = K.variable(0.)

	objective += args.activity_weight* K.sum(K.square(x[:, 2: w-2, 2:h-2])) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor) #/ np.prod(x.shape[1:])
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor))# / np.prod(x.shape[1:])
	
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def iterate_neuron(args, model, layer_dict, neuron, layer_name='conv5_1'):
	input_tensor = model.input[0]

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered

	if K.image_data_format()== 'channels_first':
		x = layer_dict[layer_name].output[:,:, neuron[0],neuron[1]]			
	else:
		x = layer_dict[layer_name].output[:,neuron[0],neuron[1], :]			
	
	shape = layer_dict[layer_name].output_shape

	objective = K.variable(0.)

	objective += args.activity_weight* K.sum(K.square(x)) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective += args.total_variation * total_variation_norm(input_tensor)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective += args.l2 * K.sum(K.square(input_tensor))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def iterate_layer(args, model, layer_dict, layer_name):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered

	x = layer_dict[layer_name].output
	w = x.shape[1]
	h = x.shape[2]
	shape = layer_dict[layer_name].output_shape

	objective = K.variable(0.)

	objective += args.activity_weight* K.sum(K.square(x[:, 2: w-2, 2:h-2])) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def iterate_softmax(args, model, layer_dict, layer_name, neuron):
	input_tensor = model.inputs[0]
	annos = model.inputs[1]

	x = layer_dict[layer_name].output[:, neuron]

	objective = K.variable(0.)

	objective += args.activity_weight*K.sum(K.square(x)) 

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor, annos], [objective, grads])
	return iterate


def grad_towards_input(args, model, desired_input, layer_dict, layer_name='conv5_1'):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	x = layer_dict[layer_name].output
	shape = layer_dict[layer_name].output_shape

	objective = K.variable(0.)

	objective -= args.activity_weight*K.sum(K.square(desired_input-input_tensor)) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor) / np.prod(shape[1:])
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * (K.sum(K.square(input_tensor)) / np.prod(shape[1:]))
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def net_grad_towards_input(model, desired_input, layer_dict, layer_name='conv5_1'):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	loss = K.variable(0.)
	# we avoid border artifacts by only involving non-border pixels in the loss
	loss_weight_activity = 1.0
	loss_weight_continuity = 0.0
	loss_weight_l2 = 0.0

	loss -= loss_weight_activity*K.sum(K.square(desired_input-input_tensor)) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	loss += loss_weight_continuity * total_variation_norm(input_tensor) / np.prod(shape[1:])
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	loss += loss_weight_l2 * (K.sum(K.square(input_tensor)) / np.prod(shape[1:]))
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, model.trainable_weights)[0]

	print('grads shape is:', grads.output_shape)
	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([model], [loss, grads])
	return iterate


def dream_fxn(model, target_layer_dict):
	K.set_learning_phase(0)

	# Build the InceptionV3 network with our placeholder.
	# The model will be loaded with pre-trained ImageNet weights.

	dream = model.input
	print('Model loaded.')

	# Get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	# Define the loss.
	loss = K.variable(0.)
	for layer_name in target_layer_dict:
		# Add the L2 norm of the features of a layer to the loss.
		assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
		coeff = target_layer_dict[layer_name]
		x = layer_dict[layer_name].output
		# We avoid border artifacts by only involving non-border pixels in the loss.
		scaling = K.prod(K.cast(K.shape(x), 'float32'))
		if K.image_data_format() == 'channels_first':
			loss += coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
		else:
			loss += coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

	# Compute the gradients of the dream wrt the loss.
	grads = K.gradients(loss, dream)[0]
	# Normalize gradients.
	grads /= K.maximum(K.mean(K.abs(grads)), 1e-7)

	# Set up function to retrieve the value
	# of the loss and gradients given an input image.
	outputs = [loss, grads]
	return K.function([dream], outputs)


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


##############################
###### Image Utilities #######
##############################
def num_layer_channels(layer):
	if len(layer.output_shape) < 4:
		return layer.output_shape[-1]
	elif K.image_data_format()== 'channels_first':
		return layer.output_shape[1]
	else:
		return layer.output_shape[3]

def preprocess_image(image_path):
	# Util function to open, resize and format pictures
	# into appropriate tensors.
	img = load_img(image_path)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img


def deprocess_image(x, convert_bgr2rgb=True):
	# Util function to convert a tensor into a valid image.
	if K.image_data_format() == 'channels_first':
		x = x.reshape((3, x.shape[2], x.shape[3]))
		x = x.transpose((1, 2, 0))
	else:
		x = x.reshape((x.shape[1], x.shape[2], 3))
	x /= 2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	if convert_bgr2rgb:
		x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
	return x


def normalize_pixels(x):
	x /= 255.
	x -= 0.5
	x *= 2.
	return x


def cv2_image_load(args, image_path, expand_dims=True):
	im = cv2.resize(cv2.imread(image_path), (args.width, args.height))
	
	if K.image_data_format() == 'channels_first':
		im = im.transpose((2,0,1))

	im = np.array(im, np.float32)
	im /= 255.0
	im -= 0.5
	im *= 2.0	
	
	if expand_dims:
		im = np.expand_dims(im, axis=0)

	return im


def resize_img(img, size):
	img = np.copy(img)
	if K.image_data_format() == 'channels_first':
		factors = (1, 1,
				   float(size[0]) / img.shape[2],
				   float(size[1]) / img.shape[3])
	else:
		factors = (1,
				   float(size[0]) / img.shape[1],
				   float(size[1]) / img.shape[2],
				   1)
	return scipy.ndimage.zoom(img, factors, order=1)



def write_dream_video(args, model):
	#args.video_writer = get_video_writer(args)
	deep_dream2(args, model)
	#args.video_writer.release()
	print('wrote dream video')


def get_video_writer(args):
	# Define the codec and create VideoWriter object
	#fourcc = cv2.VideoWriter_fourcc(*'XVID')
	fourcc = cv2.VideoWriter_fourcc('P','I','M','1')
	return cv2.VideoWriter(args.video_path, fourcc, args.fps, (args.width, args.height))




def get_tensor_channel_map_from_args(args):
	'''Return tensor mapping dict given args.tensor_name'''
	if not args.tensor_name:
		return None

	if 'read_tensor' == args.tensor_name:
		return get_tensor_channel_map_rt()
	elif '2d_2bit' == args.tensor_name:
		return get_tensor_channel_map_2bit()
	elif '1d_calling'== args.tensor_name:
		return get_tensor_channel_map_reference_reads()
	elif '2d' == args.tensor_name or '2d_annotations' == args.tensor_name or '2d_mapping_quality' == args.tensor_name:
		return get_tensor_channel_map_mq()
	elif 'reference' == args.tensor_name or '1d_dna' == args.tensor_name or '1d_annotations' == args.tensor_name:
		return get_tensor_channel_map_1d_dna()
	elif 'bqsr' == args.tensor_name:
		return bqsr_tensor_channel_map()
	elif 'mlp' == args.tensor_name:
		return annotations
	elif 'deep_variant' == args.tensor_name:
		return deep_variant_channel_map()
	else:
		raise ValueError('Unknown tensor mapping mode:', args.tensor_name)


def get_tensor_channel_map_1d_dna():
	'''1D Reference tensor with 4 channel DNA encoding.'''
	tensor_map = {}
	for k in DNA_SYMBOLS.keys():
		tensor_map[k] = DNA_SYMBOLS[k]
	
	return tensor_map


def get_tensor_channel_map_1d():
	'''1D Reference tensor with 4 channel DNA encoding'''
	tensor_map = {}
	for k in DNA_SYMBOLS.keys():
		tensor_map[k] = DNA_SYMBOLS[k]
	
	return tensor_map



def get_tensor_channel_map_reference_reads():
	'''Read and reference tensor with 4 channel DNA encoding.
	Plus insertions and deletions.
	'''
	tensor_map = {}
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['read_'+k] = DNA_INDEL_SYMBOLS[k]
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['reference_'+k] = len(DNA_INDEL_SYMBOLS) + DNA_INDEL_SYMBOLS[k]	
	
	return tensor_map

def get_tensor_channel_map():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags.
	'''
	tensor_map = {}
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['read_'+k] = DNA_INDEL_SYMBOLS[k]
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['reference_'+k] = len(DNA_INDEL_SYMBOLS) + DNA_INDEL_SYMBOLS[k]			
	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13
	return tensor_map


def get_tensor_channel_map_mq():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags.
	'''
	tensor_map = {}
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['read_'+k] = DNA_INDEL_SYMBOLS[k]
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['reference_'+k] = len(DNA_INDEL_SYMBOLS) + DNA_INDEL_SYMBOLS[k]			

	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13
	tensor_map['flag_bit_9'] = 14	
	tensor_map['flag_bit_10'] = 15

	tensor_map['mapping_quality'] = 16

	return tensor_map


def get_tensor_channel_map_rt():
	'''Read and reference tensor with 4 channel DNA encoding.
	Also includes read flags for strand and pair.
	'''
	tensor_map = {}
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['read_'+k] = DNA_INDEL_SYMBOLS[k]
	for k in DNA_INDEL_SYMBOLS.keys():
		tensor_map['reference_'+k] = len(DNA_INDEL_SYMBOLS) + DNA_INDEL_SYMBOLS[k]			

	tensor_map['flag_bit_4'] = 10
	tensor_map['flag_bit_5'] = 11	
	tensor_map['flag_bit_6'] = 12	
	tensor_map['flag_bit_7'] = 13

	tensor_map['mapping_quality'] = 14

	return tensor_map


def get_tensor_channel_map_2bit():
	'''Read and reference tensor with 2bit DNA encoding.
	Also includes read flags.
	'''	
	tensor_map = {}
	tensor_map['read_purine'] = 0
	tensor_map['read_pair'] = 1
	tensor_map['read_indel'] = 2
	tensor_map['reference_purine'] = 3
	tensor_map['reference_pair'] = 4
	tensor_map['reference_indel'] = 5	
	tensor_map['flag_bit_4'] = 6
	tensor_map['flag_bit_5'] = 7	
	tensor_map['flag_bit_6'] = 8	
	tensor_map['flag_bit_7'] = 9
	return tensor_map


def tensor_shape_from_args(args):
	in_channels = total_input_channels_from_args(args)
	if args.channels_last:
		tensor_shape = (args.read_limit, args.window_size, in_channels)
	else:
		tensor_shape = (in_channels, args.read_limit, args.window_size) 
	return tensor_shape


def deep_variant_channel_map():
	tensor_map = {}
	tensor_map['bases'] = 0
	tensor_map['reference'] = 1
	tensor_map['strand'] = 2
	return tensor_map


def total_input_channels_from_args(args):
	'''Get the number of channels in the tensor map'''		
	return len(get_tensor_channel_map_from_args(args))


def get_reference_and_read_channels(args):
	'''Get the number of read and reference channels in the tensor map'''		
	count = 0
	tm = get_tensor_channel_map_from_args(args)
	for k in tm.keys():
		if 'read' in k or 'reference' in k:
			count += 1
	return count



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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


def get_metric_dict(labels=SNP_INDEL_LABELS):
	metrics = {'precision':precision, 'recall':recall}
	precision_fxns = per_class_precision(labels)
	recall_fxns = per_class_recall(labels)
	for i,label_key in enumerate(labels.keys()):
		metrics[label_key+'_precision'] = precision_fxns[i]
		metrics[label_key+'_recall'] = recall_fxns[i]

	return metrics


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


##############################
###### Utilities #############
##############################

def plain_name(full_name):
	name = os.path.basename(full_name)
	name = name.split('.')[0]
	return name


if __name__ == '__main__':
	run() # Back to the top!