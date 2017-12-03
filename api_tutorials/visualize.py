# November 2017
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import cv2
import h5py
import time
import scipy
import argparse
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

from keras import metrics
from vgg_16 import vgg_16
from scipy.misc import imsave
from keras import backend as K
from keras.applications import inception_v3
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Convolution2D, Input, ZeroPadding2D, MaxPooling2D

data_root = '/home/sam/Dropbox/'
#data_root = '/Users/sam/Dropbox/'

vgg_weights_tf ='/home/sam/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
inception_weights_tf = '/home/sam/weights/inception_v3_weights_tf_dim_ordering_tf_kernels.h5'
vgg_weights = data_root + 'Code/python/cnn/saved_networks/vgg16_weights_th_dim_ordering_th_kernels.h5'
inception_weights = data_root + 'Code/python/cnn/saved_networks/inception_v3_weights_th_dim_ordering_th_kernels.h5'


def run():
	'''Parse arguments, create a model and dispatch on mode'''
	args = parse_args()
	model = model_from_args(args)

	if 'write_filters' == args.mode:
		write_filters(args, model)
	elif 'excite_neuron' == args.mode:
		excite_neuron(args, model)
	elif 'excite_layer' == args.mode:
		excite_layer(args, model)		
	elif 'deep_dream' == args.mode:
		deep_dream(args, model)		
	elif 'recover' == args.mode:
		recover_image(args, model)			
	elif 'draw' == args.mode:
		draw_loop(args, model)
	elif 'journey' == args.mode:
		image_journey(args, model)	
	elif 'write_video' == args.mode:
		write_dream_video(args, model)		
	else:
		print('unknown visualize mode:', args.mode)


def parse_args():
	parser = argparse.ArgumentParser()

	# Required mode argument: What type of image would you like to make?
	parser.add_argument('mode')

	parser.add_argument('--architecture', default='my_inception.hd5')
	parser.add_argument('--weights', default=inception_weights_tf)
	parser.add_argument('--model', default='inception')
	parser.add_argument('--labels', default=1000, type=int)
	parser.add_argument('--width', default=512, type=int)
	parser.add_argument('--height', default=512, type=int)
	parser.add_argument('--channels', default=3, type=int)
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--iterations', default=30, type=int)
	parser.add_argument('--data_path', default=data_root+'/Photos/')		
	parser.add_argument('--image_path_2', default=data_root+'/Photos/cat.jpg')	
	parser.add_argument('--image_path', default=data_root+'/Photos/dog.jpg')	
	parser.add_argument('--save_path', default=data_root+'/Video/activations/frames/')
	parser.add_argument('--video_path', default=data_root+'/Video/activations/cat_viz5.mpg')
	parser.add_argument('--video_writer', default=None)
	parser.add_argument('--maxfun', default=9, type=int)
	parser.add_argument('--fps', default=6, type=int)
	parser.add_argument('--input_shape', default=None)
	parser.add_argument('--learning_rate', default=0.01, type=float)
	parser.add_argument('--jitter', default=0.0, type=float)
	parser.add_argument('--l2', default=0.0, type=float)
	parser.add_argument('--l1', default=0.0, type=float)
	parser.add_argument('--activity_weight', default=1.0, type=float)
	parser.add_argument('--total_variation', default=0.00001, type=float)

	args = parser.parse_args()
	print('Arguments are', args)	
	return args


def model_from_args(args):
	'''Return a model as specified by the command line/ default arguments'''
	K.set_learning_phase(0)

	if K.image_data_format()== 'channels_first':
		args.input_shape = (args.channels, args.height, args.width)
		input_image = Input(shape=args.input_shape, name='input_image')
	else:
		args.input_shape = (args.height, args.width, args.channels)
		input_image = Input(shape=args.input_shape, name='input_image')		
	
	if 'inception' == args.model:
		model = inception_v3.InceptionV3(weights='imagenet', input_shape=args.input_shape, include_top=False)
	elif 'vgg' == args.model:
		model = vgg_16(args.labels, args.weights, input_image)
	else:
		print('\n\nError: unknown model architecture:', args.model)

	model.summary()
	return model


def draw_loop(args, model):
	'''Interactive window for debugging'''
	cur_img = cv2.imread(args.image_path)

	while True:
		canvas = np.zeros((1090, 1090, 3), np.uint8)
		canvas[:cur_img.shape[0],:cur_img.shape[1],:] = cur_img
		cv2.imshow('Canvas', canvas)

		char = cv2.waitKey(1) & 0xFF
		if char > 31 and char < 127: # Ascii text
			pass 
		elif char == 27: # Escape
			break
		elif char == 8: # Delete
			pass
		elif char == 10: # Enter/return key
				args.video_writer = get_video_writer(args)
				deep_dream2(args, model, canvas)
				args.video_writer.release()
		elif char != 255:
			print('special char:', char)


################################################
###### High-Level Image Making Functions #######
################################################

def write_filters(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	for filter_index in range(2, 25, 4):
		exclude = ['input','zero','fc','flatten','predictions', 'activation', 'batch_normalization']
		for layer in model.layers:
			if any([ex in layer.name for ex in exclude]):
				continue
			print(" layer name:", layer.name, "filter index:", filter_index)

			iterate = iterate_channel(args, model, layer_dict, layer.name, filter_index)
			
			if os.path.exists(args.image_path):
				input_img_data = cv2_image_load(args, args.image_path)
				out_file = args.save_path + '%s/%s/%s/filter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), layer.name, filter_index)
			else:
				if K.image_data_format()== 'channels_first':
					input_img_data = np.random.random((1, 3, args.width, args.height))
				else:
					input_img_data = np.random.random((1, args.width, args.height, 3)) 
				out_file = args.save_path + '%s/random_%s_filter_%d.png' % (plain_name(args.weights), layer.name, filter_index)

			# run gradient ascent
			for i in range(args.iterations):
				random_jitter = args.jitter * (np.random.random(args.input_shape) - 0.5)
				input_img_data += random_jitter
				loss_value, grads_value = iterate([input_img_data])
				input_img_data -= random_jitter

				input_img_data += args.learning_rate*grads_value
				if i % 4 == 0:
					print("After iteration:", i, "loss is:", loss_value," layer name:", layer.name, "filter index:", filter_index)
			
			img = deprocess_image(input_img_data)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			print('Saving image to:\n', out_file)
			imsave(out_file, img)


def excite_neuron(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	
	target_layer_name = 'conv2d_42'
	neuron = [15, 15]

	iterate = iterate_neuron(args, model, layer_dict, neuron, target_layer_name)

	if os.path.exists(args.image_path):
		input_img_data = cv2_image_load(args, args.image_path)
	else:
		if K.image_data_format()== 'channels_first':
			input_img_data = np.random.random((1, 3, args.width, args.height))
		else:
			input_img_data = np.random.random((1, args.width, args.height, 3)) 

	# run gradient ascent
	for i in range(args.iterations):
		random_jitter = args.jitter * (np.random.random(args.input_shape) - 0.5)
		input_img_data += random_jitter
		loss_value, grads_value = iterate([input_img_data])
		input_img_data -= random_jitter

		input_img_data += args.learning_rate*grads_value
		if i % args.fps == 0:
			print("After iteration:", i, "loss is:", loss_value)
	
			img = deprocess_image(input_img_data.copy())
			out_file = args.save_path + '%s/%s/%s/iter_%d_neuron_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i, neuron[0])
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)
			print("After iteration:", i, "loss is:", loss_value, '\n\nImage saved at:', out_file)


def excite_layer(args, model, target_layer_name='block5_conv2'):
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)	

	iterate = iterate_channel(args, model, layer_dict, target_layer_name)

	if os.path.exists(args.image_path):
		im = cv2.resize(cv2.imread(args.image_path ), (args.width, args.height))
		im = np.array(im.transpose((2,0,1)), np.float32)
		im -= im.mean()
		im /= (im.std() + 1e-5)
		input_img_data = np.expand_dims(im, axis=0)
	else:
		input_img_data = np.random.random((1, args.channels, args.width, args.height))

	# run gradient ascent
	for i in range(args.iterations):
		random_jitter = 0.1 * (np.random.random(img_size) - 0.5)
		input_img_data += random_jitter
		loss_value, grads_value = iterate([input_img_data, 1])
		input_img_data -= random_jitter
		input_img_data += grads_value * args.learning_rate

		if i % args.fps == args.fps-1:
			print("After iteration:", i, "loss is:", loss_value)
			img = input_img_data[0]
			img = deprocess_image(img)
			out_file = args.save_path + '%s/%s/%s/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)


def recover_image(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)
	target_layer_name = 'block3_conv2'#'conv2d_27'

	im = cv2_image_load(args, args.image_path)

	iterate = net_grad_towards_input(model, im, layer_dict, target_layer_name)
	im2 = cv2_image_load(args, args.image_path_2)
	input_img_data = np.expand_dims(im2, axis=0)

	# run gradient ascent
	lr = 0.2
	for i in range(args.iterations):
		random_jitter = 0.0 * (np.random.random(img_size) - 0.5)
		input_img_data += random_jitter
		loss_value, grads_value = iterate([input_img_data])
		input_img_data -= random_jitter
		model.trainable_weights += grads_value * lr
		#input_img_data += grads_value * lr
		if i % args.fps == args.fps-1:
			img = input_img_data[0]
			img = deprocess_image(img)
			out_file = args.save_path + '%s/%s/%s/recover/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)
			print("After iteration:", i, "loss is:", loss_value, 'out file:', out_file)



def image_journey(args, model,):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)
	target_layer_name = 'block4_conv3'#'conv2d_27'
	
	im = cv2_image_load(args, args.image_path)
	im2 = cv2_image_load(args, args.image_path_2)
	input_img_data = np.expand_dims(im2, axis=0)

	iterate_in = grad_towards_input(model, im, layer_dict, target_layer_name)
	iterate_in2 = grad_towards_input(model, im2, layer_dict, target_layer_name)
	iterate_dream = iterate_fxn(model, layer_dict, target_layer_name)
	iterate_dream2 = iterate_fxn(model, layer_dict, 'block5_conv2')
	iterate_dream3 = iterate_fxn(model, layer_dict, 'block5_conv3')

	# run gradient ascent
	outer_loops = 12
	counter = 0
	for loop in range(outer_loops):
		jitter_size = 0.01
		lr = 0.15
		for i in range(args.iterations):
			random_jitter = jitter_size * (np.random.random(img_size) - 0.5)
			input_img_data += random_jitter
			
			if loop % 5 == 0:
				lr = 0.01
				jitter_size = 0.01
				loss_value, grads_value = iterate_dream2([input_img_data, 1])
			elif loop % 5 == 1:
				lr = 0.02
				jitter_size = 0.01

				loss_value, grads_value = iterate_dream3([input_img_data, 1])
			elif loop % 5 == 2:
				lr = 1.7
				jitter_size = 0.0001
				loss_value, grads_value = iterate_in([input_img_data, 1])
			elif loop % 5 == 3:
				lr = 0.05
				jitter_size = 0.01
				loss_value, grads_value = iterate_dream([input_img_data, 1])
			else:
				lr = 1.8
				jitter_size = 0.0001				
				loss_value, grads_value = iterate_in2([input_img_data, 1])

			input_img_data -= random_jitter
			input_img_data += grads_value * lr

			if i % args.fps == args.fps-1:
				img = input_img_data[0]
				img = deprocess_image(img)
				out_file = args.save_path + '%s/%s/%s/recover/loop%d_counter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, loop, counter)
				if not os.path.exists(os.path.dirname(out_file)):
					os.makedirs(os.path.dirname(out_file))
				imsave(out_file, img)
				print("Iteration:", i, "loss:", loss_value, "frames:", counter, 'saved_at:', out_file)
				counter += 1
	

def deep_dream(args, model):
	"""Process:
	- Load the original image.
	- Define a number of processing scales (i.e. image shapes),
		from smallest to largest.
	- Resize the original image to the smallest scale.
	- For every scale, starting with the smallest (i.e. current one):
		- Run gradient ascent
		- Upscale image to the next scale
		- Reinject the detail that was lost at upscaling time
	- Stop when we are back to the original size.
	To obtain the detail lost during upscaling, we simply
	take the original image, shrink it down, upscale it,
	and compare the result to the (resized) original image.
	"""

	# Playing with these hyperparameters will also allow you to achieve new effects
	step = 0.01  # Gradient ascent step size
	num_octave = 3  # Number of scales at which to run gradient ascent
	octave_scale = 1.2  # Size ratio between scales
	iterations = 10  # Number of ascent steps per scale
	max_loss = 15.

	K.set_learning_phase(0)
	model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
	model.summary()

	k_fxn = dream_fxn(model)

	img = preprocess_image(base_image_path)
	if K.image_data_format() == 'channels_first':
		original_shape = img.shape[2:]
	else:
		original_shape = img.shape[1:3]
	successive_shapes = [original_shape]
	for i in range(1, num_octave):
		shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
		successive_shapes.append(shape)
	successive_shapes = successive_shapes[::-1]
	original_img = np.copy(img)
	shrunk_original_img = resize_img(img, successive_shapes[0])

	for shape in successive_shapes:
		print('Processing image shape', shape)
		img = resize_img(img, shape)
		img = gradient_ascent(img, k_fxn,
							  iterations=iterations,
							  step=step,
							  max_loss=max_loss)
		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
		same_size_original = resize_img(original_img, shape)
		lost_detail = same_size_original - upscaled_shrunk_original_img

		img += lost_detail
		shrunk_original_img = resize_img(original_img, shape)

	save_img(img, fname=result_prefix + '.png')



def eval_loss_and_grads(k_fxn, x):
	outs = k_fxn([x])
	loss_value = outs[0]
	grad_values = outs[1]
	return loss_value, grad_values

def gradient_ascent(x, k_fxn, iterations, step, max_loss=None):
	for i in range(iterations):
		loss_value, grad_values = eval_loss_and_grads(k_fxn, x)
		if max_loss is not None and loss_value > max_loss:
			break
		print('...Loss value at', i, ':', loss_value)
		x += step * grad_values
	return x


########################################
###### Gradient & Loss Functions #######
########################################

def iterate_channel(args, model, layer_dict, layer_name='conv5_1', channel=0):
	K.set_learning_phase(1)
	input_tensor = model.input
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
	input_tensor = model.input

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


def iterate_softmax(model, neuron):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	print('X shape', model.output[:, neuron])
	x = model.output

	loss_weight_continuity = 0.0
	loss_weight_activity = 1.0

	loss = K.mean(x)
	#loss += loss_weight_continuity * total_variation_norm(input_tensor)

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_tensor)[0]
	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	return K.function([input_tensor], [loss, grads])


def grad_towards_input(model, desired_input, layer_dict, layer_name='conv5_1'):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	x = layer_dict[layer_name].output
	shape = layer_dict[layer_name].output_shape

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
	grads = K.gradients(loss, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor, K.learning_phase()], [loss, grads])
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


def dream_fxn(model):
	K.set_learning_phase(0)

	# Build the InceptionV3 network with our placeholder.
	# The model will be loaded with pre-trained ImageNet weights.

	dream = model.input
	print('Model loaded.')

	# Get the symbolic outputs of each "key" layer (we gave them unique names).
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	# Define the loss.
	loss = K.variable(0.)
	for layer_name in settings['features']:
		# Add the L2 norm of the features of a layer to the loss.
		assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
		coeff = settings['features'][layer_name]
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
def continuity_loss(args, x):
	assert K.ndim(x) == 4
	if K.image_data_format()== 'channels_first':
		a = K.square(x[:, :, :args.width - 1, :args.height - 1] -
					 x[:, :, 1:, :args.height - 1])
		b = K.square(x[:, :, :args.width - 1, :args.height - 1] -
					 x[:, :, :args.width - 1, 1:])
	else:
		a = K.square(x[:, :args.width - 1, :args.height-1, :] -
					 x[:, 1:, :args.height - 1, :])
		b = K.square(x[:, :args.width - 1, :args.height-1, :] -
					 x[:, :args.width - 1, 1:, :])
	return K.sum(K.pow(a + b, 1.25))


def alpha_norm(x, alpha=6, lambdaa=0.05):
	x -= K.mean(x)
	return lambdaa * K.pow(K.sum(x), alpha)


def total_variation_norm(x):
	x -= K.mean(x)
	a = K.square(x[:, :, 1:, :-1] - x[:, :, :-1, :-1])
	b = K.square(x[:, :, :-1, 1:] - x[:, :, :-1, :-1])
	tv = K.sum(K.pow(a + b, 1.25))

	return tv


##############################
###### Image Utilities #######
##############################

def preprocess_image(image_path):
	# Util function to open, resize and format pictures
	# into appropriate tensors.
	img = load_img(image_path)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img


def deprocess_image(x):
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


##############################
###### Utilities #############
##############################

def plain_name(full_name):
	name = os.path.basename(full_name)
	name = name.split('.')[0]
	return name


if __name__ == '__main__':
	run() # Back to the top!