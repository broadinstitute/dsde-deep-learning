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
from PIL import Image
from scipy import interpolate
import matplotlib.pyplot as plt

from keras import metrics
from vgg_16 import vgg_16
from scipy.misc import imsave
from keras import backend as K
from keras.models import Sequential, load_model
from keras.applications import inception_v3, vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.utils.layer_utils import convert_all_kernels_in_model
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
	elif 'write_video' == args.mode:
		write_dream_video(args, model)
	elif 'frame_journey' == args.mode:
		image_frame_journey(args, model)	
	else:
		print('unknown visualize mode:', args.mode)


def parse_args():
	parser = argparse.ArgumentParser()

	# Required mode argument: What type of image would you like to make?
	parser.add_argument('mode')

	parser.add_argument('--weights', default='')
	parser.add_argument('--model', default='inception')
	parser.add_argument('--id', default='run_id')
	parser.add_argument('--labels', default=1000, type=int)
	parser.add_argument('--width', default=299, type=int)
	parser.add_argument('--height', default=299, type=int)
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
	parser.add_argument('--continuity_loss', default=0.00001, type=float)
	parser.add_argument('--alpha_norm', default=0.0, type=float)
	parser.add_argument('--neuron', default=58, type=int)
	parser.add_argument('--convert_kernels', default=False, action='store_true')
	parser.add_argument('--layers', nargs='+', default=['conv', 'dense'], type=str, help='List of layer name to investigate.')
	parser.add_argument('--images', nargs='+', default=[], type=str, help='List of image paths to load.')
	parser.add_argument('--frames', help='Directory of video frames saved as images.')

	args = parser.parse_args()
	print('Arguments are', args)	
	return args


def model_from_args(args):
	'''Return a model as specified by the command line/ default arguments'''
	K.set_learning_phase(0)

	if K.image_data_format() == 'channels_first':
		args.input_shape = (args.channels, args.height, args.width)
	else:
		args.input_shape = (args.height, args.width, args.channels)
		
	input_image = Input(shape=args.input_shape, name='input_image')		
	
	if os.path.exists(args.weights):
		model = load_model(args.weights)
	elif 'inception' == args.model:
		if args.height == 299 and args.width == 299:
			model = inception_v3.InceptionV3(input_tensor=input_image, input_shape=args.input_shape, include_top=True)
		else:
			model = inception_v3.InceptionV3(input_tensor=input_image, input_shape=args.input_shape, include_top=False, pooling='max')

	elif 'vgg' == args.model:
		if args.height == 224 and args.width == 224:
			model = vgg16.VGG16(input_tensor=input_image, include_top=True)
		else:
			model = vgg16.VGG16(input_tensor=input_image, input_shape=args.input_shape, include_top=False, pooling='max')
	else:
		print('\n\nError: unknown model architecture:', args.model)

	if args.convert_kernels:
		convert_all_kernels_in_model(model)

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
###### High-Level Image-Making Functions #######
################################################

def write_filters(args, model): 
	layer_dict = dict([(layer.name, layer) for layer in model.layers])

	for layer in model.layers:
		if not any([l in layer.name for l in args.layers]):
			continue
		
		for filter_index in range(num_layer_channels(layer)):
			print("Layer name:", layer.name, "filter index:", filter_index)

			if not any([l in layer.name for l in ['global', 'dense', 'softmax', 'predictions']]):
				iterate = iterate_softmax(args, model, layer_dict, layer.name, filter_index)
			else:
				iterate = iterate_channel(args, model, layer_dict, layer.name, filter_index)
	
			if os.path.exists(args.image_path):
				input_img_data = cv2_image_load(args, args.image_path)
				out_file = args.save_path + '%s/%s/%s/filter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), layer.name, filter_index)
			else:
				input_img_data = np.random.random((1,)+args.input_shape) 
				out_file = args.save_path + '%s/random/write_filters/%s_filter_%d.png' % (plain_name(args.weights), layer.name, filter_index)

			# run gradient ascent
			for i in range(args.iterations):
				random_jitter = args.jitter * (np.random.random(args.input_shape) - 0.5)
				input_img_data += random_jitter
				loss_value, grads_value = iterate([input_img_data])
				input_img_data -= random_jitter

				input_img_data += args.learning_rate*grads_value
				if i % (args.iterations//6) == 0:
					print("  After iteration:", i, "loss is:", loss_value," layer name:", layer.name, "filter index:", filter_index)
			
			img = deprocess_image(args, input_img_data)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			print('Saving image to:\n', out_file)
			plt.imsave(out_file, img)
			#pimg = Image.fromarray(img)
			#pimg.save(out_file)


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
	
			img = deprocess_image(args, input_img_data.copy())
			out_file = args.save_path + '%s/%s/%s/iter_%d_neuron_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i, neuron[0])
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			#imsave(out_file, img)
			plt.imsave(out_file, img)
			print("After iteration:", i, "loss is:", loss_value, '\n\nImage saved at:', out_file)


def excite_layer(args, model, target_layer_name='conv2d_42'):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	iterate = iterate_layer(args, model, layer_dict, target_layer_name)

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
		loss_value, grads_value = iterate([input_img_data, 1])
		input_img_data -= random_jitter
		input_img_data += grads_value * args.learning_rate

		if i % args.fps == args.fps-1:
			print("After iteration:", i, "loss is:", loss_value)
			img = deprocess_image(args, input_img_data.copy())
			out_file = args.save_path + '%s/%s/excite_layer_%s/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)


def excite_softmax(args, model, target_layer_name='predictions'):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	iterate = iterate_softmax(args, model, layer_dict, target_layer_name, args.neuron)

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
		loss_value, grads_value = iterate([input_img_data, 1])
		input_img_data -= random_jitter
		input_img_data += grads_value * args.learning_rate

		if i % args.fps == args.fps-1:
			print("After iteration:", i, "loss is:", loss_value)
			img = deprocess_image(args, input_img_data.copy())
			out_file = args.save_path + '%s/%s/excite_softmax_%s/neuron_%d/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, args.neuron, i)
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
			img = deprocess_image(args, img)
			out_file = args.save_path + '%s/%s/%s/recover/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)
			print("After iteration:", i, "loss is:", loss_value, 'out file:', out_file)



def image_journey(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	
	# run gradient ascent
	objective_switch = 20 
	counter = 0
	
	input_img_data = cv2_image_load(args, images[0])
	for img in args.images[1:]:
		cur_img = cv2_image_load(args, img)
		img_fxn = grad_towards_input(args, model, cur_img, layer_dict)

		for layer in model.layers:
			if not any([l in layer.name for l in args.layers]):
				continue			
			layer_fxn = iterate_layer(args, model, layer_dict, layer.name)
			
			for i in range(args.iterations):

				if i < args.iterations/3:
					loss_value, grads_value = img_fxn([input_img_data])
					lr = args.learning_rate 
				elif i < 2*args.iterations/3:
					random_jitter = args.jitter * (np.random.random(args.input_shape) - 0.5)
					input_img_data += random_jitter
					loss_value, grads_value = layer_fxn([input_img_data])
					lr = args.learning_rate / 10
					input_img_data -= random_jitter
				else:
					loss_value, grads_value = img_fxn([input_img_data])
					lr = args.learning_rate 
				
				input_img_data += lr * grads_value

				if i % args.fps == args.fps-1:
					img = deprocess_image(args, input_img_data.copy())
					out_file = args.save_path + '/image_journey/%s/_counter_%d.png' % (args.id, counter)
					if not os.path.exists(os.path.dirname(out_file)):
						os.makedirs(os.path.dirname(out_file))
					imsave(out_file, img)
					print("Iteration:", i, "loss:", loss_value, "frames:", counter, 'saved_at:', out_file)
					counter += 1
	
def image_frame_journey(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	
	frame = 0
	counter = 0
	dream_step = True
	dream_steps = 600
	image_exts = ['.png', '.jpg']
	images = [os.path.join(args.frames, img) for img in sorted(os.listdir(args.frames)) if os.path.splitext(img)[1] in image_exts]
	input_img_data = cv2_image_load(args, images[0])
	
	for img in images[1:]:
		cur_img = cv2_image_load(args, img)
		img_fxn = grad_towards_input(args, model, cur_img, layer_dict)

		for layer in model.layers:
			if not any([l in layer.name for l in args.layers]):
				continue			
			layer_fxn = iterate_layer(args, model, layer_dict, layer.name)
			
			for i in range(args.iterations):

				if counter%dream_steps == 0:
					dream_step = not dream_step

				if dream_step:
					random_jitter = args.jitter * (np.random.random(args.input_shape) - 0.5)
					input_img_data += random_jitter
					loss_value, grads_value = layer_fxn([input_img_data])
					lr = args.learning_rate / 8
					input_img_data -= random_jitter
				else:
					loss_value, grads_value = img_fxn([input_img_data])
					lr = args.learning_rate 
				
				input_img_data += lr * grads_value

				counter += 1
				if counter % args.fps == 0:
					frame_img = deprocess_image(args, input_img_data.copy())
					out_file = args.save_path + '/image_frame_journey/%s/counter_%d.png' % (args.id, counter)
					if not os.path.exists(os.path.dirname(out_file)):
						os.makedirs(os.path.dirname(out_file))
					imsave(out_file, frame_img)
					print("Loss:", loss_value, "frame:", frame, 'counter', counter, 'saved_at:', out_file, '\nimg:', img)
					frame += 1
	

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

	num_octave = 3  # Number of scales at which to run gradient ascent
	octave_scale = 1.2  # Size ratio between scales
	max_loss = 15.
	target_layers = {'mixed2': 0.2, 'mixed3': 0.3, 'mixed4': 0.2, 'mixed5': 0.4}
	
	model = inception_v3.InceptionV3(weights='imagenet', include_top=False)
	model.summary()
	img = preprocess_image(args.image_path)
	k_fxn = dream_fxn(model, target_layers)

	#img = cv2_image_load(args, args.image_path)

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
							  iterations=args.iterations,
							  step=args.learning_rate,
							  max_loss=max_loss)
		upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
		same_size_original = resize_img(original_img, shape)
		lost_detail = same_size_original - upscaled_shrunk_original_img

		img += lost_detail
		shrunk_original_img = resize_img(original_img, shape)

	out_file = args.save_path + '%s/%s/deep_dream/dreamy.png' % (plain_name(args.weights), plain_name(args.image_path))
	if not os.path.exists(os.path.dirname(out_file)):
		os.makedirs(os.path.dirname(out_file))
	im = deprocess_image(args, img, convert_bgr2rgb=False)
	imsave(out_file, im)



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


def num_layer_channels(layer):
	if len(layer.output_shape) < 4:
		return layer.output_shape[-1]
	elif K.image_data_format()== 'channels_first':
		return layer.output_shape[1]
	else:
		return layer.output_shape[3]



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
	
	w = x.shape[1]
	h = x.shape[2]
	shape = layer_dict[layer_name].output_shape


	objective = K.variable(0.)

	objective += args.activity_weight* K.sum(K.square(x[:, 2: w-2, 2:h-2])) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor) / np.prod(x.shape[1:])
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor)) / np.prod(x.shape[1:])
	
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


def iterate_layer(args, model, layer_dict, layer_name):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	x = layer_dict[layer_name].output

	shape = layer_dict[layer_name].output_shape

	objective = K.variable(0.)
	
	if K.image_data_format()== 'channels_first':
		w = x.shape[2]
		h = x.shape[3]
		objective += args.activity_weight* K.sum(K.square(x[:, :, 2: w-2, 2:h-2])) / np.prod(shape[1:])
	else:
		w = x.shape[1]
		h = x.shape[2]		
		objective += args.activity_weight* K.sum(K.square(x[:, 2: w-2, 2:h-2, :])) / np.prod(shape[1:])
		
	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor) / np.prod(shape[1:])
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor)) / np.prod(shape[1:])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def iterate_softmax(args, model, layer_dict, layer_name, neuron):
	input_tensor = model.input

	x = layer_dict[layer_name].output[:, neuron]

	objective = K.variable(0.)

	objective += args.activity_weight*K.sum(K.square(x)) 

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor)
	objective -= args.alpha_norm * alpha_norm(input_tensor)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * K.sum(K.square(input_tensor))

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(objective, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-6)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [objective, grads])
	return iterate


def grad_towards_input(args, model, desired_input, layer_dict):
	input_tensor = model.input

	objective = K.variable(0.)

	objective -= args.activity_weight*K.sum(K.square(desired_input-input_tensor)) / np.prod(args.input_shape)

	# add continuity loss (gives image local coherence, can result in an artful blur)
	objective -= args.total_variation * total_variation_norm(input_tensor) / np.prod(args.input_shape)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	objective -= args.l2 * (K.sum(K.square(input_tensor)) / np.prod(args.input_shape))
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


def alpha_norm(x, alpha=6):
	x -= K.mean(x)
	return K.pow(K.sum(x), alpha)


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

def preprocess_image(image_path):
	# Util function to open, resize and format pictures
	# into appropriate tensors.
	img = load_img(image_path)
	img = img_to_array(img)
	img = np.expand_dims(img, axis=0)
	img = inception_v3.preprocess_input(img)
	return img


def deprocess_image(args, x, convert_bgr2rgb=True):
	# Util function to convert a tensor into a valid image.

	if K.image_data_format() == 'channels_first':
		if args.channels == 1:
			x = x.reshape((x.shape[2], x.shape[3]))
		else:
			x = x.reshape((args.channels, x.shape[2], x.shape[3]))
			x = x.transpose((1, 2, 0))
	else:
		if args.channels == 1:
			x = x.reshape((x.shape[1], x.shape[2]))
		else:
			x = x.reshape((x.shape[1], x.shape[2], args.channels))
	
	x /= 2.
	x += 0.5
	x *= 255.
	x = np.clip(x, 0, 255).astype('uint8')
	
	if convert_bgr2rgb and args.channels == 3:
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