import os
import cv2
import h5py
import time
import argparse
import numpy as np
import scipy.ndimage as nd
from scipy import interpolate
import matplotlib.pyplot as plt

from keras import metrics
from vgg_16 import vgg_16
from scipy.misc import imsave
from keras import backend as K
from keras.models import Sequential
from inception_v3 import inception_v3
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Convolution2D, Input, ZeroPadding2D, MaxPooling2D

data_root = '/Users/sam/Dropbox'
data_root = '/home/sam/Dropbox/'
data_path = data_root+'Code/python/cnn/saved_networks/'

inception_weights = data_path + 'inception_v3_weights_th_dim_ordering_th_kernels.h5'
vgg_weights = data_path + 'vgg16_weights_th_dim_ordering_th_kernels.h5'


def run():
	args = parse_args()

	model = model_from_args(args)

	if 'write_filters' == args.mode:
		write_filters(args, model)
	elif 'excite_neuron' == args.mode:
		excite_neuron(args, model)
	elif 'excite_layer' == args.mode:
		excite_layer(args, model)		
	elif 'deep_dream' == args.mode:
		sam_deep_dream(args, model)		
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

	parser.add_argument('mode')

	parser.add_argument('--weights', default=vgg_weights)
	parser.add_argument('--model', default='vgg')
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
	parser.add_argument('--fps', default=1, type=int)

	args = parser.parse_args()
	print('Arguments are', args)	
	return args



def model_from_args(args):
	input_image = Input(shape=(args.channels, args.height, args.width), name='input_image')

	if 'inception' == args.model:
		model = inception_v3(args.labels, args.weights, input_image)
	elif 'vgg' == args.model:
		model = vgg_16(args.labels, args.weights, input_image)
	else:
		print('\n\nError: unknown model architecture:', args.model)

	return model



def draw_loop(args, model):

	#img = model.get_layer('input_image').output

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


def sam_deep_dream(args, model, octave_n=4, octave_scale=2, target_layer_name='block5_conv3', clip=True, **step_params):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)
	
	for it in range(args.iterations):

		iterate = iterate_fxn(model, layer_dict, target_layer_name)
		if os.path.exists(args.image_path):
			img = cv2.resize(cv2.imread(args.image_path ), (args.width, args.height))
			img = np.array(img.transpose((2,0,1)), np.float32)
			img -= img.mean()
			img /= (img.std() + 1e-5)
			out_file = args.save_path + '%s/%s/%s/iter_%d.png' % (os.path.basename(args.weights), os.path.basename(args.image_path), target_layer_name, it)
		else:
			input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.
			img = np.random.random((args.channels, args.width, args.height)) * 20 + 128.
			out_file = args.save_path + 'excite_softmax_%d.png' % (it)

		print("Base img shape:", img.shape)
		octaves = [np.float32(img)] #np.rollaxis(img, 2)[::-1])]

		for i in xrange(1, octave_n-1):
			octave_inv = 1.0/octave_scale
			print('Adding octave:', i, ' octave inv', octave_inv, "octave img shape:", octaves[-1].shape)
			octaves.append(nd.zoom(octaves[-1], (1, octave_inv, octave_inv), order=1))
			# imgplt = np.rollaxis(octaves[-1], 0, 3)
			# plt.imshow(imgplt)
			# plt.show()
		
		detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
		
		for octave, octave_base in enumerate(octaves[::-1]):
			h, w = octave_base.shape[-2:]
			if octave > 0:
				h1, w1 = detail.shape[-2:]
				detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

			# run gradient ascent
			lr = 0.3
			last_x = octave_base+detail
			#print 'cur x shape is', last_x.shape
			x = last_x.transpose((1, 2, 0))
			#print 'transposed x shape is', x.shape
			x = cv2.resize(x, (args.width, args.height))
			#print 'resized x shape is', x.shape
			x = x.transpose((2, 0, 1))
			#print 'cur x shape is', x.shape

			input_img_data = np.expand_dims(x, axis=0)
			#print 'after trnaspose x shape is', x.shape			
			for i in range(args.iterations):
				random_jitter = 0.1 * (np.random.random(input_img_data.shape) - 0.5)
				input_img_data += random_jitter
				loss_value, grads_value = iterate([input_img_data, 1])
				input_img_data -= random_jitter
				input_img_data += grads_value * lr
				if i % 4 == 0:
					print("After iteration:", i, "loss is:", loss_value, "filter index:", it)
			
			detail = last_x-octave_base
		img = input_img_data[0]
		img = deprocess_image(img)
		imsave(out_file, img)


def excite_neuron(args, model, neuron=10):
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)
	target_layer_name ='predictions'# 'block5_conv3'# 

	iterate = iterate_neuron(model, layer_dict, neuron, target_layer_name)

	if os.path.exists(args.image_path):
		im = cv2.resize(cv2.imread(args.image_path ), (args.width, args.height))
		im = np.array(im.transpose((2,0,1)), np.float32)
		im -= im.mean()
		im /= (im.std() + 1e-5)
		input_img_data = np.expand_dims(im, axis=0)
	else:
		input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.

	# run gradient ascent
	lr = 0.01
	for i in range(args.iterations):
		random_jitter = 0.1 * (np.random.random(img_size) - 0.5)
		input_img_data += random_jitter
		loss_value, grads_value = iterate([input_img_data])
		input_img_data -= random_jitter
		input_img_data += grads_value * lr
		if i % args.fps == args.fps-1:
	
			img = input_img_data[0]
			img = deprocess_image(img)
			out_file = args.save_path + '%s/%s/%s/iter_%d_neuron_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i, neuron)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)
			print("After iteration:", i, "loss is:", loss_value, '\n\nImage saved at:', out_file)


def excite_layer(args, model, target_layer_name='block5_conv2'):
	
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)	

	iterate = iterate_fxn(model, layer_dict, target_layer_name)

	if os.path.exists(args.image_path):
		im = cv2.resize(cv2.imread(args.image_path ), (args.width, args.height))
		im = np.array(im.transpose((2,0,1)), np.float32)
		im -= im.mean()
		im /= (im.std() + 1e-5)
		input_img_data = np.expand_dims(im, axis=0)
	else:
		input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.

	# run gradient ascent
	lr = 0.01
	for i in range(args.iterations):
		random_jitter = 0.1 * (np.random.random(img_size) - 0.5)
		input_img_data += random_jitter
		loss_value, grads_value = iterate([input_img_data, 1])
		input_img_data -= random_jitter
		input_img_data += grads_value * lr
		if i % args.fps == args.fps-1:
			print ("After iteration:", i, "loss is:", loss_value)
	
			img = input_img_data[0]
			img = deprocess_image(img)
			out_file = args.save_path + '%s/%s/%s/iter_%d.png' % (plain_name(args.weights), plain_name(args.image_path), target_layer_name, i)
			if not os.path.exists(os.path.dirname(out_file)):
				os.makedirs(os.path.dirname(out_file))
			imsave(out_file, img)



def cv2_image_load(args, image_path):
	im = cv2.resize(cv2.imread(image_path ), (args.width, args.height))
	im = np.array(im.transpose((2,0,1)), np.float32)
	#im -= im.mean()
	im /= (im.std() + 1e-5)	
	return im

def recover_image(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	img_size = (args.channels, args.height, args.width)
	target_layer_name = 'block3_conv2'#'conv2d_27'

	im = cv2_image_load(args, args.image_path)

	iterate = net_grad_towards_input(model, im, layer_dict, target_layer_name)
	im2 = cv2_image_load(args, args.image_path_2)
	input_img_data = np.expand_dims(im2, axis=0)

	#input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.

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


	#input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.

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
		




def write_filters(args, model):
	for filter_index in range(2, 25, 4):
		exclude = ['input','zero','fc','flatten','predictions']
		for layer in model.layers:
			if any([ex in layer.name for ex in exclude]):
				continue
			print(" layer name:", layer.name, "filter index:", filter_index)
			layer_dict = dict([(layer.name, layer) for layer in model.layers])

			iterate = iterate_neuron(model, layer_dict, filter_index, layer.name)
			
			if args.image_path != '':
				im = cv2.resize(cv2.imread(args.image_path), (args.width, args.height))
				im = np.array(im.transpose((2,0,1)), np.float32)
				input_img_data = np.expand_dims(im, axis=0)
				out_file = args.save_path + '%s/%s/%s/filter_%d.png' % (os.path.basename(args.weights), os.path.basename(args.image_path), layer.name, filter_index)
			else:
				input_img_data = np.random.random((1, 3, args.width, args.height)) * 20 + 128.
				out_file = args.save_path + '/random_%s_filter_%d.png' % (layer.name, filter_index)

			# run gradient ascent
			for i in range(args.iterations):
				loss_value, grads_value = iterate([input_img_data, K.learning_phase()])
				input_img_data += grads_value
				if i % 4 == 0:
					print("After iteration:", i, "loss is:", loss_value," layer name:", layer.name, "filter index:", filter_index)
			
			img = input_img_data[0]
			img = deprocess_image(img)
			imsave(out_file, img)


def normalize(x, value):
	"""
	Normalizes the value with respect to image dimensions. This makes regularizer weight factor more or less
	uniform across various input image dimensions.
	Args:
		img: 4D tensor with shape: `(samples, channels, rows, cols)` if dim_ordering='th' or
				`(samples, rows, cols, channels)` if dim_ordering='tf'.
		value: The function to normalize
	Returns:
		The normalized expression.
	"""
	return value / np.prod((3,1024, 768))



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


def iterate_fxn(model, layer_dict, layer_name='conv5_1'):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the layer given by layer name
	x = layer_dict[layer_name].output
	shape = layer_dict[layer_name].output_shape

	loss = K.variable(0.)
	# we avoid border artifacts by only involving non-border pixels in the loss
	loss_weight_activity = 1.0
	loss_weight_continuity = 1.0
	loss_weight_l2 = 1.0

	if K.image_data_format()== 'channels_first':
		#loss -= loss_weight_activity*K.sum(K.square(x[:, :, 2: shape[2] - 2, 2: shape[3] - 2])) / np.prod(shape[1:])
		loss += loss_weight_activity*K.sum(K.square(x)) / np.prod(shape[1:])
	else:
		#loss -= loss_weight_activity*K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / np.prod(shape[1:])
		loss += loss_weight_activity*K.sum(K.square(x)) / np.prod(shape[1:])

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


def iterate_neuron(model, layer_dict, neuron, layer_name='conv5_1'):
	input_tensor = model.input

	# this is a placeholder tensor that will contain our generated images

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	print('X shape', layer_dict[layer_name].output_shape)
	#fc1_in =  layer_dict['fc1'].input
	x = layer_dict[layer_name].output[:,neuron,:,:]

	loss = K.variable(0.)
	loss_weight_activity = 1.0
	loss_weight_continuity = 1.0
	loss_weight_l2 = 0.0

	if K.image_data_format()== 'channels_first':
		# we avoid border artifacts by only involving non-border pixels in the loss
		#loss -= loss_weight_activity*K.sum(K.square(x[:, :, 2: shape[2] - 2, 2: shape[3] - 2])) / np.prod(shape[1:])
		loss += loss_weight_activity*K.sum(K.square(x)) 
	else:
		#loss -= loss_weight_activity*K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / np.prod(shape[1:])
		loss += loss_weight_activity*K.sum(K.square(x))

	# add continuity loss (gives image local coherence, can result in an artful blur)
	#loss += loss_weight_continuity * total_variation_norm(input_tensor)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	#loss -= loss_weight_l2 * (K.sum(K.square(input_tensor)))
	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_tensor)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_tensor], [loss, grads])
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


# util function to convert a tensor into a valid image
def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.3

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	#print 'cur x shape is', x.shape
	x = x.transpose((1, 2, 0))
	#print 'after trnaspose x shape is', x.shape
	x = x[:,:,::-1]
	#print 'after bgr x shape is', x.shape

	#x = np.clip(x, 0, 255).astype('uint8')
	x = x.astype('uint8')
	return x




def preprocess_image(args, image_path):
	img = load_img(image_path, target_size=(args.height, args.width))
	img = img_to_array(img)

	#print 'cur x shape is', img.shape
#	img = img.transpose((1, 2, 0))
	#print 'after transpose x shape is', img.shape
	#img = img[:,:,::-1]
	#print 'after bgr rotate x shape is', img.shape
  
#	img = np.expand_dims(img, axis=0)
	img = normalize_pixels(img)
	return img


def normalize_pixels(x):
	x /= 255.
	#x -= 0.5
	x *= 2.
	return x


def plain_name(full_name):
	name = os.path.basename(full_name)
	name = name.split('.')[0]
	return name

if __name__ == '__main__':
	run()