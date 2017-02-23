import os
import cv2
import h5py
import time
import argparse
import numpy as np
from vgg_16 import vgg_16
from scipy.misc import imsave
from keras import backend as K
from keras.models import Sequential
from inception_v3 import inception_v3
from scipy.optimize import fmin_l_bfgs_b
from keras.preprocessing.image import load_img, img_to_array
from keras.layers import Convolution2D, Input, ZeroPadding2D, MaxPooling2D

#data_root = '/Users/sam/Dropbox'
data_root = '/home/sam/Dropbox/'
data_path = data_root+'Code/python/cnn/saved_networks/'

inception_weights = data_path + 'inception_v3_weights_th_dim_ordering_th_kernels.h5'
vgg_weights = data_path + 'vgg16_weights_th_dim_ordering_th_kernels.h5'


# some settings we found interesting
saved_settings = {
	'bad_trip': {'features': {'block4_conv1': 0.05,
							  'block4_conv2': 0.01,
							  'block4_conv3': 0.01},
				 'continuity': 0.1,
				 'dream_l2': 0.8,
				 'jitter': 5},
	'dreamy': {'features': {'block5_conv1': 0.05,
							'block5_conv2': 0.02,
							'block5_conv3': 0.02},
			   'continuity': 0.2,
			   'dream_l2': 0.02,
			   'jitter': 0},
	'sams': {'features': {'convolution2d_13': 0.05,
						  'convolution2d_11': 0.02},
			   'continuity': 0.1,
			   'dream_l2': 0.02,
			   'jitter': 0},
}


# the settings we will use in this experiment
settings = saved_settings['dreamy']

def run():
	args = parse_args()

	model = model_from_args(args)

	if 'write_filters' == args.mode:
		write_filters(args, model)
	elif 'excite_softmax' == args.mode:
		excite_softmax(args, model)
	elif 'deep_dream' == args.mode:
		deep_dream(args, model)		
	elif 'draw' == args.mode:
		draw_loop(args, model)
	else:
		print 'unknown visualize mode:', args.mode

def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--weights', default='')
	parser.add_argument('--model', default='vgg')
	parser.add_argument('--labels', default=1000, type=int)
	parser.add_argument('--width', default=299, type=int)
	parser.add_argument('--height', default=299, type=int)
	parser.add_argument('--channels', default=3, type=int)
	parser.add_argument('--mode', default='deep_dream')
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--iterations', default=20, type=int)
	parser.add_argument('--image_path', default=data_root+'/Photos/dog.jpg')	
	parser.add_argument('--save_path', default=data_root+'/Photos/activations/')
	parser.add_argument('--video_path', default=data_root+'/Photos/activations/viz.avi')
	parser.add_argument('--video_writer', default=None)
	parser.add_argument('--fps', default=20, type=int)

	args = parser.parse_args()
	print 'Arguments are', args	
	return args


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
			print 'wrote dream video'
		elif char != 255:
			print 'special char:', char


def model_from_args(args):
	input_image = Input(shape=(args.channels, args.height, args.width), name='input_image')

	if 'inception' == args.model:
		model = inception_v3(args.labels, inception_weights, input_image)
	elif 'vgg' == args.model:
		model = vgg_16(args.labels, vgg_weights, input_image)
	else:
		print '\n\nError: unknown model architecture:', args.model	

	return model


def get_video_writer(args):
	# Define the codec and create VideoWriter object
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	return cv2.VideoWriter(args.video_path, fourcc, args.fps, (args.width, args.height))


def get_deep_dream_fxn(args, model):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	dream = model.get_layer('input_image').output

	# build the VGG16 network with our placeholder
	# the model will be loaded with pre-trained ImageNet weights
	#model = vgg16.VGG16(input_tensor=dream, weights=weights_path, include_top=False)
	print('Model from:'+ args.weights)


	print('Model loaded. Layer Names:'+ str(layer_dict.keys()) )

	# define the loss
	loss = K.variable(0.)
	for layer_name in settings['features']:
		# add the L2 norm of the features of a layer to the loss
		assert layer_name in layer_dict.keys(), 'Layer ' + layer_name + ' not found in model.'
		coeff = settings['features'][layer_name]
		x = layer_dict[layer_name].output
		shape = layer_dict[layer_name].output_shape
		# we avoid border artifacts by only involving non-border pixels in the loss
		if K.image_dim_ordering() == 'th':
			loss -= coeff * K.sum(K.square(x[:, :, 2: shape[2] - 2, 2: shape[3] - 2])) / np.prod(shape[1:])
		else:
			loss -= coeff * K.sum(K.square(x[:, 2: shape[1] - 2, 2: shape[2] - 2, :])) / np.prod(shape[1:])

	# add continuity loss (gives image local coherence, can result in an artful blur)
	img_size = (args.channels, args.height, args.width)
	loss += settings['continuity'] * continuity_loss(args, dream) / np.prod(img_size)
	# add image L2 norm to loss (prevents pixels from taking very high values, makes image darker)
	loss += settings['dream_l2'] * K.sum(K.square(dream)) / np.prod(img_size)

	# feel free to further modify the loss as you see fit, to achieve new effects...

	# compute the gradients of the dream wrt the loss
	grads = K.gradients(loss, dream)

	outputs = [loss]
	if type(grads) in {list, tuple}:
		outputs += grads
	else:
		outputs.append(grads)

	f_outputs = K.function([dream], outputs)
	return f_outputs

def deep_dream(args, model):
	img_size = (args.channels, args.height, args.width)
	f_out = get_deep_dream_fxn(args, model)
	evaluator = Evaluator(args, f_out)

	# run scipy-based optimization (L-BFGS) over the pixels of the generated image
	# so as to minimize the loss
	x = preprocess_image(args, args.image_path)
	for i in range(5):
		print('Start of iteration', i)
		start_time = time.time()

		# add a random jitter to the initial image. This will be reverted at decoding time
		random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
		x += random_jitter

		# run L-BFGS for 7 steps
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
										 fprime=evaluator.grads, maxfun=7)
		print('Current loss value:', min_val)
		# decode the dream and save it
		x = x.reshape(img_size)
		x -= random_jitter
		img = deprocess_image(np.copy(x))
		fname = args.save_path + 'dream2_at_iteration_%d.png' % i
		imsave(fname, img)
		end_time = time.time()
		print('Image saved as', fname)
		print('Iteration %d completed in %ds' % (i, end_time - start_time))


def deep_dream2(args, model, canvas):
	img_size = (args.channels, args.height, args.width)
	f_out = get_deep_dream_fxn(args, model)
	evaluator = Evaluator(args, f_out)

	# run scipy-based optimization (L-BFGS) over the pixels of the generated image
	# so as to minimize the loss
	x = preprocess_image(args, args.image_path)
	for i in range(args.iterations):
		print('Start of iteration', i)
		start_time = time.time()

		# add a random jitter to the initial image. This will be reverted at decoding time
		random_jitter = (settings['jitter'] * 2) * (np.random.random(img_size) - 0.5)
		x += random_jitter

		# run L-BFGS for 7 steps
		x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
										 fprime=evaluator.grads, maxfun=2)
		print('Current loss value:', min_val)
		# decode the dream and save it
		x = x.reshape(img_size)
		x -= random_jitter
		img = deprocess_image(np.copy(x))
		fname = args.save_path + 'dream2_at_iteration_%d.png' % i
		imsave(fname, img)
		end_time = time.time()
		print('Image saved as', fname)
		print('Iteration %d completed in %ds' % (i, end_time - start_time))
		if args.video_writer:
			print('Frame written...')
			args.video_writer.write(img)

		canvas[:img.shape[0],:img.shape[1],:] = img



def eval_loss_and_grads(x, args, f_outputs):
	img_size = (args.channels, args.height, args.width)
	if args.video_writer:
		frame = x.reshape(img_size).astype('uint8')
		print 'frame shape is:', frame.shape
		args.video_writer.write(frame)	
	x = x.reshape((1,) + img_size)
	outs = f_outputs([x])
	loss_value = outs[0]
	if len(outs[1:]) == 1:
		grad_values = outs[1].flatten().astype('float64')
	else:
		grad_values = np.array(outs[1:]).flatten().astype('float64')

	return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.


class Evaluator(object):

	def __init__(self, args, f_outputs):
		self.loss_value = None
		self.grad_values = None
		self.args = args
		self.f_outputs = f_outputs

	def loss(self, x):
		assert self.loss_value is None
		loss_value, grad_values = eval_loss_and_grads(x, self.args, self.f_outputs)
		self.loss_value = loss_value
		self.grad_values = grad_values
		return self.loss_value

	def grads(self, x):
		assert self.loss_value is not None
		grad_values = np.copy(self.grad_values)
		self.loss_value = None
		self.grad_values = None
		return grad_values


def excite_softmax(args, model, img_path=None):
	layer_dict = dict([(layer.name, layer) for layer in model.layers])
	for filter_index in range(3, 56, 4):
		iterate = iterate_fxn(model, layer_dict, '1000_predictions', filter_index)

		if args.image_path != '':
			im = cv2.resize(cv2.imread(args.image_path ), (args.width, args.height))
			im = np.array(im.transpose((2,1,0)), np.float32)
			im -= im.mean()
			im /= (im.std() + 1e-5)
			im *= 0.1			
			input_img_data = np.expand_dims(im, axis=0)
			out_file = args.save_path + '%s_%s_%s_filter_%d.png' % (os.path.basename(args.weights), os.path.basename(args.image_path), layer.name, filter_index)
		else:
			input_img_data = np.random.random((1, args.channels, args.width, args.height)) * 20 + 128.
			out_file = args.save_path + 'excite_softmax_%d.png' % (filter_index)
		# run gradient ascent
		for i in range(args.iterations):
			loss_value, grads_value = iterate([input_img_data])
			input_img_data += grads_value #* i
			if i % 4 == 0:
				print "After iteration:", i, "loss is:", loss_value, "filter index:", filter_index
		
		img = input_img_data[0]
		img = deprocess_image(img)
		imsave(out_file, img)


def write_filters(args, model):
	for filter_index in range(2, 25, 4):
		exclude = ['input','zero','fc','flatten','predictions']
		for layer in model.layers:
			if any([ex in layer.name for ex in exclude]):
				continue
			print " layer name:", layer.name, "filter index:", filter_index
			layer_dict = dict([(layer.name, layer) for layer in model.layers])

			iterate = iterate_fxn(model, layer_dict, layer.name, filter_index)
			
			if args.image_path != '':
				im = cv2.resize(cv2.imread(args.image_path), (args.width, args.height))
				im = np.array(im.transpose((2,1,0)), np.float32)
				input_img_data = np.expand_dims(im, axis=0)
				out_file = args.save_path + '%s_%s_%s_filter_%d.png' % (os.path.basename(args.weights), os.path.basename(args.image_path), layer.name, filter_index)
			else:
				input_img_data = np.random.random((1, 3, args.width, args.height)) * 20 + 128.
				out_file = args.save_path + 'vgg2_overfit_random_%s_filter_%d.png' % (layer.name, filter_index)

			# run gradient ascent
			for i in range(args.iterations):
				loss_value, grads_value = iterate([input_img_data])
				input_img_data += grads_value * i
				if i % 4 == 0:
					print "After iteration:", i, "loss is:", loss_value," layer name:", layer.name, "filter index:", filter_index
			
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
	if K.image_dim_ordering() == 'th':
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
	tv = K.sum(K.pow(a + b, 1./2.))

	return 0.005*normalize(x, tv)
	 #tv

def iterate_fxn(model, layer_dict, layer_name='conv5_1', filter_index=0 ):
	first_layer = model.layers[0]
	# this is a placeholder tensor that will contain our generated images
	input_img = first_layer.input    

	# build a loss function that maximizes the activation
	# of the nth filter of the layer considered
	layer_output = layer_dict[layer_name].output

	if 'predictions' in layer_name or 'fc' in layer_name:
		loss = K.mean(layer_output[:, filter_index]) #+ total_variation_norm(input_img) #+ alpha_norm(input_img)
	else:
		loss = K.mean(layer_output[:, filter_index, :, :])

	# compute the gradient of the input picture wrt this loss
	grads = K.gradients(loss, input_img)[0]

	# normalization trick: we normalize the gradient
	grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

	# this function returns the loss and grads given the input picture
	iterate = K.function([input_img], [loss, grads])
	return iterate

# util function to convert a tensor into a valid image
def deprocess_image(x):
	# normalize tensor: center on 0., ensure std is 0.1
	x -= x.mean()
	x /= (x.std() + 1e-5)
	x *= 0.1

	# clip to [0, 1]
	x += 0.5
	x = np.clip(x, 0, 1)

	# convert to RGB array
	x *= 255
	x = x.transpose((2, 1, 0))
	x = np.clip(x, 0, 255).astype('uint8')
	return x




def preprocess_image(args, image_path):
    img = load_img(image_path, target_size=(args.height, args.width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def preprocess_input(x):
    x /= 255.
    x -= 0.5
    x *= 2.
    return x

if __name__ == '__main__':
	run()