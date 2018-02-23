# models_for_intel.py
# February 2017
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 Friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import sys
import h5py
import json
import shutil
import argparse
import tempfile
import numpy as np
from collections import Counter

# TensorFlow imports
import multiprocessing
import tensorflow as tf
from tensorflow.contrib import learn
from tensorflow.contrib import layers
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import metrics
from tensorflow.contrib.keras.python.keras.optimizers import SGD, Adam
from tensorflow.contrib.keras.python.keras.models import Sequential, Model
from tensorflow.contrib.keras.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.contrib.keras.python.keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D
from tensorflow.contrib.keras.python.keras.layers import Input, Dense, Dropout, SpatialDropout2D, Flatten, Reshape, merge


# Google API
import googleapiclient.http
import googleapiclient.discovery


tensor_exts = ['.h5', '.hd5']
snp_indel_labels = {'NOT_SNP':0, 'NOT_INDEL':1, 'SNP':2, 'INDEL':3}


def run():
	args = parse_args()
	if 'tf_model' == args.mode:
		tf_sesh(args)
	elif 'keras_model' == args.mode:
		simple_2d_train(args)	
	elif 'test_api' == args.mode:
		print('Fetching object..')
		with open('myf.hd5') as tmpfile:
			my_obj = get_object('broad-dsde-methods-mlengine', 
				'tensors/valid/SNP/recalibrated_g94982_nist_na12878_minimal-SNP-1_79331371.h5', 
				out_file=tmpfile)
		with h5py.File('my2.hd5', 'r') as hf:
			np_tensor = np.array(hf.get('read_tensor'))
		print(np_tensor)
	elif 'list' == args.mode:
		bucket = list_tensors('broad-dsde-methods-mlengine', 'valid')
		for f in bucket:
			print(f)

	else:
		print("Unknown mode argument:", args.mode)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode', default='tf_model')

	parser.add_argument('--read_flags', default=12, type=int)
	parser.add_argument('--samples', default=100, type=int)
	parser.add_argument('--epochs', default=40, type=int)	
	parser.add_argument('--batch_size', default=32, type=int)
	parser.add_argument('--read_limit', default=128, type=int)
	parser.add_argument('--window_size', default=128, type=int)
	parser.add_argument('--weights_hd5', default='')
	parser.add_argument('--labels', default=snp_indel_labels)
	parser.add_argument('--data_dir', default='./tensors/')
	parser.add_argument('--channels_in', default=10, type=int)
	parser.add_argument('--channels_last', default=True, dest='channels_last', action='store_true', help='TensorFlow Convention')
	parser.add_argument('--channels_first', dest='channels_last', action='store_false', help='Theano Convention')

	args = parser.parse_args()
	print('Arguments are', args)
	return args


def tf_sesh(args):
	# Parameters
	learning_rate = 0.001
	training_iters = 200000
	display_step = 10
	dropout = 0.75 # Dropout, probability to keep units

	train_dir = args.data_dir + 'train/'
	valid_dir = args.data_dir + 'valid/'
	train_paths = [train_dir + tp for tp in sorted(os.listdir(train_dir)) if os.path.isdir(train_dir + tp)]
	valid_paths = [valid_dir + vp for vp in sorted(os.listdir(valid_dir)) if os.path.isdir(valid_dir + vp)]
	assert(len(train_paths) == len(valid_paths))

	in_shape = in_shape_from_args(args)
	generate_train = tensor_generator(args, train_paths, in_shape)
	generate_valid = tensor_generator(args, valid_paths, in_shape)

	x = tf.placeholder(tf.float32, shape=[None, args.read_limit, args.window_size, args.channels_in])
	y = tf.placeholder(tf.float32, shape=[None, len(args.labels)])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

	# Construct model
	pred = cnn_model_tf(x, args, keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	tf.summary.scalar('cost', cost)
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	tf.summary.scalar('accuracy', accuracy)

	
	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter('./train', sess.graph)
		test_writer = tf.summary.FileWriter('./test')

		sess.run(init)
		step = 1
		# Keep training until reach max iterations
		while step * batch_size < training_iters:
			batch_x, batch_y = generate_train.next()
			# Run optimization op (backprop)
			summary, _ = sess.run([merged, optimizer], feed_dict={x:batch_x, y:batch_y, keep_prob:dropout})
			train_writer.add_summary(summary, step)
			if step % display_step == 0:
				# Calculate batch loss and accuracy
				batch_x, batch_y = generate_valid.next()
				summary, loss, acc = sess.run([merged, cost, accuracy], feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
				test_writer.add_summary(summary, step)
				print("Step " + str(step*args.batch_size) + ", Loss={:.6f}".format(loss) + ", Accuracy={:.5f}".format(acc))
			step += 1

		val_x, val_y = generate_valid.next()
		print("Optimization Finished! Testing Accuracy:", sess.run(accuracy, feed_dict={x:val_x, y:val_y, keep_prob:1.}))


def simple_2d_train(args):
	"""Trains a reference and read based architecture on tensors at the supplied data directory.

	Arguments:
		args.data_dir: must be set to an appropriate directory with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files. 

	This architecture looks at reads, and the reference genome.
	Tensors must be generated by calling td.write_tensors() before this function is used.
	After training with early stopping a performance curves are plotted on the test dataset.
	"""	
	train_dir = args.data_dir + 'train/'
	valid_dir = args.data_dir + 'valid/'
	train_paths = [train_dir + tp for tp in sorted(os.listdir(train_dir)) if os.path.isdir(train_dir + tp)]
	valid_paths = [valid_dir + vp for vp in sorted(os.listdir(valid_dir)) if os.path.isdir(valid_dir + vp)]
	assert(len(train_paths) == len(valid_paths))

	if args.channels_last:
		tensor_shape = (args.read_limit, args.window_size, args.channels_in)
	else:
		tensor_shape = (args.channels_in, args.read_limit, args.window_size) 
	generate_train = tensor_generator(args, train_paths, tensor_shape)
	generate_valid = tensor_generator(args, valid_paths, tensor_shape)

	weight_path = weight_path_from_args(args)
	model = build_read_tensor_2d_model(args)
	model = train_bam_model_from_disk(args, model, generate_train, generate_valid, weight_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def model_fn(features, targets, mode, params):
	learning_rate = 0.001
	keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

	# Construct model
	pred = cnn_model_tfp(features, params, keep_prob)

	pred_dict = {
		"classes": tf.argmax(input=pred, axis=1), 
		"probabilities": tf.nn.softmax(pred, name="softmax_tensor")
	}

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=targets))
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(targets, 1))
	eval_metric_ops = {
		"accuracy" : tf.reduce_mean(tf.cast(correct_pred, tf.float32)),
		"model_loss" : loss}

	train_op = tf.contrib.layers.optimize_loss(
		loss=loss, 
		global_step=tf.contrib.framework.get_global_step(), 
		learning_rate=learning_rate, 
		optimizer="Adam")
	
	print("Got model fn fxn. params:", params)
	return model_fn_lib.ModelFnOps(mode=mode, 
									loss=loss, 
									train_op=train_op, 
									predictions=pred_dict, 
									eval_metric_ops=eval_metric_ops)


def cnn_model_fn(features, labels, mode):
	"""Model function for CNN."""
	# Input Layer
	input_layer = tf.reshape(features, [-1, 28, 28, 1])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
	  inputs=input_layer,
	  filters=32,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
	  inputs=pool1,
	  filters=64,
	  kernel_size=[5, 5],
	  padding="same",
	  activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
	dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
	  inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=10)

	loss = None
	train_op = None

	# Calculate Loss (for both TRAIN and EVAL modes)
	if mode != learn.ModeKeys.INFER:
		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
		loss = tf.losses.softmax_cross_entropy(
			onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(
			loss=loss,
			global_step=tf.contrib.framework.get_global_step(),
			learning_rate=0.001,
			optimizer="SGD")

	# Generate Predictions
	predictions = {
	  "classes": tf.argmax(
		  input=logits, axis=1),
	  "probabilities": tf.nn.softmax(
		  logits, name="softmax_tensor")
	}

	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)


def cnn_model_tfp(x, params, keep_prob):
	"""Model function for CNN."""
	read_conv_width = 16
	my_shape = [-1, params['read_limit'], params['window_size'], params['channels_in']]
	# Input Layer
	x = tf.reshape(x, shape=my_shape)

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[read_conv_width, 1], padding="same", activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[1, read_conv_width], padding="same", activation=tf.nn.relu)
	
	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])
	
	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, read_conv_width], padding="same", activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2])

	# Dense Layer
	pool2 = tf.reshape(pool2, [-1, 32*32*64]) # MAGIC #
	dense = tf.layers.dense(inputs=pool2, units=32, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=len(params['labels']))
	return logits


def cnn_model_tf(x, args, keep_prob):
	"""Model function for CNN."""
	read_conv_width = 16

	# Input Layer
	x = tf.reshape(x, shape=[-1, args.read_limit, args.window_size, args.channels_in])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[read_conv_width, 1], padding="same", activation=tf.nn.relu)
	conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[1, read_conv_width], padding="same", activation=tf.nn.relu)
	
	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=[2, 2])
	
	# Convolutional Layer #3
	conv3 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[1, read_conv_width], padding="same", activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=[2, 2])

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 32*32*64]) # MAGIC #
	dense = tf.layers.dense(inputs=pool2_flat, units=32, activation=tf.nn.relu)
	dropout = tf.layers.dropout(inputs=dense, rate=keep_prob)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=len(args.labels))
	return logits


def build_read_tensor_2d_model(args):
	'''Build Read Tensor 2d CNN model for classifying variants.

	2d Convolutions followed by dense connection.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API. Supports theano or tensorflow channel ordering.
	Prints out model summary.

	Arguments
		args.window_size: Length in base-pairs of sequence centered at the variant to use as input.	
		args.labels: The output labels (e.g. SNP, NOT_SNP, INDEL, NOT_INDEL)
		args.channels_last: Theano->False or Tensorflow->True channel ordering flag

	Returns
		The keras model
	'''		
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, args.channels_in)
	else:
		in_shape = (args.channels_in, args.read_limit, args.window_size)

	read_tensor = Input(shape=in_shape, name="read_tensor")
	read_conv_width = 16
	x = Conv2D(128, (read_conv_width, 1), padding='valid', activation="relu", kernel_initializer="he_normal")(read_tensor)
	x = Conv2D(64, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((3,1))(x)
	x = Conv2D(64, (1, read_conv_width), padding='valid', activation="relu", kernel_initializer="he_normal")(x)
	x = MaxPooling2D((3,3))(x)
	x = Flatten()(x)
	x = Dense(units=32, kernel_initializer='normal', activation='relu')(x)
	prob_output = Dense(units=len(args.labels), kernel_initializer='normal', activation='softmax')(x)
	
	model = Model(inputs=[read_tensor], outputs=[prob_output])
	
	adamo = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	my_metrics = [metrics.categorical_accuracy]
	
	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=my_metrics)
	model.summary()
	
	if os.path.exists(args.weights_hd5):
		model.load_weights(args.weights_hd5, by_name=True)
		print('Loaded model weights from:', args.weights_hd5)

	return model


def train_bam_model_from_disk(args, model, generate_train, generate_valid, save_weight_hd5):

	history = model.fit_generator(generate_train, steps_per_epoch=args.samples, epochs=args.epochs, verbose=1, 
		validation_steps=40, validation_data=generate_valid,
		callbacks=get_callbacks(save_weight_hd5, patience=8))

	return model


def get_callbacks(save_weight_hd5, patience=2):
	checkpointer = ModelCheckpoint(filepath=save_weight_hd5, verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1)	
	return [checkpointer,earlystopper]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training Data ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def generate_tensor_input_fxn_local(args, train_paths, per_class_max=4000):
	stats = Counter()

	tensor_files = {k:[] for k in args.labels.keys()}
	for tp in train_paths:
		k = tp.split('/')[-1]
		print('k is:', k)
		if k not in args.labels.keys():
			print('Skipping label directory:', k, ' which is not in args label set:', args.labels.keys())
			continue
		for t in os.listdir(tp):
			tensor_files[k].append(tp+'/'+t)

	per_class_per_batch = args.batch_size // len(args.labels)
	print('Got tensor dict. Per class per batch:',per_class_per_batch)
	
	for k in tensor_files.keys():
		print(k , ' has: ', len(tensor_files[k]))

	def _input_fn():
		count = 0

		tensors = []
		labels = []

		for k in tensor_files.keys():
			for i in range(per_class_per_batch):
				stats[k] += 1
				if stats[k] == len(tensor_files[k]):
					stats[k] = 0

				cur_tensor = tensor_files[k][stats[k]]

				fn, file_extension = os.path.splitext(cur_tensor)
				if not file_extension.lower() in tensor_exts:
					continue

				with h5py.File(cur_tensor, 'r') as hf:
					tensors.append(np.array(hf.get('read_tensor')))
					
				y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
				y_vector[args.labels[k]] = 1.0
				labels.append(y_vector)
		features = tf.constant(np.asarray(tensors), dtype=tf.float32)
		labels_tf = tf.constant(np.asarray(labels))
		return features, labels_tf
	return _input_fn


def generate_tensor_input_fxn(args, train_paths, per_class_max=4000):	
	stats = Counter()

	tensor_files = {k:[] for k in args.labels.keys()}
	for tp in train_paths:
		k = tp.split('/')[2]
		if k not in args.labels.keys():
			print('Skipping label directory:', k, ' which is not in args label set:', args.labels.keys())
			continue
		tensor_files[k].append(tp)

	per_class_per_batch = args.batch_size // len(args.labels)
	print('Got tensor dict. Per class per batch:',per_class_per_batch)
	
	for k in tensor_files.keys():
		print(k , ' has: ', len(tensor_files[k]))
	
	def _input_fn():
		tensors = []
		labels = []

		for k in tensor_files.keys():
			for i in range(per_class_per_batch):
				stats[k] += 1
				if stats[k] == len(tensor_files[k]):
					stats[k] = 0

				cur_tensor = tensor_files[k][stats[k]]
				tensors.append(file_to_np_tensor(args.bucket, cur_tensor, 'read_tensor'))
				
				y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
				y_vector[args.labels[k]] = 1.0
				labels.append(y_vector)
		
		features = tf.constant(np.asarray(tensors), dtype=tf.float32)
		labels_tf = tf.constant(np.asarray(labels))
		
		return features, labels_tf

	return _input_fn


def tensor_generator(args, train_paths, tensor_shape):
	'''Data generator of tensors with reads, and flags.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each
		tensor_shape: Shape of the input data tensor
		flag_shape: Shape of the read flag tensors

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	'''
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels) ) 
	tensor_counts = Counter()
	tensors = {}

	tensor = np.zeros(((args.batch_size,)+tensor_shape))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels.keys():
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 
		tensors[label] = [os.path.join(tp, t) for t in os.listdir(tp) if os.path.splitext(t)[1] in tensor_exts]
		tensor_counts[label] = 0
		
	while True:
		cur_example = 0
		for label in tensors.keys():
			for i in range(per_batch_per_label):
				tensor_path = tensors[label][tensor_counts[label]]

				try:
					with h5py.File(tensor_path,'r') as hf:
						tensor[cur_example] = np.array(hf.get('read_tensor'))
				except:
					e = sys.exc_info()[0]
					print('Delete corrupt tensor at:', tensor_path)
					print('Error is:', e, 'Expected shape:', tensor_shape)
					del tensors[label][tensor_counts[label]]
					continue
					
				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					print('\n\nGenerator looped over all:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield (tensor, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Utilities ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def weight_path_from_args(args):
	save_weight_hd5 =  './vqsr_model' 

	ignore = ['weights_hd5', 'negative_vcf', 'train_vcf', 'annotations', 'reference_fasta', 'labels', 'bam_file', 'data_dir'] 
	for arg in vars(args):
		if arg in ignore:
			continue

		attr = getattr(args, arg)

		if os.path.isdir(str(attr)) or os.path.isfile(str(attr)):
			continue

		if os.path.isabs(str(attr)):
			attr = os.path.splitext(os.path.basename(attr))[0]

		save_weight_hd5 += '__' + str(arg) + '_' + str(attr)

	save_weight_hd5 += '.hd5'
	print('save weight path:' , save_weight_hd5)

	return save_weight_hd5


def create_service():
    # Construct the service object for interacting with the Cloud Storage API -
    # the 'storage' service, at version 'v1'.
    # You can browse other available api services and versions here:
    #     http://g.co/dv/api-client-library/python/apis/
    return googleapiclient.discovery.build('storage', 'v1')


def get_object(bucket, filename, out_file):
    service = create_service()

    # Use get_media instead of get to get the actual contents of the object.
    # http://g.co/dv/resources/api-libraries/documentation/storage/v1/python/latest/storage_v1.objects.html#get_media
    req = service.objects().get_media(bucket=bucket, object=filename)

    downloader = googleapiclient.http.MediaIoBaseDownload(out_file, req)

    done = False
    while done is False:
        status, done = downloader.next_chunk()
        #print("Download {}%.".format(int(status.progress() * 100)))

    return out_file


def list_bucket(bucket):
    """Returns a list of metadata of the objects within the given bucket."""
    service = create_service()

    # Create a request to objects.list to retrieve a list of objects.
    fields_to_return = \
        'nextPageToken,items(name,size,contentType,metadata(my-key))'
    req = service.objects().list(bucket=bucket, fields=fields_to_return)

    all_objects = []
    # If you have too many items to list in one request, list_next() will
    # automatically handle paging with the pageToken.
    while req:
        resp = req.execute()
        all_objects.extend(resp.get('items', []))
        req = service.objects().list_next(req, resp)
    return all_objects


def list_tensors(bucket, dataset):
	tensors = []
	bucket = list_bucket(bucket)
	for f in bucket:
		if '/'+dataset+'/' in f['name'] and '.'+f['name'].split('.')[-1] in tensor_exts:
			tensors.append(f['name'])
	return tensors


def file_to_np_tensor(bucket, tensor_name, dataset_name):
	tmpfile_name = tensor_name.split('/')[-1]
	with open(tmpfile_name, 'w') as tmpfile:
		my_obj = get_object(bucket, tensor_name, out_file=tmpfile)
	with h5py.File(tmpfile_name, 'r') as hf:
		np_tensor = np.array(hf.get(dataset_name))
	return np_tensor


def in_shape_from_args(args, with_batch_size=True):
	if args.channels_last:
		in_shape = (args.read_limit, args.window_size, args.channels_in)
	else:
		in_shape = (args.channels_in, args.read_limit, args.window_size)

	if with_batch_size:
		inshape = (args.batch_size,) + in_shape

	return in_shape



# Back to the top!
if "__main__" == __name__:
	run()

