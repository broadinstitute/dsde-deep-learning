# DSDE Deep Learning
#
# Train 1D Convolutional Neural Network from a labeled bed filepath
# Input is a small window of reference sequence
# Output is the softmax predicted probabilty of each label from the bed file
#
# January 2017
# Sam Friedman 
# sam@broadinstitute.org

from __future__ import print_function

import os
import math
import h5py
import argparse
import matplotlib
import numpy as np
matplotlib.use('Agg')
from scipy import interp
from keras import metrics
import keras.backend as K
from random import shuffle
from Bio import Seq, SeqIO
from itertools import cycle
import matplotlib.pyplot as plt
from collections import Counter

from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from sklearn.metrics import roc_curve, auc, roc_auc_score
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.convolutional import Convolution1D, MaxPooling1D, Conv1D
from keras.layers import Input, Dense, Dropout, BatchNormalization, SpatialDropout1D, Activation, Flatten

data_path = '/dsde/data/deep/'
reference_fasta = data_path + 'Homo_sapiens_assembly19.fasta'
chrom_hmm_bed_file = data_path + 'wgEncodeAwgSegmentationCombinedGm12878.bed'
breakpoint_bed_file =  '/dsde/data/deep/vqsr/beds/icgc_bkp_sorted.bed'

amiguity_codes = {'K':[0,0,0.5,0.5], 'M':[0.5,0.5,0,0], 'R':[0.5,0,0,0.5], 'Y':[0,0.5,0.5,0], 'S':[0,0.5,0,0.5], 
				  'W':[0.5,0,0.5,0], 'B':[0,0.333,0.333,0.334], 'V':[0.333,0.333,0,0.334],'H':[0.333,0.333,0.334,0],
				  'D':[0.333,0,0.333,0.334], 'X':[0.25,0.25,0.25,0.25], 'N':[0.25,0.25,0.25,0.25]}

label_sets = {
	'chrom_hmm'  : {'TSS':0, 'PF':1, 'E':2, 'WE':3, 'CTCF':4, 'T':5, 'R':6 },
	'breakpoint' : {'WT':0, 'BREAKPOINT':1}
}


def run():
	args = parse_args()

	if 'small' == args.mode:
		make_small_model(args)
	elif 'big' == args.mode:
		make_big_model(args)
	elif 'bp' == args.mode:
		make_breakpoint_model(args)
	else:
		print('Unknown model argument')


def parse_args():
	parser = argparse.ArgumentParser()

	parser.add_argument('--mode', default='bp')
	parser.add_argument('--window_size', default=128, type=int)
	parser.add_argument('--samples', default=1000, type=int)
	parser.add_argument('--reference_fasta', default=reference_fasta)
	parser.add_argument('--bed_file',default=breakpoint_bed_file)	
	parser.add_argument('--inputs', default={'A':0, 'C':1, 'T':2, 'G':3})
	parser.add_argument('--label_set', default='breakpoint', choices=label_sets.keys())
	parser.add_argument('--labels', default={})
	parser.add_argument('--conv_width', default=15, help='Width of 1D convolutional kernels.')
	parser.add_argument('--conv_dropout', default=0.0, type=float, 
		help='Dropout rate in convolutional layers.')
	parser.add_argument('--conv_batch_normalize', default=False, action='store_true',
		help='Batch normalize convolutional layers.')
	parser.add_argument('--conv_layers', nargs='+', default=[128, 64, 32], type=int,
		help='List of sizes for each convolutional filter layer')
	parser.add_argument('--same_padding', default=False, action='store_true',
		help='Valid or same border padding on the convolutional layers.')	
	parser.add_argument('--spatial_dropout', default=False, action='store_true',
		help='Spatial dropout on the convolutional layers.')	
	parser.add_argument('--max_pools', nargs='+', default=[], type=int,
		help='List of maxpooling layers.')	
	parser.add_argument('--fc_layers', nargs='+', default=[32], type=int,
		help='List of sizes for each fully connected layer')
	parser.add_argument('--fc_dropout', default=0.0, type=float, 
		help='Dropout rate in fully connected  layers.')
	parser.add_argument('--fc_batch_normalize', default=False, action='store_true',
		help='Batch normalize fully connected layers.')

	args = parser.parse_args()
	args.labels = label_sets[args.label_set]
	print('Arguments are', args)
	return args


def make_small_model(args):
	model = build_small_chrom_label(args)

	train_data = load_dna_and_chrom_label(args)
	train, valid, test = split_data(train_data)

	weight_path = weight_path_from_args(args)
	model = train_chrom_labeller(model, train, valid, weight_path)

	title = weight_path_to_title(weight_path)
	plot_roc(model, test[0], test[1], args.labels, title)
	plot_roc_per_class(model, test[0], test[1], args.labels, title)	


def make_big_model(args):
	model = build_sequential_chrom_label(args)

	# hacky class balancing
	train_data1 = load_dna_and_chrom_label(args, ['T', 'R'])
	args.samples *= 2
	train_data2 = load_dna_and_chrom_label(args, ['PF', 'E', 'TSS', 'CTCF', 'WE'])
	train_data = concat_and_shuffle(train_data1, train_data2)

	train, valid, test = split_data(train_data)

	weight_path = weight_path_from_args(args)
	model = build_sequential_chrom_label(args)
	model = train_chrom_labeller(model, train, valid, weight_path)

	title = weight_path_to_title(weight_path)
	plot_roc(model, test[0], test[1], args.labels, title)
	plot_roc_per_class(model, test[0], test[1], args.labels, title)	


def make_breakpoint_model(args):
	model = build_breakpoint_classifier(args)
	training_data = load_dna_and_breakpoint_labels(args)
	train, valid, test = split_data(training_data)
	
	weight_path = weight_path_from_args(args)
	model = build_breakpoint_classifier(args)
	model = train_chrom_labeller(model, train, valid, weight_path)

	title = weight_path_to_title(weight_path)
	plot_roc_per_class(model, test[0], test[1], args.labels, title)	


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_breakpoint_classifier(args):
	return build_reference_1d_model_from_args(args,
				conv_width = args.conv_width,
				conv_layers = args.conv_layers,
				conv_dropout = args.conv_dropout,
				conv_batch_normalize = args.conv_batch_normalize,
				spatial_dropout = args.spatial_dropout,
				max_pools = args.max_pools,
				padding='same' if args.same_padding else 'valid',
				fc_layers = args.fc_layers,
				fc_dropout = args.fc_dropout,
				fc_batch_normalize = args.fc_batch_normalize)


def build_small_chrom_label(args):
	model = Sequential()
	model.add(Convolution1D(input_dim=len(args.inputs), 
		input_length=args.window_size, 
		nb_filter=40,
		filter_length=16,
		border_mode='valid',
		activation="relu",
		init='normal'))

	model.add(MaxPooling1D(pool_length=3, stride=3))
	model.add(Convolution1D(nb_filter=64, filter_length=16, activation="relu", init='normal', border_mode='valid'))
	model.add(Dropout(0.2))	
	model.add(MaxPooling1D(pool_length=3, stride=3))
	model.add(Flatten())

	model.add(Dense(output_dim=32, init='normal'))
	model.add(Activation('relu'))

	model.add( Dense(output_dim=len(args.labels), init='normal') )
	model.add( Activation('softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	classes = args.labels.keys()
	my_metrics = [metrics.categorical_accuracy, precision, recall ]

	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=my_metrics)
	model.summary()

	return model


def build_sequential_chrom_label(args):
	model = Sequential()
	model.add(Convolution1D(input_dim=len(args.inputs), 
		input_length=args.window_size, 
		nb_filter=128,
		filter_length=16,
		border_mode='valid',
		activation="relu",
		init='normal'))

	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=192, filter_length=16, activation="relu", init='normal', border_mode='valid'))
	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=192, filter_length=16, activation="relu", init='normal', border_mode='valid'))
	model.add(Dropout(0.2))
	model.add(Convolution1D(nb_filter=256, filter_length=16, activation="relu", init='normal', border_mode='valid'))
	model.add(Dropout(0.2))	
	model.add(MaxPooling1D(pool_length=3, stride=3))
	model.add(Flatten())

	model.add(Dense(output_dim=50, init='normal'))
	model.add(Activation('relu'))

	model.add( Dense(output_dim=len(args.labels), init='normal') )
	model.add( Activation('softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=0.5)
	adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
	classes = args.labels.keys()
	my_metrics = [metrics.categorical_accuracy, precision, recall]

	model.compile(loss='categorical_crossentropy', optimizer=adamo, metrics=my_metrics)
	model.summary()

	return model


def build_reference_1d_model_from_args(args,
									conv_width = 6, 
									conv_layers = [128, 128, 128, 128],
									conv_dropout = 0.0,
									conv_batch_normalize = False,			
									spatial_dropout = False,
									max_pools = [],
									padding='valid',
									fc_layers = [64],
									fc_dropout = 0.0,
									fc_batch_normalize = False,
									fc_initializer='glorot_normal',
									kernel_initializer='glorot_normal'):
	'''Build Reference 1d CNN model for classification.

	Architecture specified by parameters.
	Dynamically sets input channels based on args via defines.total_input_channels_from_args(args)
	Uses the functional API.
	Prints out model summary.

	Arguments
		args.labels: The output labels (e.g. BREAKPOINT, WT)

	Returns
		The keras model
	'''	
	concat_axis = -1	
	x = reference = Input(shape=(args.window_size, len(args.inputs)), name='reference')

	max_pool_diff = len(conv_layers)-len(max_pools)	
	for  i,c in enumerate(conv_layers):

		if conv_batch_normalize:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='linear', padding=padding, kernel_initializer=kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation('relu')(x)
		else:
			x = Conv1D(filters=c, kernel_size=conv_width, activation='relu', padding=padding, kernel_initializer=kernel_initializer)(x)

		if conv_dropout > 0 and spatial_dropout:
			x = SpatialDropout1D(conv_dropout)(x)
		elif conv_dropout > 0:
			x = Dropout(conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling1D(max_pools[i-max_pool_diff])(x)

	x = Flatten()(x)

	for fc in fc_layers:
		if fc_batch_normalize:
			x = Dense(units=fc, activation='linear', kernel_initializer=fc_initializer)(x)
			x = BatchNormalization(axis=1)(x)
			x = Activation('relu')(x)			
		else:
			x = Dense(units=fc, activation='relu')(x)
		
		if fc_dropout > 0:
			x = Dropout(fc_dropout)(x)

	prob_output = Dense(units=len(args.labels), activation='softmax')(x)
	
	model = Model(inputs=[reference], outputs=[prob_output])
	
	adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	my_metrics = [metrics.categorical_accuracy, precision, recall]

	model.compile(optimizer=adam, loss='binary_crossentropy', metrics=my_metrics)
	model.summary()
	
	return model


def train_chrom_labeller(model, train_tuple, valid_tuple, save_weight_hd5):
	checkpointer = ModelCheckpoint(filepath=save_weight_hd5, verbose=1, save_best_only=True)
	earlystopper = EarlyStopping(monitor='val_loss', patience=5, verbose=1)	

	history = model.fit(train_tuple[0], train_tuple[1], 
		batch_size=32, epochs=150, shuffle=False, 
		validation_data=(valid_tuple[0], valid_tuple[1]), 
		callbacks=[checkpointer,earlystopper])

	plot_metric_history(history, weight_path_to_title(save_weight_hd5))

	return model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training Data ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def load_dna_and_breakpoint_labels(args):
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	bed_dict, bed_labels = bed_file_labels_to_dict(args.bed_file)

	train_data = np.zeros(( args.samples, args.window_size, len(args.inputs) ))
	train_labels = np.zeros(( args.samples, 2 ))

	idx_offset = (args.window_size/2)
	stats = Counter()
	while stats['count'] < args.samples:
		contig_key, pos = sample_from_bed(bed_dict, contig_key_prefix='')
		contig = record_dict[contig_key]
		record = contig[pos-idx_offset: pos+idx_offset]

		cur_label_key = bed_file_label(bed_dict, contig_key, pos)
		train_labels[stats['count'], args.labels[cur_label_key]] = 1

		for i,b in enumerate(record.seq):
			B=b.upper()
			if B in args.inputs.keys():
				train_data[stats['count'], i, args.inputs[B]] = 1.0
			elif B in amiguity_codes.keys():
				train_data[stats['count'], i, :4] = amiguity_codes[B]
			else:
				raise ValueError('Error! Unknown code:', b)
		
		stats['count'] += 1

	print('Labels:', args.labels, 'counts:', np.sum(train_labels, axis=0))
	print('Train data shape:', train_data.shape, ' Training labels shape:', train_labels.shape)

	return (train_data, train_labels)


def load_dna_and_chrom_label(args, only_labels=None):
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	bed_dict, bed_labels = bed_file_labels_to_dict(args.bed_file)

	train_data = np.zeros(( args.samples, args.window_size, len(args.inputs) ))
	train_labels = np.zeros(( args.samples, len(bed_labels) ))

	idx_offset = (args.window_size/2)

	count = 0
	while count < args.samples:
		contig_key, pos = sample_from_bed(bed_dict, contig_key_prefix='chr')
		contig = record_dict[contig_key]
		record = contig[pos-idx_offset: pos+idx_offset]

		cur_label_key = bed_file_label(bed_dict, contig_key, pos)

		if only_labels and not cur_label_key in only_labels:
			continue

		train_labels[count, args.labels[cur_label_key]] = 1

		for i,b in enumerate(record.seq):
			B=b.upper()
			if B in args.inputs.keys():
				train_data[count, i, args.inputs[B]] = 1.0
			elif B in amiguity_codes.keys():
				train_data[count, i, :4] = amiguity_codes[B]
			else:
				print('Error! Unknown code:', b)
				return

		count += 1

	print('Label:', bed_labels.keys(), 'label counts:', np.sum(train_labels, axis=0))
	print('Train data shape:', train_data.shape, ' Training labels shape:', train_labels.shape)

	return (train_data, train_labels)


def bed_file_labels_to_dict(bed_file):
	bed = {}
	labels = {}

	with open(bed_file)as f:
		for line in f:
			parts = line.split()
			contig = parts[0]
			if contig == 'chrom' or contig[0] == '#':
				continue

			lower = int(parts[1])
			upper = int(parts[2])
			label = parts[3]

			if contig not in bed.keys():
				bed[contig] = ([], [], [])

			bed[contig][0].append(lower)
			bed[contig][1].append(upper)
			bed[contig][2].append(label)

			if label not in labels.keys():
				labels[label] = ([], [])
			labels[label][0].append(lower)
			labels[label][1].append(upper)

	for k in bed.keys():
		bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]), bed[k][2])		
		print('key is:', k , 'len ', len(bed[k][0]))

	for l in labels.keys():
		labels[l] = (np.array(labels[l][0]), np.array(labels[l][1]))		
		print('label is:', l , 'len ', len(labels[l][0]))

	return bed, labels


def in_bed_file(bed_dict, contig, pos):
	lows = bed_dict[contig][0]
	ups = bed_dict[contig][1]

	return np.any((lows <= pos) & (pos <= ups))


def bed_file_label(bed_dict, contig, pos):

	if in_bed_file(bed_dict, contig, pos):
		lows = bed_dict[contig][0]
		ups = bed_dict[contig][1]
		i = np.argmax((lows <= pos) & (pos <= ups))
		label = bed_dict[contig][2][i]

		return label


def split_data(datasets, valid_ratio=0.1, test_ratio=0.4):
	assert(valid_ratio + test_ratio < 1.0 and valid_ratio >= 0.0 and test_ratio >= 0.0)

	samples = datasets[0].shape[0]
	indices = range(samples)
	shuffle(indices)

	train = []
	valid = []
	test = []

	valid_idx = int(valid_ratio * float(samples))
	test_idx = int(test_ratio * float(samples))

	for d in datasets:
		valid.append( d[ :valid_idx] )
		test.append(  d[valid_idx : valid_idx + test_idx] )
		train.append( d[valid_idx + test_idx: ] )

	return train, valid, test


def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)


# Samples must be in first axis
def concat_and_shuffle(data_label_tuple1, data_label_tuple2):
	concat_data = np.concatenate((data_label_tuple1[0], data_label_tuple2[0]))
	concat_labels = np.concatenate((data_label_tuple1[1], data_label_tuple2[1]))
	shuffle_in_unison(concat_data, concat_labels)
	return (concat_data, concat_labels)


def sample_from_fasta(record_dict):
	c_idx = str(np.random.randint(1,20))
	contig = record_dict[c_idx]
	p_idx = np.random.randint(len(contig))
	return c_idx, p_idx


def sample_from_bed(bed_dict, contig_key_prefix=''):
	contig_key = contig_key_prefix + str(np.random.randint(1,22))
	lowers = bed_dict[contig_key][0]
	uppers = bed_dict[contig_key][1]

	idx = np.random.randint(len(lowers))
	mid_pos = (lowers[idx] + uppers[idx]) / 2
	return contig_key, mid_pos


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
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def per_class_precision(classes):

	def class_precision(y_true, y_pred):
		'''Calculates the per class recall
		'''
		precisions = {}
		true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)), axis=0)

		for i,c in enumerate(classes):
			precisions[c+'_PRECISION'] = true_positives[i] / (predicted_positives[i] + K.epsilon())

		return precisions
	return class_precision


def per_class_recall(classes):

	def class_recall(y_true, y_pred):
		'''Calculates the per class recall
		'''
		recalls = {}
		true_positives = K.sum(K.round(K.clip(y_true*y_pred, 0, 1)), axis=0)
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)

		for i,c in enumerate(classes):
			recalls[c+'_RECALL'] = true_positives[i] / (possible_positives[i] + K.epsilon())

		return recalls
	return class_recall


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Plots ~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_roc(model, test_data, test_truth, labels, title):
	y_pred = model.predict(test_data, batch_size=32, verbose=0)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for i,k in enumerate(labels.keys()):
		fpr[i], tpr[i], _ = roc_curve(test_truth[:,i], y_pred[:,i])
		roc_auc[i] = auc(fpr[i], tpr[i])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(test_truth.ravel(), y_pred.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])	

	plt.figure(figsize=(16,16))
	lw = 2
	plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC:' + str(labels) + '\n' + title)
	plt.legend(loc="lower right")
	plt.savefig("./roc_"+title+".png")	


def get_fpr_tpr_roc(model, test_data, test_truth, labels):
	y_pred = model.predict(test_data, batch_size=32, verbose=0)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()
	for k in labels.keys():
		cur_idx = labels[k]
		fpr[labels[k]], tpr[labels[k]], _ = roc_curve(test_truth[:,cur_idx], y_pred[:,cur_idx])
		roc_auc[labels[k]] = auc(fpr[labels[k]], tpr[labels[k]])

	# Compute micro-average ROC curve and ROC area
	fpr["micro"], tpr["micro"], _ = roc_curve(test_truth.ravel(), y_pred.ravel())
	roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
	return fpr, tpr, roc_auc


def plot_roc_per_class(model, test_data, test_truth, labels, title):
	# Compute macro-average ROC curve and ROC area
	fpr, tpr, roc_auc = get_fpr_tpr_roc(model, test_data, test_truth, labels)
	# First aggregate all false positive rates
	n_classes = len(labels)
	all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

	# Then interpolate all ROC curves at this points
	mean_tpr = np.zeros_like(all_fpr)
	for i in range(n_classes):
	    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

	# Finally average it and compute AUC
	mean_tpr /= n_classes

	fpr["macro"] = all_fpr
	tpr["macro"] = mean_tpr
	roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

	# Plot all ROC curves
	lw = 2
	plt.figure(figsize=(20,16))

	colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'pink', 'magenta', 'grey', 'purple'])
	idx = 0
	for key, color in zip(labels.keys(), colors):
	    plt.plot( fpr[idx], tpr[idx], color=color, lw=lw, label='ROC curve of class '+str(key) )
	    idx += 1

	plt.plot([0, 1], [0, 1], 'k--', lw=lw)
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC:'+ str(labels) + '\n' + title)
	plt.legend(loc="lower right")
	plt.savefig("./per_class_roc_"+title+".png")	


def plot_history(history, title):
	# list all data in history
	print(history.history.keys())
	# summarize history for accuracy
	plt.plot(history.history['categorical_accuracy'])
	plt.plot(history.history['val_categorical_accuracy'])
	plt.title('Accuracy: '+title)
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Loss: '+title)
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.savefig("./plot_history_"+title+".png")	


def plot_metric_history(history, title):
	# list all data in history
	print(history.history.keys())

	row = 0
	col = 0
	num_plots = len(history.history)/2.0 # valid and train plot together
	rows = 4
	cols = int(math.ceil(num_plots/float(rows)))

	f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))
	
	if cols>1:
		for k in history.history.keys():

			if 'val' not in k:
				axes[row, col].plot(history.history[k])
				axes[row, col].plot(history.history['val_'+k])

				axes[row, col].set_ylabel(str(k))
				axes[row, col].legend(['train', 'valid'], loc='upper left')
				axes[row, col].set_xlabel('epoch')

				row += 1
				if row == rows:
					row = 0
					col += 1
					if row*col >= rows*cols:
						break

		axes[0, 1].set_title(title)
	else:
		for k in history.history.keys():

			if 'val' not in k:
				axes[row].plot(history.history[k])
				axes[row].plot(history.history['val_'+k])

				axes[row].set_ylabel(str(k))
				axes[row].legend(['train', 'valid'], loc='upper left')
				axes[row].set_xlabel('epoch')

				row += 1
		axes[0].set_title(title)


	plt.savefig("./metric_history_"+title+".png")	

	
def weight_path_from_args(args):
	save_weight_hd5 =  './chrom_hmm_cnn_model' 

	include = ['window_size', 'samples', 'conv_width', 'mode'] 
	for arg in vars(args):
		if not arg in include:
			continue

		attr = getattr(args, arg)

		if os.path.isdir(str(attr)) or os.path.isfile(str(attr)):
			continue

		if os.path.isabs(str(attr)):
			attr = os.path.splitext(os.path.basename(attr))[0]

		save_weight_hd5 += '__' + str(arg) + '_' + str(attr)

	save_weight_hd5 += '.hd5'
	print('save weight path:', save_weight_hd5)

	return save_weight_hd5


def weight_path_to_title(wp):
	return wp.split('/')[-1].replace('__', '-')


if '__main__'==__name__:
	run() 