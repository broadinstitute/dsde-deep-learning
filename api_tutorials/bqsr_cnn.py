# bqsr_cnn.py
# May 2018
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import sys
import vcf
import json
import h5py
import time
import math
import pdb
import scipy
import pysam
import argparse
import numpy as np
import numpy.ma as ma
from array import array
from scipy import interpolate

import matplotlib
matplotlib.use('Agg') # Need this to write images from the GSA servers.  Order matters:
import matplotlib.pyplot as plt # First import matplotlib, then use Agg, then import plt
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score

from Bio import Seq, SeqIO
from collections import Counter, defaultdict

# Keras Imports
import keras
from keras import layers
from keras import metrics
import keras.backend as K
import keras_resnet.models
from keras.preprocessing import image
from keras.utils.vis_utils import model_to_dot
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import plot_model, to_categorical
from keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.layers import Activation, Flatten, Reshape, LSTM, merge, Permute, GlobalAveragePooling2D, MaxPooling1D
from keras.layers import Add, Input, Dense, Dropout, BatchNormalization, SpatialDropout2D, SpatialDropout1D
from keras.layers.convolutional import Conv1D, Conv2D, ZeroPadding2D, UpSampling1D, UpSampling2D, Conv2DTranspose

HD5_EXT = '.hd5'
IMAGE_EXT = '.png'

SKIP_CHAR = '~'
INDEL_CHAR = '*'
DNA_SYMBOLS = {'A':0, 'C':1, 'G':2, 'T':3}
DNA_INDEL_SYMBOLS = {'A':0, 'C':1, 'G':2, 'T':3, INDEL_CHAR:4}
DNA_AND_ANNOTATIONS = {'A':0, 'C':1, 'G':2, 'T':3, 'pair':4, 'cycle':5, 'mq':6}
INPUT_SYMBOLS = {
	'dna' : DNA_SYMBOLS,
	'dna_indel' : DNA_INDEL_SYMBOLS,
	'dna_annotations' : DNA_AND_ANNOTATIONS
}

# Base calling ambiguities, See https://www.bioinformatics.org/sms/iupac.html
AMBIGUITY_CODES = {'K':[0, 0, 0.5, 0.5], 'M':[0.5, 0.5, 0, 0], 'R':[0.5, 0, 0, 0.5], 
				   'Y':[0, 0.5, 0.5, 0], 'S':[0, 0.5, 0, 0.5], 'W':[0.5, 0, 0.5, 0],
				   'B':[0,0.333,0.333,0.334], 'V':[0.333,0.333,0,0.334],'H':[0.333,0.333,0.334,0],'D':[0.333,0,0.333,0.334],
				   'X':[0.25,0.25,0.25,0.25], 'N':[0.25,0.25,0.25,0.25]}


# Annotation sets
ANNOTATIONS = {
				'_' : [], # Allow command line to unset annotations
				'quick4' : ['Pair', 'Strand', 'Cycle', 'MappingQuality'],
				'best_practices' : ['Pair', 'Strand', 'Cycle', 'MappingQuality'],
				'annotations' : ['Pair', 'Strand', 'Cycle', 'MappingQuality']
			   }
MAX_MQ=60.0


BQSR_LABELS = {'GOOD_BASE':0, 'BAD_BASE':1}
KEY_COLORS = {'GOOD_BASE':'green', 'BAD_BASE':'red'}
precision_label = 'Precision | Positive Predictive Value | TP/(TP+FP)'
recall_label = 'Recall | Sensitivity | True Positive Rate | TP/(TP+FN)'
fallout_label = 'Fallout | 1 - Specificity | False Positive Rate | FP/(FP+TN)'
bad_bases_key, perfect_read_key, good_bases_key = 'bad bases', 'perfect read', 'good bases'
read_with_mismatch_key = 'read with mismatch'
dropped_error_q_key = 'dropped read with q < 6'
BQSR_TENSOR_NAME, OQ_TENSOR_NAME, LABEL_TENSOR_NAME = 'bqsr', 'oq', 'bqsr_labels'

def run():
	'''Parse arguments, create a model and dispatch on mode'''
	args = parse_args()

	if 'bqsr_train_tensor' == args.mode:
		bqsr_train_tensor(args)
	elif 'write_bqsr_tensors' == args.mode:
		write_base_recalibrate_tensors(args)
	elif 'recalibrate' == args.mode:
		recalibrate(args)
	else:
		raise ValueError('unknown bqsr mode:', args.mode)


def parse_args():
	data_path = '/dsde/data/deep/vqsr/'
	reference_fasta = data_path + 'Homo_sapiens_assembly19.fasta'
	
	parser = argparse.ArgumentParser()

	parser.add_argument('--maxfun', default=9, type=int)
	parser.add_argument('--fps', default=1, type=int)
	parser.add_argument('--learning_rate', default=0.01, type=float)
	parser.add_argument('--jitter', default=0.0, type=float)
	parser.add_argument('--l2', default=0.0, type=float)
	parser.add_argument('--l1', default=0.0, type=float)
	parser.add_argument('--activity_weight', default=1.0, type=float)
	parser.add_argument('--total_variation', default=0.00001, type=float)

	# Tensor defining arguments
	parser.add_argument('--labels', default=BQSR_LABELS, help='Dict mapping label names to their index within label tensors.')
	parser.add_argument('--label_weights',  nargs='+', default=[0.5, 0.5], type=float, help='List of weights for each label.')
	parser.add_argument('--input_symbol_set', default='dna_annotations', choices=INPUT_SYMBOLS.keys(), help='Key which maps to an input symbol to index mapping.')
	parser.add_argument('--input_symbols', default=None, help='Dict mapping input symbols to their index within input tensors, initialised via input_symbols_set argument')
	parser.add_argument('--batch_size', default=32, type=int, help='Mini batch size for stochastic gradient descent algorithms.')
	parser.add_argument('--read_limit', default=128, type=int, help='Maximum number of reads to load.')
	parser.add_argument('--window_size', default=151, type=int, help='Size of sequence window to use as input, typically centered at a variant.')
	parser.add_argument('--window_offset', default=2, type=int, help='Shift where in the window to predict. Positive numbers go further along the reading strand')
	parser.add_argument('--channels_last', default=False, dest='channels_last', action='store_true', help='Store the channels in the last axis of tensors, tensorflow->true, theano->false')
	parser.add_argument('--label_smoothing', default=0.0, type=float, help='Rate of smoothing for class labels  [0.0, 1.0] i.e. [label_smoothing, 1.0-label_smoothing].')
	parser.add_argument('--base_quality_mode', default='phot', choices=['phot', 'phred', '1hot'],
		help='How to treat base qualities, must be in [phot, phred, 1hot]')
	parser.add_argument('--map_input_to_logspace', default = False, type=bool, help='')
	## TODO: this argument should say use_bqsr_quality - OQ should be the default
	parser.add_argument('--use_original_quality', default = True, type=bool, help='Use original base qualities if available')
	parser.add_argument('--remove_reads_with_error', default = True, type=bool, help='remove reads with quality < 6 from training set')


	# Annotation arguments
	parser.add_argument('--annotations', help='Array of annotation names, initialised via annotation_set argument')
	parser.add_argument('--annotation_set', default='_', choices=ANNOTATIONS.keys(), help='Key which maps to an annotations list (or None for architectures that do not take annotations).')

	# Architecture defining arguments
	parser.add_argument('--conv_dropout', default=0.0, type=float, 
		help='Dropout rate in convolutional layers.')
	parser.add_argument('--conv_batch_normalize', default=False, action='store_true',
		help='Batch normalize convolutional layers.')
	parser.add_argument('--conv_layers', nargs='+', default=[128, 96, 64, 48], type=int,
		help='List of sizes for each convolutional filter layer')
	parser.add_argument('--conv_widths', nargs='+', default=[5, 5, 5, 5], type=int, 
		help='List of widths for each 1D convolutional kernels.')
	parser.add_argument('--kernel_initializer', default='glorot_normal',
		help='Kernel initializer for convolutional filter layers.')
	parser.add_argument('--activation', default='relu',
		help='Activation function for hidden units in neural nets dense layers.')	
	parser.add_argument('--padding', default='same',
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
	parser.add_argument('--fc_initializer', default='glorot_normal',
		help='Initializer for fully connected (dense) layers.')
	parser.add_argument('--resnet', default=False, action='store_true',
		help='Add residual connections around hidden layers.')
	parser.add_argument('--skip_connections', default=[0, 0, 0, 0], nargs='+', type=int,
		help='Residual connections. Should have the same number of elements as the conv_layers')

	# I/O files and directories: vcfs, bams, beds, hd5, fasta
	parser.add_argument('--semantics_json', default='')
	parser.add_argument('--tensor_name', default=BQSR_TENSOR_NAME, help='Key which looks up the map from tensor channels to their meaning.')
	parser.add_argument('--weights_hd5', default='', help='A hd5 file of weights to initialize a model, will use all layers with names that match.')
	parser.add_argument('--ignore_vcf', help='VCF of variant sites to ignore when making training data.')
	parser.add_argument('--bed_file', help='Bed file specifying high confident intervals.')
	parser.add_argument('--bam_file',  help='Path to a BAM file to train from or generate tensors with.')
	parser.add_argument('--data_dir', help='Directory of tensors, must be split into test/valid/train sets with directories for each label within.')
	parser.add_argument('--output_dir', default='./weights/', help='Directory to write models or other data out.')
	parser.add_argument('--reference_fasta', default=reference_fasta, help='The reference FASTA file (e.g. HG19 or HG38).')
	parser.add_argument('--out', default='recalibrated.bam', help='Path to the recalibrated BAM file')
	parser.add_argument('--model', help='Path to the trained model hd5 file')

	# Training and optimization related arguments
	parser.add_argument('--epochs', default=25, type=int, help='Number of epochs, typically passes through the entire dataset, not always well-defined.')
	parser.add_argument('--iterations', default=5, type=int, help='Generic iteration limit for hyperparameter optimization, animation, and other counts.')
	parser.add_argument('--patience', default=4, type=int, help='Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.')
	parser.add_argument('--training_steps', default=80, type=int, help='Number of training batches to examine in an epoch.')
	parser.add_argument('--validation_steps', default=40, type=int, help='Number of validation batches to examine in an epoch validation.')

	# Dataset generation related arguments
	parser.add_argument('--downsample_perfect_reads', default=0.0, type=float, 
		help='Rate of perfect reads that are downsampled must be in [0.0, 1.0].')	
	parser.add_argument('--valid_ratio', default=0.1, type=float,
		help='Rate of training tensors to save for validation must be in [0.0, 1.0].')	
	parser.add_argument('--test_ratio', default=0.2, type=float,
		help='Rate of training tensors to save for testing [0.0, 1.0].')	
	parser.add_argument('--valid_contigs', nargs='+', default=['18', '19', 'chr18', 'chr19'],
		help='Contigs to reserve for validation data in addition to those reserved by valid_ratio.')	
	parser.add_argument('--test_contigs', nargs='+', default=['20', '21', 'chr20', 'chr21'],
		help='Contigs to reserve for testing data in addition to those reserved by test_ratio.')	
	

	# Genomic position for parallelization
	parser.add_argument('--chrom', nargs='+', help='Chromosomes to load for parallel tensor writing.')
	parser.add_argument('--start_pos', default=0, type=int,
		help='Genomic position start for parallel tensor writing.')
	parser.add_argument('--end_pos', default=0, type=int,
		help='Genomic position end for parallel tensor writing.')


	# Run specific arguments
	parser.add_argument('--mode', help='High level recipe: write tensors, train, test or evaluate models.')
	parser.add_argument('--id', default='no_id', help='Identifier for this run, user-defined string to keep experiments organized.')
	parser.add_argument('--random_seed', default=12878, type=int, help='Random seed to use throughout run.  Always use np.random.')
	parser.add_argument('--samples', default=500, type=int)
	parser.add_argument('--inspect_model', default=False, action='store_true',
		help='Plot model architecture, measure inference and training speeds.')
	parser.add_argument('--inspect_show_labels', default=False, action='store_true',
		help='Plot model architecture with labels for each layer.')

	args = parser.parse_args()
	args.annotations = ANNOTATIONS[args.annotation_set]
	args.input_symbols = INPUT_SYMBOLS[args.input_symbol_set]
	np.random.seed(args.random_seed)
	print('Arguments are', args)
	K.set_learning_phase(0)

	assert len(args.conv_layers) == len(args.skip_connections)
	
	return args

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ High-Level Recipes ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bqsr_train_tensor(args):
	'''Trains the bqsr tensor architecture on tensors at the supplied data directory.

	Arguments:
		args.data_dir: must be set to an appropriate directory with
			subdirectories of test, valid and train, each containing
			bqsr labelled tensors stored as hd5 files. 

	This architecture looks at reads, flags and annotations.
	Tensors must be generated by calling write_base_recalibrate_tensors() before this function is used.
	After training with early stopping performance curves are plotted on the test dataset.
	'''
	with open(args.output_dir + "/arguments.txt", "w") as arg_log:
		print('Arguments are', args, file=arg_log)

	train_paths, valid_paths, test_paths = bqsr_get_train_valid_test_paths_all(args.data_dir)

	generate_train = bqsr_label_tensors_generator(args, train_paths, False)
	generate_valid = bqsr_label_tensors_generator(args, valid_paths, False)
	generate_test = bqsr_label_tensors_generator(args, test_paths, False)

	model = label_bases_model_from_args(args)
	if args.inspect_model:
		bqsr_inspect_model(args, model, generate_train, generate_valid, args.output_dir+args.id+IMAGE_EXT)
	model = bqsr_train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir+args.id+HD5_EXT)
	
	test = next(generate_test)
	bqsr_plot_roc_per_class(model, test[0][args.tensor_name], test[1], args.labels, args.id, melt=True)
	test_tensors = np.zeros((args.iterations*args.batch_size, args.window_size, len(args.input_symbols)))
	test_labels = np.zeros((args.iterations*args.batch_size, args.window_size, len(args.labels)))

	for i in range(args.iterations):
		next_batch = next(generate_test)
		test_tensors[i*args.batch_size:(i+1)*args.batch_size,:,:] = next_batch[0][args.tensor_name]
		test_labels[i*args.batch_size:(i+1)*args.batch_size,:,:] = next_batch[1]

	predictions = model.predict(test_tensors)
	print('prediction shape:', predictions.shape)

	melt_shape = (predictions.shape[0]*predictions.shape[1], predictions.shape[2])
	predictions = predictions.reshape(melt_shape)
	test_truth = test_labels.reshape(melt_shape)	
	bqsr_plot_precision_recall_per_class_predictions(predictions, test_truth, args.labels, args.id)

	with open(args.data_dir + "/finished.txt", "w") as f:
		print('finished at (TODO: time goes here)', file=f)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def label_bases_model_from_args(args):
	'''Build 1d CNN model for base recalibration from command line arguments.

	Architecture specified by parameters.
	Uses the functional API.
	Prints out model summary.

	Arguments
		args: The command-line arguments namespace 

	Returns
		The keras model
	'''	
	concat_axis = -1	
	x = read_tensor = Input(shape=(args.window_size, len(args.input_symbols)), name=args.tensor_name)
	xs = []
	
	max_pool_diff = len(args.conv_layers)-len(args.max_pools)
	for i, (c, w) in enumerate(zip(args.conv_layers, args.conv_widths)):
		in_x = x
		if args.conv_batch_normalize:
			x = Conv1D(filters=c, kernel_size=w, activation='linear', 
						padding=args.padding, kernel_initializer=args.kernel_initializer)(x)
			x = BatchNormalization(axis=concat_axis)(x)
			x = Activation(args.activation)(x)
		else:
			x = Conv1D(filters=c, kernel_size=w, activation=args.activation, 
						padding=args.padding, kernel_initializer=args.kernel_initializer)(x)

		if args.conv_dropout > 0 and args.spatial_dropout:
			x = SpatialDropout1D(args.conv_dropout)(x)
		elif args.conv_dropout > 0:
			x = Dropout(args.conv_dropout)(x)

		if i >= max_pool_diff:
			x = MaxPooling1D(args.max_pools[i-max_pool_diff])(x)

		xs.append(x)
		skip_origin_idx = i + int(args.skip_connections[i])
		if args.skip_connections[i] != 0:
			shortcut = xs[skip_origin_idx]
			if args.conv_batch_normalize:
				shortcut = BatchNormalization(axis=concat_axis)(shortcut)
			x = keras.layers.add([x, shortcut])   

	conv_label = Conv1D(len(args.labels), 1, activation='linear', padding='same')(x)
	conv_out = Activation('softmax', name='bqsr_labels')(conv_label)

	model = Model(inputs=read_tensor, outputs=conv_out)

	weighted_loss = bqsr_weighted_categorical_crossentropy(args.label_weights)

	model.compile(optimizer=Adam(lr=1e-4), loss=weighted_loss, metrics=bqsr_get_metrics(args.labels, dim=3))

	model.summary()
	
	return model


def bqsr_serialize_model_semantics(args, architecture_hd5):
	'''Save a json file specifying model semantics, I/O contract.

	Arguments
		args.tensor_name: String which indicates tensor map to use or None
		args.window_size: sites included in the tensor map
		args.read_limit: Maximum reads included in the tensor map
		args.annotations: List of annotations or None
		args.id: the id of the run will be the name of the semantics file
		architecture_hd5: Keras model and weights hd5 file (created with save_model())
	'''
	semantics = {
					'id' : args.id, 
					'output_labels' : args.labels, 
					'architecture' : os.path.basename(architecture_hd5),
					'input_symbols' : args.input_symbols,
				}

	if args.tensor_name:
		semantics['input_tensor_name'] = args.tensor_name
		semantics['input_tensor_map'] = args.input_symbols
		semantics['window_size'] = args.window_size
		semantics['read_limit'] = args.read_limit

	if args.annotation_set and args.annotation_set != '_':
		semantics['input_annotations'] = args.annotations
		semantics['input_annotation_set'] = args.annotation_set

	if args.data_dir:
		semantics['data_dir'] = args.data_dir

	semantics['channels_last'] = args.channels_last

	json_file_name = args.output_dir + args.id + '.json'
	with open(json_file_name, 'w') as outfile:
		json.dump(semantics, outfile)

	print('Saved model semantics at:', json_file_name)


def bqsr_set_args_and_get_model_from_semantics(args, semantics_json):
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

	if 'input_tensor_name' in semantics:
		args.tensor_name = semantics['input_tensor_name']
		args.window_size = semantics['window_size']
		args.read_limit = semantics['read_limit']
		assert(len(args.input_symbols) == len(semantics['input_tensor_name']))
		for key in args.input_symbols:
			assert(args.input_symbols[key] == semantics['input_tensor_name'][key])

	if 'input_annotations' in semantics:
		args.annotations = semantics['input_annotations']
		args.annotation_set = semantics['input_annotation_set']

	if 'channels_last' in semantics:
		args.channels_last = semantics['channels_last']
		if args.channels_last:
			K.set_image_data_format('channels_last')
		else:
			K.set_image_data_format('channels_first')
				
	args.input_symbols = semantics['input_symbols']
	args.labels = semantics['output_labels']

	weight_path_hd5 = os.path.join(os.path.dirname(semantics_json),semantics['architecture'])
	model = load_model(weight_path_hd5, custom_objects=bqsr_get_metric_dict(args.labels, args.label_weights))
	model.summary()
	return model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Writing and Generating Tensors ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def write_base_recalibrate_tensors(args, include_annotations=True):
	"""Create tensors structured as tensor map of read quality.

	Defines true bases as those not in args.db_snp, where read and reference agree.
	False bases are sites those not in args.db_snp where the read and the reference do NOT agree.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.ignore_vcf: VCF file with sites of known variation, from NIST, DBSNP etc.
		args.bed_file: Intervals of interest
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	"""	
	# NEED to label, set max size, go to only p-hot 4xwindow size, 
	# eliminate any known variant covering read
	# iterate over vcf
	# check between variants for reads
	print('Writing base recalibration tensors from tensor channel map.')
	stats = Counter()

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	vcf_ram = vcf.Reader(open(args.ignore_vcf, 'rb'))
	
	bed_dict = bqsr_bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))

	margin = args.window_size
    # TODO: int(c) below will not work with e.g. ChrX
	contigs = [ record_dict[c] for c in args.chrom ]
	for (i,c) in enumerate(args.chrom):
		for start,stop in zip(bed_dict[c][0], bed_dict[c][1]):
			if stop - start < args.window_size:
				stats['interval too small'] += 1
				continue
			last_variant = None
			if not any(vcf_ram.fetch(c, start, stop)):
				# no variants in the interval and the interval is larger than the window size
				write_reads_in_region_to_tensors(args, samfile, contigs[i], c, start, stop - margin, stats)
			else:
				for v in vcf_ram.fetch(c, start, stop):
					if last_variant is not None and (v.POS-last_variant.POS) > 2*args.window_size:
						write_reads_in_region_to_tensors(args, samfile, contigs[i], c, last_variant.POS+margin, v.POS-margin, stats)
					last_variant = v

			if stats['count'] >= args.samples:
				break


	with open(args.data_dir + "stats.txt", "w") as stats_log:
		for k in stats.keys():
			print('%s has %d' % (k, stats[k]), file=stats_log)

		p = stats_to_empirical_quality(stats, args)
		q = -10.0 * np.log10(p)

		print('empirical p:', p, ' and quality:', q, file=stats_log)
		print('Done generating BQSR tensors. Wrote them to:', args.data_dir ,' Known variation vcf:', args.ignore_vcf, file=stats_log)

	with open(args.data_dir + "arguments.txt", "w") as arg_log:
		print('Arguments are', args, file=arg_log)


def stats_to_empirical_quality(stats, args):
	num_bad_bases = stats[bad_bases_key]
	num_good_bases = stats[good_bases_key] + stats[perfect_read_key] * args.window_size
	return num_bad_bases / (num_bad_bases + num_good_bases)

def read_meets_data_qc(read, stats, remove_read_with_error_base=True):
	read_group = read.get_tag('RG')
	if 'artificial' in read_group.lower():
		return False
	if not read.is_proper_pair or not read.is_paired:
		return False
	if read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_qcfail or read.is_unmapped:
		return False
	if "S" in read.cigarstring:
		return False
	if remove_read_with_error_base and any(q < 6 for q in read.get_forward_qualities()):
		stats[dropped_error_q_key] += 1
		return False
	return True

def write_reads_in_region_to_tensors(args, samfile, chrom_seq, chrom, start, stop, stats):
	for read in samfile.fetch(chrom, start, stop):
		read_group = read.get_tag('RG')
		if not read_meets_data_qc(read, stats, args.remove_reads_with_error):
			continue

		assert(len(read.query_sequence) <= args.window_size)
		got_bad_base = False
	
		label_vector = np.zeros((args.window_size, len(BQSR_LABELS)))
		for ref_pos, read_idx in zip(read.get_reference_positions(), np.arange(len(read.query_sequence))):
			if chrom_seq[ref_pos] != read.query_sequence[read_idx]:
				label_vector[read_idx, BQSR_LABELS['BAD_BASE']] = 1.0
				got_bad_base = True
				stats[bad_bases_key] += 1
			else:
				label_vector[read_idx, BQSR_LABELS['GOOD_BASE']] = 1.0
				stats[good_bases_key] += 1

		if got_bad_base:
			stats[read_with_mismatch_key] += 1
			# TODO: a histogram of num mismatches would be nice
		else:
			stats[perfect_read_key] += 1
			dice = np.random.rand()
			if dice < args.downsample_perfect_reads:
				stats['downsampled perfect reads'] += 1
				continue

		# If the bam comes with OQ, then we will add the BQSR qualities to the tensor, too
		# Contract is that the user should always use OQ if it's available
		if args.use_original_quality:
			# note that the scope of variables in an if-statement is the function that contains it
			oq_tensor = read_to_bqsr_tensor(read, args, use_oq=True)
			bqsr_tensor = read_to_bqsr_tensor(read, args, use_oq=False)
		else:
			bqsr_tensor = read_to_bqsr_tensor(read, args, use_oq=False)


		if len(args.annotations) > 0:
			annotation_data = np.zeros(( len(args.annotations), ))
			for i,a in enumerate(args.annotations):
				if a.lower() == 'strand':
					annotation_data[i] = float(read.is_reverse)
				elif a.lower() == 'pair':
					annotation_data[i] = float(read.is_read1)
				elif a.lower() == 'mappingquality':
					annotation_data[i] = float(read.mapping_quality) / max_mq
				elif a.lower() == 'cycle':
					annotation_data[i] = float(read_idx) / args.window_size

		tensor_path = bqsr_get_path_to_train_valid_or_test(args, chrom)
		tensor_prefix = bqsr_plain_name(args.bam_file) +'_'+ bqsr_plain_name(args.ignore_vcf)
		tensor_path += tensor_prefix + '-' + read_group.replace(':', '')+ chrom +'_'+ str(read.reference_start) + '.hd5'
		if not os.path.exists(os.path.dirname(tensor_path)):
			os.makedirs(os.path.dirname(tensor_path))
		with h5py.File(tensor_path, 'w') as hf:
			if args.use_original_quality:
				hf.create_dataset(OQ_TENSOR_NAME, data=oq_tensor, compression='gzip')
			hf.create_dataset(BQSR_TENSOR_NAME, data=bqsr_tensor, compression='gzip')
			if len(args.annotations) > 0:
				hf.create_dataset(args.annotation_set, data=annotation_data, compression='gzip')
			hf.create_dataset(LABEL_TENSOR_NAME, data=label_vector, compression='gzip')
		stats['count'] += 1
		if stats['count']%400 == 0:
			print('Wrote', stats['count'], 'tensors out of', args.samples)
	
def read_to_bqsr_tensor(read, args, use_oq=True):
	bqsr_tensor = np.zeros((args.window_size, len(args.input_symbols)))
	base_qualities = read.get_forward_qualities() if not use_oq else get_forward_oq(read)

	bqsr_tensor[:, :4] = bqsr_base_string_to_tensor(args, read.get_forward_sequence(), base_qualities.tolist())
	for a in args.input_symbols:
		if a.lower() == 'strand':
			bqsr_tensor[:, args.input_symbols[a]] = float(read.is_reverse)
		elif a.lower() == 'pair':
			bqsr_tensor[:, args.input_symbols[a]] = float(read.is_read1)
		elif a.lower() == 'mq':
			bqsr_tensor[:, args.input_symbols[a]] = float(read.mapping_quality) / float(MAX_MQ)
		elif a.lower() == 'cycle':
			bqsr_tensor[:, args.input_symbols[a]] = np.arange(args.window_size) / float(args.window_size)
	return bqsr_tensor

def get_forward_oq(read):
	''' get the original qualities if available '''
	OQ_TAG = 'OQ'
	if not read.has_tag(OQ_TAG):
		sys.exit("original quality requested but unavailable")
	oq_string = read.get_tag(OQ_TAG)[::-1] if read.is_reverse else read.get_tag(OQ_TAG)
	return array('b', np.array([ord(elem) for elem in oq_string]) - 33) # don't forget to subtract the 33


def bqsr_get_path_to_train_valid_or_test(args, contig):
	if any(x == contig for x in args.valid_contigs):
		return os.path.join(args.data_dir, 'valid/')
	if any(x == contig for x in args.test_contigs):
		return os.path.join(args.data_dir, 'test/')

	dice = np.random.rand()
	if dice < args.valid_ratio:
		return os.path.join(args.data_dir, 'valid/')
	elif dice < (args.valid_ratio+args.test_ratio):	
		return os.path.join(args.data_dir, 'test/')
	else:	
		return os.path.join(args.data_dir, 'train/')


def bqsr_base_string_to_tensor(args, bases, qualities):
	assert(len(bases) <= args.window_size)
	tensor = np.zeros( (args.window_size, len(DNA_SYMBOLS)) )

	for i,b in enumerate(bases):
		if b in DNA_SYMBOLS:
			tensor[i, :len(DNA_SYMBOLS)] = bqsr_quality_from_mode(args, qualities[i], b)
		elif b in AMBIGUITY_CODES:
			tensor[i, :len(DNA_SYMBOLS)] = AMBIGUITY_CODES[b]
		elif b == SKIP_CHAR:
			continue
		else:
			raise ValueError('Error! Unknown symbol in seq block:', b)
	
	return tensor


def bqsr_base_quality_to_phred_array(base_quality, base, base_dict):
	phred = np.zeros((4,))
	exponent = float(-base_quality) / 10.0
	p = 1.0-(10.0**exponent) # Convert to probability
	not_p = (1.0-p) / 3.0 # Error could be any of the other 3 bases
	not_base_quality = -10 * np.log10(not_p) # Back to Phred
	
	for b in DNA_SYMBOLS:
		if b == INDEL_CHAR:
			continue
		elif b == base:
			phred[base_dict[b]] = base_quality
		else:
			phred[base_dict[b]] = not_base_quality
	return phred


def bqsr_base_quality_to_p_hot_array(base_quality, base, base_dict):
	if base_quality <= 6:
		return 0.25*np.ones(4)
	p_hot = np.zeros((4,))
	exponent = float(-base_quality) / 10.0
	p = 1.0-(10.0**exponent)
	not_p = (1.0-p)/3.0

	for b in DNA_SYMBOLS:
		if b == base:
			p_hot[base_dict[b]] = p
		elif b == INDEL_CHAR:
			continue
		else:
			p_hot[base_dict[b]] = not_p

	return p_hot


def bqsr_quality_from_mode(args, base_quality, base):
	base_dict=args.input_symbols
	if args.base_quality_mode == 'phot':
		return bqsr_base_quality_to_p_hot_array(base_quality, base, base_dict)
	elif args.base_quality_mode == 'phred':
		return bqsr_base_quality_to_phred_array(base_quality, base, base_dict)
	elif args.base_quality_mode == '1hot':
		one_hot = np.zeros((4,))
		one_hot[DNA_SYMBOLS[base]] = 1.0
		return one_hot
	else:
		raise ValueError('Error! Unknown base quality mode:', args.base_quality_mode)


def bqsr_plain_name(full_name):
	name = os.path.basename(full_name)
	return name.split('.')[0]


def bqsr_label_tensors_generator(args, train_paths, include_bqsr=True):
	'''Data generator of read tensors for calling variants and site labels for segmentation ground truth.

	Loops over all examples yielding args.batch_size examples.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: directory with hd5 calling tensors made with write_calling_tensors()

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels
	'''


	'''
	For evaluation purposes it's nice to store both OQ and post-BQSR qualities in the hd5 file.
	The rule is the caller must explicitly specify 'bqsr' or 'oq' in order to access these tensors.
	For now I will always use OQ, but in the future, if I start using novaseq and what now, I may need to revisit this code.
	Maybe the oq tensor would be empty when it is not available.
	'''
	tensors = {}
	i = 0

	tensor = np.zeros((args.batch_size, args.window_size, len(args.input_symbols)))
	bqsr_tensor = np.zeros((args.batch_size, args.window_size, len(args.input_symbols)))
	label_matrix = np.zeros((args.batch_size, args.window_size, len(args.labels)))
	print('batch shape is:', tensor.shape)
	while True:
		for tp in train_paths:
			try:
				with h5py.File(tp, 'r') as hf:
					tensor[i] = np.array(hf.get(OQ_TENSOR_NAME))
					bqsr_tensor[i] = np.array(hf.get(BQSR_TENSOR_NAME))

					if args.map_input_to_logspace:
						log_tmp = np.log(tensor[i,:,:4])
						tensor[i,:,:4] = log_tmp
					label_matrix[i] = np.array(hf.get(LABEL_TENSOR_NAME))

			except Exception as e:
				print('Exception for tensor at:', tp, '\n\n\nError is:', str(e))
				print('Expected tensor shape:', (args.batch_size, args.window_size, len(args.input_symbols))) #, 'but received shape:', np.array(hf.get('read_tensor')).shape)
				print('Expected site labels shape:',(args.window_size, len(args.labels))) #, 'received:', np.array(hf.get('site_labels')).shape)
				raise Exception('bad tensor')
				#continue

			i += 1
			if i == args.batch_size:
				if not include_bqsr:
					yield ({OQ_TENSOR_NAME: tensor}, label_matrix)
				else:
					yield ({OQ_TENSOR_NAME : tensor, BQSR_TENSOR_NAME : bqsr_tensor}, label_matrix)
				i = 0

		print('\n\nGenerator looped over all ', len(train_paths),' tensors, now shuffle them. Last tensor was:', train_paths[-1])
		np.random.shuffle(train_paths)


def bqsr_position_string_from_tensor_name(tensor_name):
	'''Genomic position as underscore delineated string from a filename.

	Includes an allele index if the filename includes _allele_
	This is ugly, we need file names ending with genomic position 
	(e.g. my_tensor-12_1234.hd5 returns 12_1234 and a_tensor_allele_1-8_128.hd5 returns 8_128_1)

	Arguments:
		tensor_name: the filename to parse
	Returns:
		Genomic position string Contig_Position or Contig_Position_AlleleIndex
	'''
	slash_split = tensor_name.split('/')
	dash_split = slash_split[-1].split('-')
	gsplit = dash_split[0].split('_')

	gpos = dash_split[-1]
	chrom = gpos.split('_')[0]
	pos = os.path.splitext(gpos.split('_')[1])[0]
	pos_str = chrom + '_' + pos
	
	for i,p in enumerate(gsplit):
		if p == 'allele':
			pos_str += '_'+str(gsplit[i+1])

	return pos_str	


def bqsr_get_train_valid_test_paths_all(data_dir):
	train_dir = data_dir + 'train/'
	valid_dir = data_dir + 'valid/'
	test_dir = data_dir + 'test/'
	train_paths = [train_dir + tp for tp in os.listdir(train_dir)]
	valid_paths = [valid_dir + vp for vp in os.listdir(valid_dir)]
	test_paths = [test_dir + vp for vp in os.listdir(test_dir)]

	return train_paths, valid_paths, test_paths


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training ~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bqsr_train_model_from_generators(args, model, generate_train, generate_valid, save_weight_hd5):
	'''Train an image model for classifying variants.

	Training data lives on disk, it will be loaded by generator functions.
	Plots the metric history after training. Creates a directory to save weights at if necessary.

	Arguments
		args.batch_size: size of the mini-batches
		args.patience: Maximum number of epochs to run without validation loss improvement
		args.epochs: Maximum number of epochs to run regardless of Early Stopping
		args.training_steps: Number of mini-batches in each so-called epoch
		args.validation_steps: Number of validation mini-batches to examine after each epoch.
		model: the model to optimize
		generate_train: training data generator function	
		valid_tuple: Validation data data generator function
		save_weight_hd5: path to save the model weights at

	Returns
		The now optimized keras model
	'''
	if not os.path.exists(os.path.dirname(save_weight_hd5)):
		os.makedirs(os.path.dirname(save_weight_hd5))	
	bqsr_serialize_model_semantics(args, save_weight_hd5)

	# if args.inspect_model:
	# 	image_path = args.id+'.png' if args.image_dir is None else args.image_dir+args.id+'.png'
	# 	inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

	history = model.fit_generator(generate_train, 
		steps_per_epoch=args.training_steps, epochs=args.epochs, verbose=1, 
		validation_steps=args.validation_steps, validation_data=generate_valid,
		callbacks=bqsr_get_callbacks(args, save_weight_hd5))

	bqsr_plot_metric_history(history, bqsr_weight_path_to_title(save_weight_hd5))
	print('Model weights saved at: %s' % save_weight_hd5)
	
	return model


def bqsr_get_callbacks(args, save_weight_hd5):
	callbacks = []
	
	callbacks.append(ModelCheckpoint(filepath=save_weight_hd5, verbose=1, save_best_only=True))
	callbacks.append(EarlyStopping(monitor='val_loss', patience=args.patience*3, verbose=1))
	callbacks.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=args.patience, verbose=1))
	
	return callbacks
		

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Plotting ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bqsr_plot_metric_history(history, title, prefix='./figures/'):
	# list all data in history
	print(history.history.keys())
	num_plots = len ([k for k in history.history.keys() if not 'val' in k])

	row = 0
	col = 0
	rows = 4
	cols = max(2, int(math.ceil(num_plots/float(rows))))

	f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))
	for k in sorted(history.history.keys()):
		if 'val' not in k:
			axes[row, col].plot(history.history[k])
			axes[row, col].set_ylabel(str(k))
			axes[row, col].set_xlabel('epoch')
			if 'val_'+k in history.history:
				axes[row, col].plot(history.history['val_'+k])
				labels = ['train', 'valid']
			else:
				labels = [k] 
			axes[row, col].legend(labels, loc='upper left')

			row += 1
			if row == rows:
				row = 0
				col += 1
				if row*col >= rows*cols:
					break

	axes[0, 1].set_title(title)
	figure_path = prefix+"metric_history_"+title+IMAGE_EXT
	if not os.path.exists(os.path.dirname(figure_path)):
		os.makedirs(os.path.dirname(figure_path))	
	plt.savefig(figure_path)	


def bqsr_plot_roc_per_class(model, test_data, test_truth, labels, title, batch_size=32, prefix='./figures/', melt=False):
	fpr, tpr, roc_auc = bqsr_get_fpr_tpr_roc(model, test_data, test_truth, labels, batch_size, melt)

	lw = 3
	plt.figure(figsize=(28,22))
	matplotlib.rcParams.update({'font.size': 34})	
	

	for key in labels.keys():
		if key in KEY_COLORS:
			color = KEY_COLORS[key]
		else:
			color = np.random.choice(color_array)
		plt.plot( fpr[labels[key]], tpr[labels[key]], color=color, lw=lw, label=str(key)+' area under ROC: %0.3f'%roc_auc[labels[key]]  )


	plt.plot([0, 1], [0, 1], 'k:', lw=0.5)
	plt.xlim([0.0, 1.0])
	plt.ylim([-0.02, 1.03])
	plt.xlabel(fallout_label)
	plt.ylabel(recall_label)
	plt.title('ROC:'+ title + '\n')

	matplotlib.rcParams.update({'font.size': 56})	
	plt.legend(loc="lower right")
	figure_path = prefix+"per_class_roc_"+title+IMAGE_EXT
	if not os.path.exists(os.path.dirname(figure_path)):
		os.makedirs(os.path.dirname(figure_path))
	plt.savefig(figure_path)
	print('Saved figure at:', figure_path)


def bqsr_plot_precision_recall_per_class_predictions(predictions, truth, labels, title, prefix='./figures/'):
	# Compute Precision-Recall and plot curve
	precision = dict()
	recall = dict()
	average_precision = dict()
	lw = 4.0
	plt.figure(figsize=(22,18))
	matplotlib.rcParams.update({'font.size': 34})	
	
	for k in labels.keys():
		if k in KEY_COLORS:
			c = KEY_COLORS[k]
		else:
			c = np.random.choice(color_array)
		
		precision[k], recall[k], _ = precision_recall_curve(truth[:, labels[k]], predictions[:, labels[k]])
		average_precision[k] = average_precision_score(truth[:, labels[k]], predictions[:, labels[k]])
		plt.plot(recall[k], precision[k], lw=lw, color=c, label=k+' area = %0.3f' % average_precision[k])

	plt.ylim([-0.02, 1.03])
	plt.xlim([0.0, 1.00])
	
	plt.xlabel(recall_label)
	plt.ylabel(precision_label)
	plt.title(title)

	plt.legend(loc="lower left")
	
	plot_name = prefix+"precision_recall_"+title+IMAGE_EXT
	if not os.path.exists(os.path.dirname(plot_name)):
		os.makedirs(os.path.dirname(plot_name))		
	plt.savefig(plot_name)
	print('Saved plot at:%s' % plot_name)



def bqsr_inspect_model(args, model, generate_train, generate_valid, image_path=None):
	'''Collect statistics on model inference and training times.

	Arguments
		args.samples: number of optimization steps to take
		args.batch_size: size of the mini-batches
		model: the model to inspect
		generate_train: training data generator function	
		generate_valid: Validation data generator function

	Returns
		The slightly optimized keras model
	'''
	if image_path:
		bqsr_plot_dot_model_in_color(model_to_dot(model, show_shapes=True), image_path)

	t0 = time.time()
	history = model.fit_generator(generate_train, steps_per_epoch=args.training_steps, epochs=1, verbose=1, validation_steps=5, validation_data=generate_valid)
	t1 = time.time()
	train_speed = (t1-t0)/(args.batch_size*args.training_steps)
	print('Spent: ', t1-t0, ' seconds training, batch_size:', args.batch_size, 'steps:', args.training_steps, ' Per example training speed:', train_speed)

	t0 = time.time()
	predictions = model.predict_generator(generate_valid, steps=args.training_steps, verbose=1)
	t1 = time.time()
	inference_speed = (t1-t0)/(args.batch_size*args.training_steps)
	print('Spent: ', t1-t0, ' seconds predicting. Per tensor inference speed:', inference_speed)
	
	return model


def bqsr_plot_dot_model_in_color(dot, image_path):
	for n in dot.get_nodes():
		if n.get_label():
			if 'Conv1' in n.get_label():
				n.set_fillcolor("cyan")
			elif 'Conv2' in n.get_label():
				n.set_fillcolor("deepskyblue1")				
			elif 'BatchNormalization' in n.get_label():
				n.set_fillcolor("goldenrod1")		
			elif 'softmax' in n.get_label():
				n.set_fillcolor("chartreuse")										
			elif 'Activation' in n.get_label():
				n.set_fillcolor("yellow")	
			elif 'MaxPooling' in n.get_label():
				n.set_fillcolor("aquamarine")
			elif 'Dense' in n.get_label():
				n.set_fillcolor("gold")
			elif 'Flatten' in n.get_label():
				n.set_fillcolor("coral3")
			elif 'Reshape' in n.get_label():
				n.set_fillcolor("coral")			
			elif 'Input' in n.get_label():
				n.set_fillcolor("darkolivegreen1")
			elif 'Concatenate' in n.get_label():
				n.set_fillcolor("darkorange")
			elif 'Dropout' in n.get_label():
				n.set_fillcolor("tomato")
		n.set_style("filled")
	print('Saving architecture diagram to:',image_path)
	dot.write_png(image_path)


def bqsr_weight_path_to_title(wp):
	return wp.split('/')[-1].replace('__', '-').split('.')[0]


def bqsr_get_fpr_tpr_roc(model, test_data, test_truth, labels, batch_size=32, melt=False):
	y_pred = model.predict(test_data, batch_size=batch_size, verbose=0)
	
	if melt:		
		melt_shape = (y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2])
		y_pred = y_pred.reshape(melt_shape)
		test_truth = test_truth.reshape(melt_shape)

	return bqsr_get_fpr_tpr_roc_pred(y_pred, test_truth, labels)


def bqsr_get_fpr_tpr_roc_pred(y_pred, test_truth, labels):
	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	for k in labels.keys():
		cur_idx = labels[k]
		fpr[labels[k]], tpr[labels[k]], _ = roc_curve(test_truth[:,cur_idx], y_pred[:,cur_idx])
		roc_auc[labels[k]] = auc(fpr[labels[k]], tpr[labels[k]])

	return fpr, tpr, roc_auc



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bqsr_get_metric_dict(labels=BQSR_LABELS, label_weights=[0.05, 0.95]):
	metrics = {}
	precision_fxns = per_class_recall_3d(labels)
	recall_fxns = per_class_recall_3d(labels)
	for i,label_key in enumerate(labels.keys()):
		metrics[label_key+'_precision'] = precision_fxns[i]
		metrics[label_key+'_recall'] = recall_fxns[i]
	metrics['loss'] = bqsr_weighted_categorical_crossentropy(label_weights)
	# on hold for now
	# metrics['KL_divergence'] = KL_divergence
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


def bqsr_get_metrics(classes=None, dim=2):
    if classes and dim == 3:
        return [metrics.categorical_accuracy] + per_class_precision_3d(classes) + per_class_recall_3d(classes)
    else:
        return [metrics.categorical_accuracy]


def bqsr_weighted_categorical_crossentropy(weights):
	"""
	A weighted version of keras.objectives.categorical_crossentropy

	Variables:
	weights: numpy array of shape (C,) where C is the number of classes

	Usage:
	weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
	loss = weighted_categorical_crossentropy(weights)
	model.compile(loss=loss,optimizer='adam')
	"""

	weights = K.variable(weights)

	def loss(y_true, y_pred):
		# scale predictions so that the class probs of each sample sum to 1
		y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
		# clip to prevent NaN's and Inf's
		y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
		# calc
		loss = y_true * K.log(y_pred) * weights
		loss = -K.sum(loss, -1)
		return loss

	return loss

# def KL_divergence(y_true, y_pred):
# 	'''
# 	Compute the KL divergence between the match bases and the mismatch bases
# 	'''
#
# 	# TODO: for this to be a metric it needs to use the backend functions, not numpy
# 	predicted_qs = -10 * np.log10(y_pred[:, :, args.labels['BAD_BASE']])
# 	match_qs = (predicted_qs[:, :, np.newaxis] * y_true)[:, :, args.labels['GOOD_BASE']]
# 	mismatch_qs = (predicted_qs[:, :, np.newaxis] * y_true)[:, :, args.labels['BAD_BASE']]
# 	match_qs = match_qs[match_qs > 0]
# 	mismatch_qs = mismatch_qs[mismatch_qs > 0]
#
# 	# bins are half open: 1 will go in the [1,2) bin
# 	max_quality = 50
# 	match_hist, match_bins = np.histogram(np.round(match_qs), bins=max_quality, range=(0, max_quality))
# 	mismatch_hist, mismatch_bins = np.histogram(np.round(mismatch_qs), bins=max_quality, range=(0, max_quality))
#
# 	# compute the KL divergence KL(match||mismatch) - the order chosen arbitrariliy i.e. could've easily chosen KL(mismatch||match)
# 	# mask bins with 0 probability mass because numpy doens't know 0*log(0)=0
# 	ma_match_hist = ma.array(match_hist, mask=match_hist == 0)
# 	ma_mismatch_hist = ma.array(mismatch_hist, mask=match_hist == 0)
# 	KL = -ma.sum(ma_match_hist * ma.log(ma_mismatch_hist)) - ma.sum(- ma_match_hist * ma.log(ma_match_hist))
# 	return KL


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Utilities ~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def bqsr_bed_file_to_dict(bed_file, shift1=1):
	''' Create a dict to store intervals from a bed file.

	Arguments:
		bed_file: the file to load
		shift1: Shift the bed file 1 position over to align with 1-indexed VCFs

	Returns:
		bed: dict where keys in the dict are contig ids
			values are a tuple of arrays the first array 
			in the tuple contains the start positions
			the second array contains the end positions.
	'''
	bed = {}
	assert(shift1 == 0 or shift1 == 1)

	with open(bed_file)as f:
		for line in f:
			parts = line.split()
			contig = parts[0]
			lower = int(parts[1])+shift1 # nice1
			upper = int(parts[2])+shift1

			if contig not in bed:
				bed[contig] = ([], [])

			bed[contig][0].append(lower)
			bed[contig][1].append(upper)

	for k in bed.keys():
		bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]))		

	return bed

if __name__ == '__main__':
	run() # Back to the top!
