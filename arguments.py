# arguments.py
#
# Command Line Arguments for Variant Filtration with Neural Nets
# Shared by both recipes.py and unit_tests.py and hyperparameter_optimizer.py
# These arguments are a bit of a hodge-podge and are used promiscuously throughout these files.
# Sometimes code overwrites user-provided arguments to enforce assumptions or sanity.
#
# May 2017
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import print_function

# Imports
import os
import defines
import argparse
import numpy as np
import keras.backend as K

def parse_args():
	parser = argparse.ArgumentParser()

	# Required mode argument: what would you like to do?
	parser.add_argument('mode', help='High level recipe: write tensors, train, test or evaluate models.')
	

	# Tensor defining dictionaries
	parser.add_argument('--tensor_map', default='read_tensor',
		help='Key which looks up the map from tensor channels to their meaning:'+str(defines.architectures.keys()))
	parser.add_argument('--input_symbols', default=defines.inputs_indel,
		help='Dict mapping input symbols to their index within input tensors.')
	parser.add_argument('--labels', default=defines.snp_indel_labels,
		help='Dict mapping label names to their index within label tensors.')
	

	# Tensor defining arguments
	parser.add_argument('--batch_size', default=32, type=int,
		help='Mini batch size for stochastic gradient descent algorithms.')
	parser.add_argument('--read_limit', default=128, type=int,
		help='Maximum number of reads to load.')
	parser.add_argument('--window_size', default=128, type=int,
		help='Size of sequence window to use as input, typically centered at a variant.')
	parser.add_argument('--channels_last', default=True, dest='channels_last', action='store_true',
		help='Store the channels in the last axis of tensors, tensorflow->true, theano->false')				
	parser.add_argument('--channels_first', dest='channels_last', action='store_false',
		help='Store the channels in the first axis of tensors, tensorflow->false, theano->true')	
	parser.add_argument('--base_quality_mode', default='phot', choices=['phot', 'phred', '1hot'],
		help='How to treat base qualities, must be in [phot, phred, 1hot]')


	# Annotation arguments
	parser.add_argument('--annotations', help='Array of annotation names, initialised via annotation_set argument')
	parser.add_argument('--annotation_set', default='best_practices', choices=defines.annotations.keys(),
		help='Key which maps to an annotations list (or None for architectures that do not take annotations).')
	parser.add_argument('--normalize_annotations', default=False, action='store_true',
		help='If true tensor generators will look for mean and std files and normalize annotations.')
	parser.add_argument('--max_normalize_sites', default=1e9,
		help='Maximum number of sites from which to derive normalization values.')
	parser.add_argument('--sample_name', default='NA12878',
		help='The sample name from which to gather genotype information from the negative VCF.')


	# Training and optimization related arguments
	parser.add_argument('--epochs', default=25, type=int,
		help='Number of epochs, typically passes through the entire dataset, not always well-defined.')	
	parser.add_argument('--batch_normalization', default=False, action='store_true',
		help='Mini batch normalization layers after convolutions.')
	parser.add_argument('--samples', default=500, type=int,
		help='Maximum number of data samples to write or load.')
	parser.add_argument('--patience', default=4, type=int,
		help='Early Stopping parameter: Maximum number of epochs to run without validation loss improvements.')
	parser.add_argument('--training_steps', default=80, type=int,
		help='Number of training batches to examine in an epoch.')
	parser.add_argument('--validation_steps', default=40, type=int,
		help='Number of validation batches to examine in an epoch validation.')
	parser.add_argument('--iterations', default=5, type=int,
		help='Generic iteration limit for hyperparameter optimization, animation, and other counts.')


	# Dataset generation related arguments
	parser.add_argument('--downsample_snps', default=1.0, type=float,
		help='Rate of SNP examples that are kept must be in [0.0, 1.0].')	
	parser.add_argument('--downsample_indels', default=1.0, type=float,
		help='Rate of INDEL examples that are kept must be in [0.0, 1.0].')	
	parser.add_argument('--downsample_not_snps', default=1.0, type=float,
		help='Rate of NOT_SNP examples that are kept must be in [0.0, 1.0].')	
	parser.add_argument('--downsample_not_indels', default=1.0, type=float,
		help='Rate of NOT_INDEL examples that are kept must be in [0.0, 1.0].')	
	parser.add_argument('--downsample_reference', default=0.001, type=float,
		help='Rate of reference genotype examples that are kept must be in [0.0, 1.0].')		
	parser.add_argument('--downsample_homozygous', default=0.001, type=float,
		help='Rate of homozygous genotypes that are kept must be in [0.0, 1.0].')	
	parser.add_argument('--start_pos', default=0, type=int,
		help='Genomic position start for parallel tensor writing.')
	parser.add_argument('--end_pos', default=0, type=int,
		help='Genomic position end for parallel tensor writing.')
	parser.add_argument('--skip_positive_class', default=False, action='store_true',
		help='Whether to skip positive examples when writing tensors.')
	parser.add_argument('--chrom', help='Chromosome to load for parallel tensor writing.')


	# Input files and directories: vcfs, bams, beds, hd5, fasta
	parser.add_argument('--image_dir', default=None, help='Directory to write images and plots to.')
	parser.add_argument('--weights_hd5', default='',
		help='A hd5 file of weights to initialize a model, will use all layers with names that match.')
	parser.add_argument('--architecture', default='',
		help='A hd5 file of specifying weights and architecture of a neural net.')
	parser.add_argument('--architectures', nargs='+',
		help='Specify one or more architecture configuration files.')
	parser.add_argument('--bam_file', default=defines.bam_file,
		help='Path to a BAM file to train from or generate tensors with.')
	parser.add_argument('--train_vcf', default=defines.nist_vcf,
		help='Path to a VCF that has verified true calls from NIST, platinum genomes, etc.')
	parser.add_argument('--annotation_vcf', default=None,
		help='Path to a VCF that has annotations (typically from Haplotype Caller).')
	parser.add_argument('--negative_vcf', default=defines.negative_vcf,
		help='Haplotype Caller or VQSR generated VCF with raw annotation values [and quality scores].')
	parser.add_argument('--negative_vcf_2', default=None,
		help='Additional Haplotype Caller or VQSR generated VCF with raw annotation values [and quality scores].')
	parser.add_argument('--negative_vcf_3', default=None,
		help='Additional Haplotype Caller or VQSR generated VCF with raw annotation values [and quality scores].')
	parser.add_argument('--ignore_vcf', default=None,
		help='Optional VCF of sites to ignore when doing evaluations.')
	parser.add_argument('--include_vcf', default=None,
		help='Optional VCF of sites to include while ignoring all other sites when doing evaluations.')
	parser.add_argument('--output_vcf', default=None,
		help='Optional VCF to write to.')
	parser.add_argument('--output_dir', default='./weights/',
		help='Directory to write models or other data out.')	
	parser.add_argument('--deep_variant_vcf', default=None,
		help='Optional VCF with Google deep variant QUAL scores for comparisons.')	
	parser.add_argument('--bed_file', default=defines.nist_bed_file,
		help='Bed file specifying high confidence intervals associated with args.train_vcf.')
	parser.add_argument('--data_dir', default=defines.data_dir,
		help='Directory of tensors, must be split into test/valid/train sets with directories for each label within.')
	parser.add_argument('--reference_fasta', default=defines.reference_fasta,
		help='The reference FASTA file (e.g. HG19 or HG38).')


	# Evaluation related arguments
	parser.add_argument('--multiallelics', default='include', choices=['include', 'only', 'ignore'],
		help='How to handle multiallelic sites: can be include, only, or ignore. Only used in gnomad evaluation.')
	parser.add_argument('--random_forest_training_sites', default='ignore', choices=['include', 'only', 'ignore'],
		help='How to handle Random Forest Training sites: can be include, only, or ignore. Only used in gnomad evaluation.')
	parser.add_argument('--emit_interesting_sites', default=False, action='store_true',
		help='Emit sites where classification algorithms disagree or of extreme CNN scores. Only used in gnomad evaluation.')
	parser.add_argument('--single_sample_vqsr', default=False, action='store_true',
		help='Include single sample VQSR results in ROC curve. Only used in gnomad evaluation.')
	parser.add_argument('--gnomad_ac_max', default=1e10, type=int,
		help='gnomAD allele count maximum value, set arbitrarily high by default.')
	parser.add_argument('--gnomad_ac_min', default=0, type=int,
		help='gnomAD allele count minimum value, set to 0 by default.')
	parser.add_argument('--score_keys', nargs='+', default=['VQSLOD'],
		help='List of variant score keys for performance comparisons.')
	parser.add_argument('--inspect_model', default=False, action='store_true',
		help='Plot model architecture, measure inference and training speeds.')

	# Run specific arguments
	parser.add_argument('--id', default='no_id',
		help='Identifier for this run, user-defined string to keep experiments organized.')	
	parser.add_argument('--random_seed', default=12878, type=int,
		help='Random seed to use throughout run.  Always use np.random.')


	# Parse, print, set annotations, image data format and seed
	args = parser.parse_args()
	args.annotations = defines.annotations_from_args(args)
	np.random.seed(args.random_seed)

	if args.channels_last:
		K.set_image_data_format('channels_last')
	else:
		K.set_image_data_format('channels_first')

	print('Arguments are', args)
	
	return args


def weight_path_from_args(args):
	'''Create a weight file name from the command line arguments

	Arguments:
		args: puts arguments into the file name skips args in the ignore array
	'''		
	save_weight_hd5 =  args.output_dir + args.id + '.hd5'
	print('save weight path:' , save_weight_hd5)
	return save_weight_hd5

