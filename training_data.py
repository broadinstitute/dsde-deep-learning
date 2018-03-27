# training_data.py
#
# This file contains all the nitty-gritty of training data generation for neural networks.
# Reads input files, BAMs VCFs, BEDs, etc. and transforms them into tensors (or images).
# The generated data is of set dimension for processing by a particular neural net architecture.
# Supports division of training, testing and validation data.
# Provides basic queries for variants from vcfs and intervals from beds. 
#
# December 2016
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
import math
import h5py
import plots
import errno
import pysam
import random
import defines
import operator
import arguments
import numpy as np

from random import shuffle
from Bio import Seq, SeqIO
from scipy.stats import norm
from collections import Counter, defaultdict

tensor_exts = ['.h5', '.hd5']


def run_training_data():
	'''Dispatch on args.mode command-line supplied recipe'''
	args = arguments.parse_args()

	# Writing tensor datasets for training
	if 'write_tensors' == args.mode:
		tensors_from_tensor_map(args, include_annotations=True)
	elif 'write_paired_read_tensors' == args.mode:
		paired_read_tensors_from_map(args, include_annotations=True)
	elif 'write_tensors_2bit' == args.mode:
		tensors_from_tensor_map_2channel(args, include_annotations=True)
	elif 'write_tensors_no_annotations' == args.mode:
		tensors_from_tensor_map(args, include_annotations=False)
	elif 'write_tensors_gnomad_annotations' == args.mode:
		tensors_from_tensor_map_gnomad_annos(args)
	elif 'write_tensors_gnomad_annotations_per_allele_1d' == args.mode:
		tensors_from_tensor_map_gnomad_annos_per_allele(args, include_reads=False, include_reference=True)
	elif 'write_tensors_gnomad_1d' == args.mode:
		tensors_from_tensor_map_gnomad_annos(args, include_reads=False, include_reference=True)		
	elif 'write_depristo' == args.mode:
		nist_samples_to_png(args)
	elif 'write_calling_tensors' == args.mode:
		calling_tensors_from_tensor_map(args)
	elif 'write_pileup_filter_tensors' == args.mode:
		tensors_from_tensor_map(args, pileup=True)		
	elif 'write_calling_tensors_1d' == args.mode:
		calling_tensors_from_tensor_map(args, pileup=True)		
	elif 'write_dna_tensors' == args.mode:
		write_dna_and_annotations(args)
	elif 'write_bed_tensors' == args.mode:
		write_dna_multisource_annotations(args)
	elif 'write_bed_tensors_dna' == args.mode:
		write_dna_multisource_annotations(args, include_annotations=False)		
	elif 'write_bed_tensors_annotations' == args.mode:
		write_dna_multisource_annotations(args, include_dna=False)	
	elif 'write_bqsr_tensors' == args.mode:
		bqsr_tensors_from_tensor_map(args, include_annotations=True)	
	elif 'write_tranches' == args.mode:
		write_tranches(args)

	# Inspections			
	elif 'inspect_tensors' == args.mode:
		inspect_read_tensors(args)
	elif 'inspect_dataset' == args.mode:
		inspect_dataset(args)
	elif 'inspect_gnomad' == args.mode:
		inspect_gnomad_low_ac(args)
	elif 'combine_vcfs' == args.mode:
		combine_vcfs(args)	
	
	# Ooops
	else:
		raise ValueError('Unknown recipe mode:', args.mode)


def tensors_from_tensor_map(args, include_annotations=True, pileup=False, reference_map='reference'):
	'''Create tensors structured as tensor map of reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	'''	
	print('Writing tensors with:', args.tensor_map, 'channel map.')
	stats = Counter()
	debug = False

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))

	tensor_channel_map = defines.get_tensor_channel_map_from_args(args)

	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[ ref_start : ref_end ]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue

			if include_annotations:
				if all(map(lambda x: x not in variant.INFO and x not in variant.FORMAT and x != "QUAL", args.annotations)):
					stats['Missing ALL annotations'] += 1
					continue # Require at least 1 annotation...
				annotation_data = get_annotation_data(args, variant, stats)
					
			good_reads, insert_dict = get_good_reads(args, samfile, variant)
			if len(good_reads) >= args.read_limit:
				stats['More reads than read_limit'] += 1
			if len(good_reads) == 0:
				stats['No reads aligned'] += 1
				continue

			reference_seq = record.seq
			if reference_map is not None:
				reference_tensor = np.zeros( (args.window_size, len(defines.inputs)) )
				for i,b in enumerate(reference_seq):
					b = b.upper()
					if b in defines.inputs:
						reference_tensor[i, defines.inputs[b]] = 1.0
					elif b in defines.ambiguity_codes:
						reference_tensor[i] = defines.ambiguity_codes[b]
					else:
						raise ValueError('Error! Unknown code:', b)

			for i in sorted(insert_dict.keys(), key=int, reverse=True):
				if i < 0:
					reference_seq = defines.indel_char*insert_dict[i] + reference_seq
				else:
					reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

			read_tensor = good_reads_to_tensor(args, good_reads, ref_start, insert_dict)
			reference_sequence_into_tensor(args, reference_seq, read_tensor)

			tensor_path = get_path_to_train_valid_or_test(args.data_dir)
			tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) + '_allele_' + str(allele_idx) + '-' + cur_label_key 
			tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
			stats[cur_label_key] += 1

			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				hf.create_dataset(args.tensor_map, data=read_tensor, compression='gzip')
				if include_annotations:
					hf.create_dataset(args.annotation_set, data=annotation_data, compression='gzip')
				if reference_map is not None:
					hf.create_dataset(reference_map, data=reference_tensor, compression='gzip')
				if pileup:
					pileup_tensor = read_tensor_to_pileup(args, read_tensor)
					hf.create_dataset('pileup_tensor', data=pileup_tensor, compression='gzip')
			
			if debug:
				print('Reads:', len(good_reads), 'count:', stats['count'],  'Variant:', variant.CHROM, variant.POS, variant.REF, variant.ALT, '\n')
				print(str(reference_seq)+'<- reference sequence\n\n')

			stats['count'] += 1
			if stats['count']%400 == 0:
				print('Wrote', stats['count'], 'tensors out of', args.samples, ' last variant:', str(variant))
			if stats['count'] >= args.samples:
				break

	for s in stats.keys():
		print(s, 'has:', stats[s])
	if variant:
		print('Done generating tensors. Last variant:', str(variant), 'from vcf:', args.negative_vcf)


def paired_read_tensors_from_map(args, include_annotations=True):
	'''Create tensors structured as tensor map of paired reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	'''	
	print('Writing paired read tensors from', args.tensor_map, 'channel map.')
	stats = Counter()
	debug = False

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	
	tensor_channel_map = defines.get_tensor_channel_map_from_args(args)

	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[ ref_start : ref_end ]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue

			if include_annotations:
				if all(map(lambda x: x not in variant.INFO and x not in variant.FORMAT and x != "QUAL", args.annotations)):
					stats['Missing ALL annotations'] += 1
					continue # Require at least 1 annotation...
				annotation_data = get_annotation_data(args, variant, stats)

			good_reads, insert_dict = get_good_reads_in_window(args, samfile, variant.POS-1, variant.POS+1, variant)
			reference_seq = record.seq
			for i in sorted(insert_dict.keys(), key=int, reverse=True):
				reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

			read_tensor = good_reads_and_mates_to_tensor(args, variant, good_reads, ref_start, insert_dict, samfile)
			reference_sequence_into_tensor(args, reference_seq, read_tensor)
			tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
			tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) + '_allele_' + str(allele_idx) + '-' + cur_label_key 
			tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
			stats[cur_label_key] += 1

			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				hf.create_dataset(args.tensor_map, data=read_tensor, compression='gzip')
				if include_annotations:
					hf.create_dataset(args.annotation_set, data=annotation_data, compression='gzip')
			
			if debug:
				print('Reads:', len(good_reads), 'count:', stats['count'],  'Variant:', variant.CHROM, variant.POS, variant.REF, variant.ALT, '\n')
				print(str(reference_seq)+'<- reference sequence\n\n')

			stats['count'] += 1
			if stats['count']%400 == 0:
				print('Wrote', stats['count'], 'tensors out of', args.samples, ' last variant:', str(variant))
			if stats['count'] >= args.samples:
				break

	for s in stats.keys():
		print(s, 'has:', stats[s])
	if stats['count'] > 0:
		print('Done generating tensors. Last variant:', str(variant), 'from vcf:', args.negative_vcf, 'count is:', stats['count'])


def calling_tensors_from_tensor_map(args, pileup=False):
	'''Create tensors structured as tensor map of reads Labels are 1d segmentation (genotyping) of the reference.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
		pileup: Boolean whether the read tensors should be summed over each dimension to make them 1d
	'''	
	print('Writing tensors for Variant Calling from tensor channel map:', args.tensor_map)
	stats = Counter()
	debug = False

	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))

	tensor_channel_map = defines.get_tensor_channel_map_from_args(args)
	
	cur_pos = args.start_pos
	contig = record_dict[args.chrom]		
	label_vector = np.zeros((args.window_size,))

	while cur_pos < args.end_pos - args.window_size:
		
		skip_this = False
		label_vector[:] = defines.calling_labels['REFERENCE']
		
		record = contig[cur_pos : cur_pos+args.window_size]
		
		cur_labels = []
		known_inserts = {}
		variants = vcf_ram.fetch(args.chrom, cur_pos, cur_pos+args.window_size)
		for variant in variants:
			
			if len(variant.get_hets()) == 1:
				cur_label_key = 'HET_'
			elif len(variant.get_hom_alts()):
				cur_label_key = 'HOM_'
			else:
				stats['variant not het or hom'] += 1
				skip_this = True
				break

			if variant.is_snp:
				cur_label_key += 'SNP'
				alt_length = 1
				v_start = (variant.POS-cur_pos) - 1
			elif variant.is_deletion:
				cur_label_key += 'DELETION'
				alt_length = len(variant.REF) - min(map(len, variant.ALT))
				v_start = (variant.POS-cur_pos)
			elif variant.is_indel: # indel & !is_deletion -> insertion
				cur_label_key += 'INSERTION'
				alt_length = max(map(len, variant.ALT)) - len(variant.REF)
				v_start = (variant.POS-cur_pos)
				known_inserts[v_start] = alt_length
			else:
				stats['Not SNP or INDEL'] += 1
				skip_this = True
				break
			
			cur_labels.append(cur_label_key)
			v_end = min(args.window_size, v_start+alt_length)
			label_vector[v_start:v_end] = defines.calling_labels[cur_label_key]

		if len(cur_labels) == 0 and args.downsample_reference < 1.0:
			dice = np.random.rand()
			if dice > args.downsample_reference:
				stats['Downsampled Reference'] += 1
				skip_this = True

		if args.downsample_snps < 1.0 and any(map(lambda x: 'SNP' in x, cur_labels)):
			dice = np.random.rand()
			if dice > args.downsample_snps:
				stats['Downsampled SNPs'] += 1
				skip_this = True

		if args.downsample_homozygous < 1.0 and any(map(lambda x: x in ['HOM_INSERTION', 'HOM_DELETION'], cur_labels)):
			dice = np.random.rand()
			if dice > args.downsample_homozygous:
				stats['Downsampled Homozygous'] += 1
				skip_this = True

		if len(cur_labels) == 0:
			stats['Reference only tensor'] += 1
			good_reads, insert_dict = get_good_reads_in_window(args, samfile, cur_pos, cur_pos+args.window_size)
		else:
			good_reads, insert_dict = get_good_reads_in_window(args, samfile, cur_pos, cur_pos+args.window_size, variant)

		if len(good_reads) == 0:
			stats['No reads aligned'] += 1
			skip_this = True

		if skip_this:
			cur_pos += args.window_size
			continue	

		for l in cur_labels:
			stats[l] += 1	

		reference_seq = record.seq
		for i in sorted(insert_dict.keys(), key=int, reverse=True):
			if i < 0:
				reference_seq = defines.indel_char*insert_dict[i] + reference_seq
			else:
				reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]
			if i not in known_inserts and i < args.window_size: # This does not properly handle complex multi-insertion sites
				known_insert_offset = sum(v for k,v in known_inserts.items() if k < i)
				label_vector = np.insert(label_vector, known_insert_offset+i, np.zeros((insert_dict[i],)))[:args.window_size]

		read_tensor = good_reads_to_tensor(args, good_reads, cur_pos, insert_dict)
		reference_sequence_into_tensor(args, reference_seq, read_tensor)

		tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
		tensor_prefix = 'calling_tensor_' + plain_name(args.bam_file) +'_'+ plain_name(args.train_vcf) 
		tensor_path += tensor_prefix + '-' + args.chrom + '_' + str(cur_pos) + '_' +str(cur_pos+args.window_size) + '.hd5'

		if not os.path.exists(os.path.dirname(tensor_path)):
			os.makedirs(os.path.dirname(tensor_path))
		with h5py.File(tensor_path, 'w') as hf:
			if pileup:
				pileup_tensor = read_tensor_to_pileup(args, read_tensor)
				hf.create_dataset('pileup_tensor', data=pileup_tensor, compression='gzip')
			else:
				hf.create_dataset(args.tensor_map, data=read_tensor, compression='gzip')
			hf.create_dataset('site_labels', data=label_vector, compression='gzip')
		
		cur_pos += args.window_size
		stats['count'] += 1
		if stats['count']%400 == 0:
			print('Wrote', stats['count'], 'calling tensors out of', args.samples, ' last variant:', str(variant))
			for s in stats.keys():
				print(s, 'has:', stats[s])
		if stats['count'] >= args.samples:
			break

		if debug:
			if got_variation and variant.is_indel:
				print('Got a', cur_label_key,' at:', tensor_path, ' site labels:', label_vector)
				print('Ref:', variant.REF, 'alt:', variant.ALT)
				print(zip(str(reference_seq)[:args.window_size], label_vector))
				print(str(reference_seq)+'<- reference sequence\n\n')
	
	if stats['count'] > 0:
		print('Done generating tensors from vcf:', args.train_vcf, 'count is:', stats['count'])

	for s in stats.keys():
		print(s, 'has:', stats[s])



def tensors_from_tensor_map_gnomad_annos(args, include_reads=True, include_annotations=True, include_reference=False):
	'''Create tensors structured as tensor map of reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	'''	
	print('Writing tensors from tensor channel map, with gnomAD annotations.')
	stats = Counter()
	debug = False

	gnomads = gnomads_to_dict()

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))

	tensor_channel_map = defines.get_tensor_channel_map() 

	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		idx_offset, ref_start, ref_end = get_variant_window(args, variant)

		contig = record_dict[variant.CHROM]	
		record = contig[variant.POS-idx_offset: variant.POS+idx_offset]

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if variant_in_vcf(variant, vcf_ram) and in_bed:
			class_prefix = ''
		elif in_bed:
			class_prefix = 'NOT_'
		else:
			stats['Variant outside confident bed file'] += 1
			continue

		if variant.is_snp:
			cur_label_key = class_prefix + 'SNP'
		elif variant.is_indel:
			cur_label_key = class_prefix + 'INDEL'
		else:
			stats['Not SNP or INDEL'] += 1
			continue

		if args.downsample_snps < 1.0 and cur_label_key == 'SNP':
			dice = np.random.rand() 
			if dice > args.downsample_snps:
				stats['Downsampled SNPs'] += 1
				continue

		if include_annotations:
			annotation_variant = variant_in_vcf(variant, gnomads[variant.CHROM])
			if not annotation_variant:
				stats['Variants not in annotation_vcf'] += 1
				continue

			if all(map(lambda x: x not in annotation_variant.INFO and x != "QUAL", args.annotations)):
				stats['Missing ALL annotations'] += 1
				continue # Require at least 1 annotation...

			annotation_data = np.zeros(( len(args.annotations), ))
			qual_and_dp_normalizer = 1000000.0
			for i,a in enumerate(args.annotations):
				if a == "QUAL":
					annotation_data[i] = float(annotation_variant.QUAL)
				else:
					annotation_data[i] = annotation_variant.INFO[a]
				

				if a == "DP" or a == "QUAL":
					 annotation_data[i] /= qual_and_dp_normalizer


		if include_reference:
			dna_data = np.zeros( (args.window_size, len(defines.inputs)) )
			for i,b in enumerate(record.seq):
				if b in defines.inputs:
					dna_data[i, defines.inputs[b]] = 1.0
				elif b in defines.ambiguity_codes:
					dna_data[i] = defines.ambiguity_codes[b]
				else:
					raise ValueError('Error! Unknown code:', b)
					
		
		if include_reads:
			good_reads, insert_dict = get_good_reads(args, samfile, variant)
			reference_seq = record.seq
			for i in sorted(insert_dict.keys(), key=int, reverse=True):
				reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

			sequences, qualities, mapping_qualities, flags = good_reads_to_arrays(args, good_reads, ref_start, insert_dict)

			if len(sequences) > 0:
				ref_read_idx = defines.get_reference_and_read_channels(args)
				if args.channels_last:
					read_tensor = np.zeros( (args.read_limit, args.window_size, len(tensor_channel_map)) )
					read_tensor[:,:,:ref_read_idx] = reads_to_tensor(args, sequences, qualities, reference_seq)
				else:
					read_tensor = np.zeros( (len(tensor_channel_map), args.read_limit, args.window_size) )
					read_tensor[:ref_read_idx,:,:] = reads_to_tensor(args, sequences, qualities, reference_seq)
				add_flags_to_read_tensor(args, read_tensor, tensor_channel_map, flags)
				add_mq_to_read_tensor(args, read_tensor, tensor_channel_map, mapping_qualities)

		tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
		tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) + '-' + cur_label_key 
		tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
		stats[cur_label_key] += 1

		if not os.path.exists(os.path.dirname(tensor_path)):
			os.makedirs(os.path.dirname(tensor_path))
		with h5py.File(tensor_path, 'w') as hf:
			if include_reads:
				hf.create_dataset(args.tensor_map, data=read_tensor)
			if include_annotations:
				hf.create_dataset(args.annotation_set, data=annotation_data)
			if include_reference:
				hf.create_dataset('reference', data=dna_data)

		stats['count'] += 1
		if stats['count']%400 == 0:
			print('Wrote', stats['count'], 'tensors out of', args.samples, ' last variant:', str(variant))
		if stats['count'] >= args.samples:
			break

	for s in stats.keys():
		print(s, 'has:', stats[s])
	print('Done generating gnomAD annotated tensors. Last variant:', str(variant), 'from vcf:', args.negative_vcf, 'count is:', stats['count'])



def tensors_from_tensor_map_gnomad_annos_per_allele(args, include_reads=True, include_annotations=True, include_reference=False):
	'''Create allele specific tensors structured as tensor map of reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	'''	
	print('Writing allele specific tensors from tensor channel map with gnomAD annotations.')
	stats = Counter()
	debug = False

	gnomads = gnomads_to_dict()

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))

	tensor_channel_map = defines.get_tensor_channel_map() 

	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_index, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)

			contig = record_dict[variant.CHROM]	
			record = contig[variant.POS-idx_offset: variant.POS+idx_offset]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue

			if include_annotations:
				annotation_variant = variant_in_vcf(variant, gnomads[variant.CHROM])
				if not annotation_variant:
					stats['Variants not in annotation_vcf'] += 1
					continue

				if all(map(lambda x: x not in annotation_variant.INFO and x != "QUAL", args.annotations)):
					stats['Missing ALL annotations'] += 1
					continue # Require at least 1 annotation...

				annotation_data = np.zeros(( len(args.annotations), ))
				qual_and_dp_normalizer = 1000000.0
				as_normalizer = 100.0
				for i,a in enumerate(args.annotations):
					if a in ['DP_MEDIAN', 'DREF_MEDIAN', 'GQ_MEDIAN', 'AB_MEDIAN']:
						if not math.isnan(annotation_variant.INFO[a][allele_index]):
							annotation_data[i] = annotation_variant.INFO[a][allele_index]
					elif a == "QUAL":
						annotation_data[i] = float(annotation_variant.QUAL)
					else:
						annotation_data[i] = annotation_variant.INFO[a]
					
					if a == "DP" or a == "QUAL":
						 annotation_data[i] /= qual_and_dp_normalizer
					if a == "DP_MEDIAN" or a == "GQ_MEDIAN":
						 annotation_data[i] /= as_normalizer

			if include_reference:
				dna_data = np.zeros( (args.window_size, len(defines.inputs)) )
				for i,b in enumerate(record.seq):
					if b in defines.inputs:
						dna_data[i, defines.inputs[b]] = 1.0
					elif b in defines.ambiguity_codes:
						dna_data[i] = defines.ambiguity_codes[b]
					else:
						raise ValueError('Error! Unknown code:', b)
						
			
			if include_reads:
				good_reads, insert_dict = get_good_reads(args, samfile, variant)
				reference_seq = record.seq
				for i in sorted(insert_dict.keys(), key=int, reverse=True):
					reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

				sequences, qualities, mapping_qualities, flags = good_reads_to_arrays(args, good_reads, ref_start, insert_dict)

				if len(sequences) > 0:
					ref_read_idx = defines.get_reference_and_read_channels(args)
					if args.channels_last:
						read_tensor = np.zeros( (args.read_limit, args.window_size, len(tensor_channel_map)) )
						read_tensor[:,:,:ref_read_idx] = reads_to_tensor(args, sequences, qualities, reference_seq)
					else:
						read_tensor = np.zeros( (len(tensor_channel_map), args.read_limit, args.window_size) )
						read_tensor[:ref_read_idx,:,:] = reads_to_tensor(args, sequences, qualities, reference_seq)
					add_flags_to_read_tensor(args, read_tensor, tensor_channel_map, flags)
					add_mq_to_read_tensor(args, read_tensor, tensor_channel_map, mapping_qualities)

			tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
			tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) +'_allele_'+ str(allele_index) + '-' + cur_label_key 
			tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
			stats[cur_label_key] += 1
			stats['Allele index '+str(allele_index)] += 1

			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				if include_reads:
					hf.create_dataset(args.tensor_map, data=read_tensor)
				if include_annotations:
					hf.create_dataset(args.annotation_set, data=annotation_data)
				if include_reference:
					hf.create_dataset('reference', data=dna_data)

			stats['count'] += 1
			if stats['count']%400 == 0:
				print('Wrote', stats['count'], 'tensors out of', args.samples, ' last variant:', str(variant))
			if stats['count'] >= args.samples:
				break

	for s in stats.keys():
		print(s, 'has:', stats[s])
	print('Done generating gnomAD annotated tensors. Last variant:', str(variant), 'from vcf:', args.negative_vcf, 'count is:', stats['count'])





def tensors_from_tensor_map_2channel(args, include_annotations=True):
	'''Create tensors structured as tensor map of reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	'''	
	print('Writing 2 bit DNA tensors from tensor channel map.')
	stats = Counter()
	debug = False

	args.input_symbols = defines.dna_2bit

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	
	tensor_channel_map = defines.get_tensor_channel_map_from_args(args)

	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[ ref_start : ref_end ]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue

			if include_annotations:
				if all(map(lambda x: x not in variant.INFO and x not in variant.FORMAT and x != "QUAL", args.annotations)):
					stats['Missing ALL annotations'] += 1
					continue # Require at least 1 annotation...
				annotation_data = get_annotation_data(args, variant, stats)

			good_reads, insert_dict = get_good_reads(args, samfile, variant)
			reference_seq = record.seq
			for i in sorted(insert_dict.keys(), key=int, reverse=True):
				reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

			sequences, qualities, mapping_qualities, flags = good_reads_to_arrays(args, good_reads, ref_start, insert_dict)

			if len(sequences) > 0:
				if args.channels_last:
					read_tensor = np.zeros( (args.read_limit, args.window_size, len(tensor_channel_map)) )
					read_tensor[:,:,:6] = reads_to_2bit_tensor(args, sequences, qualities, reference_seq)
				else:
					read_tensor = np.zeros( (len(tensor_channel_map), args.read_limit, args.window_size) )
					read_tensor[:6,:,:] = reads_to_2bit_tensor(args, sequences, qualities, reference_seq)
				add_flags_to_read_tensor(args, read_tensor, tensor_channel_map, flags)
				add_mq_to_read_tensor(args, read_tensor, tensor_channel_map, mapping_qualities)
				tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
				tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) +'_allele_'+ str(allele_index) +'-'+ cur_label_key 
				tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
				stats[cur_label_key] += 1

				if not os.path.exists(os.path.dirname(tensor_path)):
					os.makedirs(os.path.dirname(tensor_path))
				with h5py.File(tensor_path, 'w') as hf:
					hf.create_dataset(args.tensor_map, data=read_tensor)
					if include_annotations:
						hf.create_dataset(args.annotation_set, data=annotation_data)
				
				if debug:
					print('Reads:', len(good_reads), 'count:', stats['count'],  'Variant:', variant.CHROM, variant.POS, variant.REF, variant.ALT, '\n')
					for i,s in enumerate(sequences):
						print(s+'  cigar:'+good_reads[i].cigarstring)
					print(str(reference_seq)+'<- reference sequence\n\n')

				stats['count'] += 1
				if stats['count']%400 == 0:
					print('Wrote', stats['count'], 'tensors out of', args.samples, ' last variant:', str(variant))
				if stats['count'] >= args.samples:
					break

	for s in stats.keys():
		print(s, 'has:', stats[s])
	print('Done generating tensors. Last variant:', str(variant), 'from vcf:', args.negative_vcf, 'count is:', stats['count'])


def bqsr_tensors_from_tensor_map(args, include_annotations=False):
	"""Create tensors structured as tensor map of read and reference organized by labels in the data directory.

	Defines true bases as those not in args.db_snp, where read and reference agree.
	False bases are sites thos not in args.db_snp where the read and the reference do NOT agree. 

	Arguments
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.train_vcf: VCF file with sites of known variation, from NIST, DBSNP etc.
		args.bed_file: Intervals of interest
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	"""		
	print('Writing BQSR tensors from tensor channel map.')
	stats = Counter()

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	contig = record_dict[args.chrom]

	tensor_channel_map = defines.bqsr_tensor_channel_map() 

	if args.chrom:
		reads  = samfile.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		reads = samfile

	for read in reads:
		if read.is_reverse:
			continue
		read_group = read.get_tag('RG')	
		if 'artificial' in read_group.lower():
			continue
		if not read.is_proper_pair or not read.is_paired:
			continue
		if read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_qcfail or read.is_unmapped:
			continue

		for ref_pos, read_idx in zip(read.get_reference_positions(), range(len(read.query_sequence))):	
			if contig[ref_pos] != read.query_sequence[read_idx]:
				variants = vcf_ram.fetch(args.chrom, ref_pos-1, ref_pos+1)
				in_vcf = False
				for v in variants:
					in_vcf |= any([a1 == read.query_sequence[read_idx] for a1 in v.ALT]) and ref_pos == v.POS
				if in_vcf:
					stats['Already in known variation VCF'] += 1
					continue
				cur_label_key = 'BAD_BASE'			

			else:
				if args.downsample_snps < 1.0:
					dice = np.random.rand()
					if dice > args.downsample_snps:
						continue
				cur_label_key = 'GOOD_BASE'

			stats[cur_label_key] += 1

			ref_string = contig.seq[ref_pos-args.window_size:ref_pos]
			read_string = read.query_sequence[max(0,read_idx-args.window_size) : read_idx]
			read_qualities = read.query_alignment_qualities[max(0,read_idx-args.window_size) : read_idx].tolist()
			if read_idx-args.window_size < 0:
				read_string = defines.skip_char * (args.window_size-read_idx) + read_string
				read_qualities = [0] * (args.window_size-read_idx) + read_qualities

			# print (cur_label_key,contig[ref_pos], read.query_sequence[read_idx] )
			# print ('read Qualzz:%s'%str(read_qualities))
			# print ('read string:%s'%read_string)
			# print ('refr string:%s'%ref_string)
			
			read_tensor = np.zeros((args.window_size, len(tensor_channel_map)))
			read_tensor[:, 0:len(args.input_symbols)] = base_string_to_tensor(args, read_string, read_qualities)
			read_tensor[:, len(args.input_symbols):(2*len(args.input_symbols))] = base_string_to_tensor(args, ref_string)
			
			#print (read_tensor)
			if include_annotations:
				max_mq = 60.0
				max_read_pos = 151.0
				annotation_data = np.zeros(( len(defines.bqsr_annotations), ))
				for i,a in enumerate(defines.bqsr_annotations):
					if a == "reverse":
						annotation_data[i] = float(read.is_reverse)
					elif a == 'first_in_pair':
						annotation_data[i] = float(read.is_read1)
					elif a == 'mapping_quality':
						annotation_data[i] = float(read.mapping_quality) / max_mq
					elif a == 'read_position':
						annotation_data[i] = float(read_idx)/ max_read_pos	

			tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
			tensor_prefix = plain_name(args.bam_file) +'_'+ plain_name(args.train_vcf) + '-' + cur_label_key 
			tensor_path += cur_label_key + '/' + tensor_prefix + '-' + args.chrom + '_' + str(ref_pos) + '.hd5'
			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))
			with h5py.File(tensor_path, 'w') as hf:
				hf.create_dataset(args.tensor_map, data=read_tensor)
				if include_annotations:
					hf.create_dataset(args.annotation_set, data=annotation_data)
		
			stats['count'] +=1
			if stats['count']%400 == 0:
				print('Wrote', stats['count'], 'tensors out of', args.samples)
			if stats['count'] >= args.samples:
				break
		if stats['count'] >= args.samples:
			break
	
	for k in stats.keys():
		print('%s has %d' %(k, stats[k]))

	print('Done generating BQSR tensors. Wrote them to:', args.data_dir ,' Known variation vcf:', args.train_vcf)


def write_dna_and_annotations(args, include_dna=True, include_annotations=True):
	debug = False
	stats = Counter()

	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	bed_dict = bed_file_to_dict(args.bed_file)

	idx_offset = (args.window_size//2)

	if args.chrom:
		variants  = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[variant.POS-idx_offset: variant.POS+idx_offset]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue

			if include_annotations:
				if all(map(lambda x: x not in variant.INFO and x not in variant.FORMAT and x != "QUAL", args.annotations)):
					stats['Missing ALL annotations'] += 1
					continue # Require at least 1 annotation...
				annotation_data = get_annotation_data(args, variant, stats)

			if include_dna:
				dna_data = np.zeros( (args.window_size, len(defines.inputs)) )
				for i,b in enumerate(record.seq):
					if b in defines.inputs:
						dna_data[i, defines.inputs[b]] = 1.0
					elif b in defines.ambiguity_codes:
						dna_data[i] = defines.ambiguity_codes[b]
					else:
						raise ValueError('Error! Unknown code:', b)
						

			tensor_path = get_path_to_train_valid_or_test(args.data_dir)
			tensor_path += cur_label_key +'/'+ plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) 
			tensor_path += '_allele_' + str(allele_idx) +'-'+ variant.CHROM +'_'+ str(variant.POS) + '.hd5'
			if not os.path.exists(os.path.dirname(tensor_path)):
				os.makedirs(os.path.dirname(tensor_path))

			if debug:
				print('Try to write tensor to:', tensor_path)
				print('Sequence was:', record.seq)
				print('DNA tensor is:', dna_data)
				print('Annotation tensor is:', annotation_data)

			with h5py.File(tensor_path, 'w') as hf:
				if include_annotations:
					hf.create_dataset(args.annotation_set, data=annotation_data, compression='gzip')
				if include_dna:
					hf.create_dataset(args.tensor_map, data=dna_data, compression='gzip')
			
			stats[cur_label_key] += 1
			stats['count'] += 1
			if stats['count']%500==0:
				print('Wrote', stats['count'], 'out of:', args.samples, 'Last variant:', variant)
			if args.samples == stats['count']:
				break

	print('Done Writing. DNA:', include_dna,' and Annotations:',include_annotations, ' Wanted: ', args.samples)
	for k in stats.keys():
		print(k, ' has:', stats[k])
	

def write_dna_multisource_annotations(args, include_dna=True, include_annotations=True):
	debug = False
	stats = Counter()

	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	bed_dict = bed_file_to_dict(args.bed_file)

	idx_offset = (args.window_size//2)

	channel_map = defines.get_tensor_channel_map_from_args(args)

	# Get bed file dicts
	bed_dicts = {}
	for b in channel_map.keys():
		if os.path.exists(b) and os.path.splitext(b)[1].lower() == '.bed':
			bed_dicts[b] = bed_file_to_dict(b)

	# Do we need to fetch a particular region of the genome?	
	if args.chrom:
		variants  = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[variant.POS-idx_offset: variant.POS+idx_offset+(args.window_size%2)]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue		

		if include_annotations:
			if all(map(lambda x: x not in annotation_variant.INFO and x != "QUAL", args.annotations)):
				stats['Missing ALL annotations'] += 1
				continue # Require at least 1 annotation...
			annotation_data = get_annotation_data(args, variant, stats)

		if include_dna:
			dna_data = np.zeros( (args.window_size, len(channel_map)) )
			for i,b in enumerate(record.seq):
				# Get the reference DNA for the first 4 channels
				if b in channel_map:
					dna_data[i, channel_map[b]] = 1.0
				elif b in defines.ambiguity_codes:
					dna_data[i] = defines.ambiguity_codes[b]
				else:
					raise ValueError('Error! Unknown code:', b)
					

				# Add data to remaining channels from bed files	
				ref_i = variant.POS-idx_offset + i
				for k in channel_map.keys():
					if k in defines.inputs:
						continue
					if in_bed_file(bed_dicts[k], variant.CHROM, ref_i):
						dna_data[i, channel_map[k]] = 1.0 # TODO: update this to read a value from the bed file

		tensor_path = get_path_to_train_valid_or_test(args.data_dir)	
		tensor_prefix = plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) +'_allele_'+ str(allele_idx) +'_'+ cur_label_key 
		tensor_path += cur_label_key + '/' + tensor_prefix + '-' + variant.CHROM + '_' + str(variant.POS) + '.hd5'
		if not os.path.exists(os.path.dirname(tensor_path)):
			os.makedirs(os.path.dirname(tensor_path))

		if debug:
			print('Try to write tensor to:', tensor_path)
			print('Sequence was:', record.seq)
			if include_dna:
				print('DNA tensor is:\n', dna_data)
				print('DNA Column sums are:', np.sum(dna_data, axis=0))
			if include_annotations:
				print('Annotation tensor is:', annotation_data)

		with h5py.File(tensor_path, 'w') as hf:
			if include_annotations:
				hf.create_dataset(args.annotation_set, data=annotation_data)
			if include_dna:
				hf.create_dataset(args.tensor_map, data=dna_data)
		
		stats[cur_label_key] += 1
		stats['count'] += 1
		if stats['count']%500==0:
			print('Wrote', stats['count'], 'out of:', args.samples, 'Last variant:', variant)
		if args.samples == stats['count']:
			break

	print('Done writing reference tensors')
	for k in stats.keys():
		print('Label:', k, 'Got', stats[k], ' examples.')


def nist_samples_to_png(args):
	"""Create PNGs from reads organized by labels in the data directory.

	Defines true variants as those in the args.train_vcf, defines false variants as 
	those called in args.negative_vcf and in the args.bed_file high confidence intervals, 
	but not in args.train_vcf.

	Arguments:
		args.data_dir: directory where tensors will live. Created here and filled with
			subdirectories of test, valid and train, each containing
			subdirectories for each label with tensors stored as hd5 files.
		args.bam_file: BAM or BAMout file where the aligned reads are stored
		args.negative_vcf: VCF file with annotation values from Haplotype caller or VQSR
		args.train_vcf: VCF file with true variant (from NIST or Platinum genomes, etc.)
		args.bed_file: High confidence intervals for the calls in args.train_vcf
		args.window_size: Size of sequence window around variant (width of the tensor)
		args.read_limit: Maximum number of reads to include (height of  the tensor)
		args.chrom: Only write tensors from this chromosome (optional, used for parallelization)
		args.start_pos: Only write tensors after this position (optional, used for parallelization)
		args.end_pos: Only write tensors before this position (optional, used for parallelization)
	"""
	import cv2 # lazy import because lot's of machines don't have opencv
	print('Writing pngs from NIST samples.')
	debug = False
	stats = Counter()

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.train_vcf, 'r'))
	bed_dict = bed_file_to_dict(args.bed_file)
	record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))

	if args.chrom:
		variants  = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		for allele_idx, allele in enumerate(variant.ALT):
			idx_offset, ref_start, ref_end = get_variant_window(args, variant)
			contig = record_dict[variant.CHROM]	
			record = contig[ ref_start : ref_end ]

			cur_label_key = get_true_label(allele, variant, bed_dict, vcf_ram, stats)
			if not cur_label_key or downsample(args, cur_label_key, stats):
				continue		
		
			reads = samfile.count(variant.CHROM, ref_start, ref_end)
			if reads <= 0:
				continue

			stats[cur_label_key] += 1
			good_reads, insert_dict = get_good_reads(args, samfile, variant, sort_by='reference_start')
			reference_seq = record.seq
			for i in sorted(insert_dict.keys(), key=int, reverse=True):
				if i < 0:
					reference_seq = defines.indel_char*insert_dict[i] + reference_seq
				else:
					reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]
			sequences, qualities, mapping_qualities, flags = good_reads_to_arrays(args, good_reads, ref_start, insert_dict)

			if len(sequences) > 0:
				image = seq_block_to_image(args, reference_seq, sequences, flags, qualities, mapping_qualities)
				image_path = get_path_to_train_valid_or_test(args.data_dir)
				image_path += cur_label_key + '/' + plain_name(args.negative_vcf) +'_'+ plain_name(args.train_vcf) +'_allele_'+ str(allele_idx)
				image_path += '_' + cur_label_key + '-' + variant.CHROM + '_' + str(variant.POS) + '.png'
				if not os.path.exists(os.path.dirname(image_path)):
					os.makedirs(os.path.dirname(image_path))
				cv2.imwrite(image_path, image)
				if debug:
					print('Reads:', len(good_reads), 'count:', count,  'Variant:', variant.CHROM, variant.POS, variant.REF, variant.ALT, '\n')
					for s in sequences:
						print(s)
					cv2.imshow('img', image)
					cv2.waitKey(2000)

				stats['count'] +=1
				if stats['count']%400 == 0:
					print('Wrote', stats['count'], 'tensors out of', args.samples)
				if stats['count'] >= args.samples:
					break
	
	for k in stats.keys():
		print('Label:', k, 'Got', stats[k], ' examples.')
	print('Done generating images. Last variant:', str(variant), 'from vcf:', args.negative_vcf, 'count is:', stats['count'])


def get_variant_window(args, variant):
	index_offset = (args.window_size//2)
	reference_start = variant.POS-index_offset
	reference_end = variant.POS+index_offset + (args.window_size%2)

	return index_offset, reference_start, reference_end


def get_annotation_data(args, annotation_variant, stats):
	'''Return an array annotation data about the variant.

	Arguments:
		args.annotations: List of variant annotations to use
		annotation_variant: the variant with annotation
		stats: Counter of run statistics

	Returns:
		annotation_data: numpy array of annotation values
	'''
	annotation_data = np.zeros(( len(args.annotations), ))
	try:
		for i,a in enumerate(args.annotations):
			if a == 'QUAL':
				annotation_data[i] = annotation_variant.QUAL
			elif a == 'AF':
				annotation_data[i] = annotation_variant.INFO[a][0]
			elif a in annotation_variant.INFO and not math.isnan(annotation_variant.INFO[a]):
				annotation_data[i] = annotation_variant.INFO[a]
			elif a == 'MBQ':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.MBQ
			elif a == 'MPOS':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.MPOS 
			elif a == 'MMQ':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.MMQ 
			elif a == 'MFRL_0':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.MFRL[0] 
			elif a == 'MFRL_1':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.MFRL[1] 
			elif a == 'AD_0':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.AD[0] 
			elif a == 'AD_1':
				call = annotation_variant.genotype(args.sample_name)
				annotation_data[i] = call.data.AD[1]
			else:
				stats['Could not handle annotation:'+a] += 1

	except ValueError as e:
		print(str(e) + '\nERROR! at variant:', annotation_variant, '\n format stuff:', annotation_variant.genotype(args.sample_name))
	
	return annotation_data


def get_true_label(allele, variant, bed_dict, truth_vcf, stats):
	'''Defines the truth status of a variant allele given a truth vcf and confident region.

	Arguments:
		allele: The allele to check
		variant: the variant whose allele we will check
		bed_dict: confident region dict defined by intervals e.g. from bed_file_to_dict()
		truth_vcf: vcf of validated variants
		stats: Counter dict used to keep track of the label distribution, etc.

	Returns:
		None if outside the confident region
		Otherwise a label string:
			SNP if variant is snp and in truth vcf
			INDEL if variant is indel and in truth vcf
			NOT_SNP if variant is snp and not in truth vcf
			NOT_INDEL if variant is indel and not in truth vcf
	'''
	in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
	if allele_in_vcf(allele, variant, truth_vcf) and in_bed:
		class_prefix = ''
	elif in_bed:
		class_prefix = 'NOT_'
	else:
		stats['Variant outside confident bed file'] += 1
		return None

	if variant.is_snp:
		cur_label_key = class_prefix + 'SNP'
	elif variant.is_indel:
		cur_label_key = class_prefix + 'INDEL'
	else:
		stats['Not SNP or INDEL'] += 1
		return None

	return cur_label_key	


def downsample(args, cur_label_key, stats):
	'''Indicates whether or not to downsample a variant.

	Arguments:
		args.skip_positive_class: Skip all positive examples
		args.downsample_snps: fraction of SNPs to keep
		args.downsample_indels: fraction of INDELs to keep
		cur_label_key: truth label from get_true_label()
		stats: Counter dict used to keep track of a run

	Returns:
		Boolean: should we downsample this variant or not.
	'''	
	if args.skip_positive_class and cur_label_key in ['SNP', 'INDEL']:
		return True

	if args.downsample_snps < 1.0 and cur_label_key == 'SNP':
		dice = np.random.rand()
		if dice > args.downsample_snps:
			stats['Downsampled SNPs'] += 1
			return True
	elif args.downsample_indels < 1.0 and cur_label_key == 'INDEL':
		dice = np.random.rand()
		if dice > args.downsample_indels:
			stats['Downsampled INDELs'] += 1
			return True
	if args.downsample_not_snps < 1.0 and cur_label_key == 'NOT_SNP':
		dice = np.random.rand()
		if dice > args.downsample_not_snps:
			stats['Downsampled NOT_SNPs'] += 1
			return True
	elif args.downsample_not_indels < 1.0 and cur_label_key == 'NOT_INDEL':
		dice = np.random.rand()
		if dice > args.downsample_not_indels:
			stats['Downsampled NOT_INDELs'] += 1
			return True


	return False


def make_reference_and_reads_tensor(args, variant, samfile, reference_seq, reference_start, stats):
	good_reads, insert_dict = get_good_reads(args, samfile, variant)
	if len(good_reads) >= args.read_limit:
		stats['More reads than read_limit'] += 1
	if len(good_reads) == 0:
		stats['No reads aligned'] += 1
		return None

	for i in sorted(insert_dict.keys(), key=int, reverse=True):
		if i < 0:
			reference_seq = defines.indel_char*insert_dict[i] + reference_seq
		else:
			reference_seq = reference_seq[:i] + defines.indel_char*insert_dict[i] + reference_seq[i:]

	read_tensor = good_reads_to_tensor(args, good_reads, reference_start, insert_dict)
	reference_sequence_into_tensor(args, reference_seq, read_tensor)
	return read_tensor


def get_good_reads(args, samfile, variant, sort_by='base'):
	'''Return an array of usable reads centered at the variant.
	
	Ignores artificial haplotype read group.
	Relies on pysam's cigartuples structure see: http://pysam.readthedocs.io/en/latest/api.html
	Match, M -> 0
	Insert, I -> 1
	Deletion, D -> 2
	Ref Skip, N -> 3
	Soft Clip, S -> 4

	Arguments:
		args.read_limit: maximum number of reads to return
		samfile: the BAM (or BAMout) file
		variant: the variant around which reads will load

	Returns:
		good_reads: array of usable reads sorted by reference start position
		insert_dict: a dict mapping read indices to max insertions at that point
	'''		
	good_reads = []
	insert_dict = {}

	idx_offset, ref_start, ref_end = get_variant_window(args, variant)

	for read in samfile.fetch(variant.CHROM, variant.POS-1, variant.POS+1):

		if not read or not hasattr(read, 'cigarstring') or read.cigarstring is None:
			continue

		read_group = read.get_tag('RG')	
		if 'artificial' in read_group.lower():
			continue

		index_dif = ref_start - read.reference_start
		if abs(index_dif) >= args.window_size:
			continue

		if 'I' in read.cigarstring:
			cur_idx = 0
			for t in read.cigartuples:
				if t[0] == defines.cigar_code['I']:
					insert_idx = cur_idx - index_dif
					if insert_idx not in insert_dict: 
						insert_dict[insert_idx] = t[1]
					elif insert_dict[insert_idx] < t[1]:
						insert_dict[insert_idx] = t[1]
				
				if t[0] in [defines.cigar_code['M'], defines.cigar_code['I'], defines.cigar_code['S'], defines.cigar_code['D']]:
					cur_idx += t[1]

		good_reads.append(read)
		
	if len(good_reads) > args.read_limit:
		good_reads = np.random.choice(good_reads, size=args.read_limit, replace=False).tolist()

	good_reads.sort(key=lambda x: x.reference_start + x.query_alignment_start)
	if sort_by == 'base':
		good_reads.sort(key=lambda read: get_base_to_sort_by(read, variant))

	return good_reads, insert_dict


def get_base_to_sort_by(read, variant):
	if len(read.query_alignment_sequence) > 0:
		max_idx = len(read.query_alignment_sequence)-1
	else:
		return 'Z'

	if variant.is_snp:
		return read.query_alignment_sequence[clamp((variant.POS-read.reference_start)-1, 0, max_idx)]
	else:
		var_idx = variant.POS-read.reference_start
		cur_idx = 0
		for cur_op, length in read.cigartuples:
			cur_idx += length
			if cur_idx > var_idx:
				if cur_op == defines.cigar_code['M']:
					return read.query_alignment_sequence[clamp(var_idx, 0, max_idx)]
				else:
					return defines.code2cigar[cur_op]
		return 'Y'
		

def clamp(n, minn, maxn):
	return max(min(maxn, n), minn)


def get_good_reads_in_window(args, samfile, start_pos, end_pos, variant=None):
	'''Return an array of usable reads centered at the variant.
	
	Ignores artificial haplotype read group.
	Relies on pysam's cigartuples structure see: http://pysam.readthedocs.io/en/latest/api.html
	Match, M -> 0
	Insert, I -> 1
	Deletion, D -> 2
	Ref Skip, N -> 3
	Soft Clip, S -> 4

	Arguments:
		args.read_limit: maximum number of reads to return
		samfile: the BAM (or BAMout) file
		start_pos: the beginning of the window in reference coordinates
		end_pos: the end of the window in reference coordinates
		variant: (optional) if provided will sort by the base at the variant, 
			for hets this should segregate the chromosomes

	Returns:
		good_reads: array of usable reads sorted by reference start position
		insert_dict: a dict mapping read indices to max insertions at that point
	'''		
	good_reads = []
	insert_dict = {}

	for read in samfile.fetch(args.chrom, start_pos, end_pos):

		if not read or not hasattr(read, 'cigarstring') or read.cigarstring is None:
			continue

		read_group = read.get_tag('RG')	
		if 'artificial' in read_group.lower():
			continue

		index_dif = start_pos - read.reference_start
		if abs(index_dif) >= args.window_size:
			continue

		if 'I' in read.cigarstring:
			cur_idx = 0
			for t in read.cigartuples:
				if t[0] == 1:
					insert_idx = cur_idx - index_dif
					if insert_idx not in insert_dict: 
						insert_dict[insert_idx] = t[1]
					elif insert_dict[insert_idx] < t[1]:
						insert_dict[insert_idx] = t[1]
				if t[0] in [defines.cigar_code['M'], defines.cigar_code['I'], defines.cigar_code['S'], defines.cigar_code['D']]:
					cur_idx += t[1]
		
		good_reads.append(read)

	if len(good_reads) > args.read_limit:
		good_reads = np.random.choice(good_reads, size=args.read_limit, replace=False).tolist()

	good_reads.sort(key=lambda x: x.reference_start + x.query_alignment_start)
	if variant:
		good_reads.sort(key=lambda read: get_base_to_sort_by(read, variant))

	return good_reads, insert_dict


def good_reads_to_arrays(args, good_reads, ref_start, insert_dict):
	'''Transform reads to aligned arrays with insertions and deletions.

	The arrays contain skip and deletion characters so that all indices are comparable across them
	The qualities and sequence arrays map 1-to-1.

	Arguments:
		args.read_limit: maximum number of reads to return
		good_reads: list of reads to make arrays from
		ref_start: the beginning of the window in reference coordinates
		insert_dict: a dict mapping read indices to max insertions at that point.

	Returns:
		sequences: aligned read bases, skip and indel characters
		qualities: the base quality of each base, same size as sequences 
		mapping_qualities: list of mapping quality of each read
		flags: array of read flags for each read.
	'''
	qualities = []
	sequences = []
	mapping_qualities = []
	
	flags = np.zeros((defines.read_flags, args.read_limit))
	no_qual_filler = 0
	
	for i,read in enumerate(good_reads):
		index_dif = ref_start - read.reference_start
		cur_idx = 0
		my_indel_dict = {}
		for t in read.cigartuples:
			my_ref_idx = cur_idx - index_dif

			if t[0] == defines.cigar_code['I']:
				my_indel_dict[my_ref_idx] = insert_dict[my_ref_idx] - t[1]
			elif t[0] == defines.cigar_code['D']:
				my_indel_dict[my_ref_idx] = t[1]
			
			if t[0] in [defines.cigar_code['M'], defines.cigar_code['I'], defines.cigar_code['S'], defines.cigar_code['D']]:
				cur_idx += t[1]

		for k in insert_dict.keys():
			if k not in my_indel_dict:
				my_indel_dict[k] = insert_dict[k]

		rseq = read.query_alignment_sequence[:args.window_size]
		rqual = read.query_alignment_qualities[:args.window_size].tolist()

		if index_dif > 0:
			rseq = rseq[index_dif:] 
			rqual = rqual[index_dif:]
		elif index_dif < 0:
			rseq = defines.skip_char*(-index_dif) + rseq
			rqual = [no_qual_filler]*(-index_dif) + rqual

		for j in sorted(my_indel_dict.keys(), key=int, reverse=True):
			if j < 1:
				rseq = (defines.indel_char*my_indel_dict[j]) + rseq
				rqual = ([no_qual_filler]*my_indel_dict[j]) + rqual
			else:
				rseq = rseq[:j] + (defines.indel_char*my_indel_dict[j]) + rseq[j:]
				rqual = rqual[:j] + ([no_qual_filler]*my_indel_dict[j]) + rqual[j:]


		flags[:, i] = flag_to_array(read.flag)
		mapping_qualities.append(read.mapping_quality)
		sequences.append(rseq)
		qualities.append(rqual)

	return sequences, qualities, mapping_qualities, flags


def good_reads_and_mates_to_tensor(args, variant, good_reads, ref_start, insert_dict, sam_file):
	'''Create a read tensor with read pairs centered at a variant.

	Assumes read pairs have the same name. 
	Only loads reads that might align inside the tensor.

	Arguments:
		args.read_limit: maximum number of reads to return
		variant: the variant at the center of the read_tensor
		good_reads: list of reads to make arrays from
		ref_start: the beginning of the window in reference coordinates
		insert_dict: a dict mapping read indices to max insertions at that point.
		sam_file: the file containing aligned reads

	Returns:
		tensor: read tensor containing read pairs if found in each row.
	'''	
	pairs = defaultdict(list) # Hash maps read names to list of reads.	
	channel_map = defines.get_tensor_channel_map_from_args(args)
	if args.channels_last:
		tensor = np.zeros( (args.read_limit, args.window_size, len(channel_map)) )
	else:
		tensor = np.zeros( (len(channel_map), args.read_limit, args.window_size) )

	idx_offset, ref_start, ref_end = get_variant_window(args, variant)
	for read in sam_file.fetch(variant.CHROM, ref_start, ref_end):
		pairs[read.query_name].append(read)

	for j,good_read in enumerate(good_reads):
		
		if j == args.read_limit:
			break
		if len(pairs[good_read.query_name]) > 2:
			print ('Got too many paired reads:',  len(pairs[good_read.query_name]) )
			for read in pairs[good_read.query_name]:
				print(read.query_name,' read flags:', flag_to_array(read.flag), 'seq:', read.seq)


		for read in pairs[good_read.query_name]:
			if not read.cigartuples: # Could be an unmapped mate
				continue
				
			rseq, rqual = sequence_and_qualities_from_read(args, read, ref_start, insert_dict)
			flag_start = -1
			flag_end = 0
			for i,b in enumerate(rseq):
				
				if i == args.window_size:
					break
				
				if b == defines.skip_char:
					continue
				elif flag_start == -1:
					flag_start = i
				else:
					flag_end = i

				if b in args.input_symbols:
					if b == defines.indel_char:
						if args.channels_last:
							tensor[j, i, args.input_symbols[b]] = 1.0
						else:
							tensor[args.input_symbols[b], j, i] = 1.0
					else:
						hot_array = quality_from_mode(args, rqual[i], b, args.input_symbols)
						if args.channels_last:
							tensor[j, i, :4] = hot_array
						else:
							tensor[:4, j, i] = hot_array

				elif b in defines.ambiguity_codes:
					if args.channels_last:
						tensor[j, i, :4] = defines.ambiguity_codes[b]
					else:
						tensor[:4, j, i] = defines.ambiguity_codes[b]
				
				else:
					raise ValueError('Error! Unknown symbol in seq block:', b)
					

			flags = flag_to_array(read.flag)
			for i in range(defines.read_flags):
				flag_str = 'flag_bit_'+ str(i)

				if flags[i] and flag_str in channel_map:
					if args.channels_last:
						tensor[j, flag_start:flag_end, channel_map[flag_str]] = 1.0
					else:
						tensor[channel_map[flag_str], j,  flag_start:flag_end] = 1.0
			
			if 'mapping_quality' in channel_map:
				if args.channels_last:
					tensor[j, flag_start:flag_end, channel_map['mapping_quality']] = float(read.mapping_quality)/defines.mapping_quality_max
				else:
					tensor[channel_map['mapping_quality'], j, flag_start:flag_end] = float(read.mapping_quality)/defines.mapping_quality_max

	return tensor



def good_reads_to_tensor(args, good_reads, ref_start, insert_dict):
	'''Create a read tensor based on a tensor channel map.

	Assumes read pairs have the same name. 
	Only loads reads that might align inside the tensor.

	Arguments:
		args.read_limit: maximum number of reads to return
		good_reads: list of reads to make arrays from
		ref_start: the beginning of the window in reference coordinates
		insert_dict: a dict mapping read indices to max insertions at that point.

	Returns:
		tensor: 3D read tensor.
	'''
	channel_map = defines.get_tensor_channel_map_from_args(args)
	tensor = np.zeros( defines.tensor_shape_from_args(args) )

	for j,read in enumerate(good_reads):

		rseq, rqual = sequence_and_qualities_from_read(args, read, ref_start, insert_dict)
		flag_start = -1
		flag_end = 0

		for i,b in enumerate(rseq):
			
			if i == args.window_size:
				break
			
			if b == defines.skip_char:
				continue
			elif flag_start == -1:
				flag_start = i
			else:
				flag_end = i

			if b in args.input_symbols:
				if b == defines.indel_char:
					if args.channels_last:
						tensor[j, i, args.input_symbols[b]] = 1.0
					else:
						tensor[args.input_symbols[b], j, i] = 1.0
				else:
					hot_array = quality_from_mode(args, rqual[i], b, args.input_symbols)
					if args.channels_last:
						tensor[j, i, :4] = hot_array
					else:
						tensor[:4, j, i] = hot_array

			elif b in defines.ambiguity_codes:
				if args.channels_last:
					tensor[j, i, :4] = defines.ambiguity_codes[b]
				else:
					tensor[:4, j, i] = defines.ambiguity_codes[b]
			
			else:
				raise ValueError('Error! Unknown symbol in seq block:', b)
				

		flags = flag_to_array(read.flag)
		for i in range(defines.read_flags):
			flag_str = 'flag_bit_'+ str(i)

			if flags[i] and flag_str in channel_map:
				if args.channels_last:
					tensor[j, flag_start:flag_end, channel_map[flag_str]] = 1.0
				else:
					tensor[channel_map[flag_str], j,  flag_start:flag_end] = 1.0
		
		if 'mapping_quality' in channel_map:
			if args.channels_last:
				tensor[j, flag_start:flag_end, channel_map['mapping_quality']] = float(read.mapping_quality)/defines.mapping_quality_max
			else:
				tensor[channel_map['mapping_quality'], j, flag_start:flag_end] = float(read.mapping_quality)/defines.mapping_quality_max

	return tensor



def sequence_and_qualities_from_read(args, read, ref_start, insert_dict):
	cur_idx = 0
	my_indel_dict = {}
	no_qual_filler = 0
	
	index_dif = ref_start - read.reference_start
	for t in read.cigartuples:
		my_ref_idx = cur_idx - index_dif
		if t[0] == defines.cigar_code['I'] and my_ref_idx in insert_dict:
			my_indel_dict[my_ref_idx] = insert_dict[my_ref_idx] - t[1]
		elif t[0] == defines.cigar_code['D']:
			my_indel_dict[my_ref_idx] = t[1]
		if t[0] in [defines.cigar_code['M'], defines.cigar_code['I'], defines.cigar_code['S'], defines.cigar_code['D']]:
			cur_idx += t[1]

	for k in insert_dict.keys():
		if k not in my_indel_dict:
			my_indel_dict[k] = insert_dict[k]

	rseq = read.query_alignment_sequence[:args.window_size]
	rqual = read.query_alignment_qualities[:args.window_size].tolist()

	if index_dif > 0:
		rseq = rseq[index_dif:] 
		rqual = rqual[index_dif:]
	elif index_dif < 0:
		rseq = defines.skip_char*(-index_dif) + rseq
		rqual = [no_qual_filler]*(-index_dif) + rqual

	for j in sorted(my_indel_dict.keys(), key=int, reverse=True):
		if j < 1:
			rseq = (defines.indel_char*my_indel_dict[j]) + rseq
			rqual = ([no_qual_filler]*my_indel_dict[j]) + rqual
		else:
			rseq = rseq[:j] + (defines.indel_char*my_indel_dict[j]) + rseq[j:]
			rqual = rqual[:j] + ([no_qual_filler]*my_indel_dict[j]) + rqual[j:]

	return rseq, rqual


def seq_block_to_image(args, reference, sequences, flags, qualities, mapping_qualities):
	debug = False

	strand_flag_idx = 4
	base_to_color = {'A':250.0, 'G':180.0, 'T':100.0, 'C':30.0, '*':15.0}
	tensor = np.zeros( (args.read_limit, args.window_size, 3) )


	for x,b in enumerate(reference):
		if x == args.window_size:
			break
		if b in args.input_symbols:			
			tensor[:, x, 0] = base_to_color[b]

	for y,l in enumerate(sequences):
		for x,b in enumerate(l):
			if x == args.window_size:
				break

			if b in args.input_symbols:
				qual_min = min(qualities[y][x], mapping_qualities[y])
				qual_color = 254.0 * (min(40, qual_min)/40.0)
				alpha = 0.2 if reference[x] == b else 1.0
				
				tensor[y, x, 0] = alpha*base_to_color[b]
				tensor[y, x, 1] = alpha*qual_color
				tensor[y, x, 2] = alpha*240.0 if flags[strand_flag_idx][y] else alpha*70.0 # grabs the strand flag

			elif b in defines.ambiguity_codes:
				continue
			elif b == defines.skip_char:
				continue
			else:
				raise ValueError('Error! Unknown symbol in seq block:', b)
				

	if debug:
		print(reference, '\n\n\n ~~~~~~ Becomes Tensor: ~~~~~~~ \n\n')
		print(tensor[:,:,0])

	return tensor


def image_as_matrix(image_path, expand_dims=False, shape=(224,224), channels_last=False):
	import cv2

	debug = False
	img = cv2.resize(cv2.imread(image_path), shape).astype(np.float32)
	img /= 255.0
	img[:,:,0] -= 0.5#(103.939 / 255.0)
	img[:,:,1] -= 0.5#(116.779 / 255.0)
	img[:,:,2] -= 0.5#(123.68 / 255.0)

	if debug:
		cv2.imshow('img', img)
		cv2.waitKey(1000)
	
	if not channels_last:
		img = img.transpose((2,0,1))

	if expand_dims:
		img = np.expand_dims(img, axis=0)
	
	return img


def train_valid_test_generators_from_args(args, with_positions=False):
	train_paths, valid_paths, test_paths = get_train_valid_test_paths(args)

	train_generator = tensor_generator_from_label_dirs_and_args(args, train_paths, with_positions)
	valid_generator = tensor_generator_from_label_dirs_and_args(args, valid_paths, with_positions)
	test_generator = tensor_generator_from_label_dirs_and_args(args, test_paths, with_positions)

	return train_generator, valid_generator, test_generator


def image_generator(args, train_paths, shape=(224,224)):
	"""Data generator of PNGs for DeepVariant.

	Assumes train paths contains examples in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each

	Yields:
		A tuple with a image_matrix theano channel ordering.
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""
	import cv2	
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels)) 
	image_exts = ['.png', '.jpg', '.jpeg', '.tif']
	image_counts = Counter()
	images = {}

	if args.channels_last:
		image_matrix = np.zeros((args.batch_size, shape[0], shape[1], 3))
	else:
		image_matrix = np.zeros((args.batch_size, 3, shape[0], shape[1]))

	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 

		images[label] = [os.path.join(tp, img) for img in os.listdir(tp) if os.path.splitext(img)[1] in image_exts]
		image_counts[label] = 0
		
	while True:
		cur_example = 0
		for label in images.keys():
			for i in range(per_batch_per_label):
				image_path = images[label][image_counts[label]]
				label_matrix[cur_example, label] = 1.0
				image_matrix[cur_example,:,:,:] = image_as_matrix(image_path, shape=shape, channels_last=args.channels_last)
				
				image_counts[label] += 1
				if image_counts[label] == len(images[label]):
					image_counts[label] = 0
					
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Image counts are:', image_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield (image_matrix, label_matrix)


def dna_annotation_generator(args, train_paths):
	"""Data generator of DNA and annotation tensors.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""		
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels)) 
	tensor_counts = Counter()
	tensors = {}

	if args.window_size > 0:
		channel_map = defines.get_tensor_channel_map_from_args(args)
		tensor = np.zeros((args.batch_size, args.window_size, len(channel_map)))

	annotation_data = np.zeros((args.batch_size, len(args.annotations)))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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
				label_matrix[cur_example, label] = 1.0
				with h5py.File(tensor_path,'r') as hf:
					annotation_data[cur_example,:] = np.array(hf.get(args.annotation_set))
					if args.window_size > 0:
						tensor[cur_example,:,:] = np.array(hf.get(args.tensor_map))
				
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator shuffled & looped over:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		if args.window_size > 0:
			yield ({args.tensor_map:tensor, args.annotation_set:annotation_data}, label_matrix)
		else:
			yield (annotation_data, label_matrix)



def pileup_tensor_generator(args, train_paths, include_annotations=False):
	"""Pileup Tensor generator of DNA and per site read summaries (i.e. pileup) tensors.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""		
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels)) 
	tensor_counts = Counter()
	tensors = {}

	if args.window_size > 0:
		channels = defines.get_reference_and_read_channels(args)
		tensor = np.zeros((args.batch_size, args.window_size, channels))

	annotation_data = np.zeros((args.batch_size, len(args.annotations)))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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
				label_matrix[cur_example, label] = 1.0
				with h5py.File(tensor_path,'r') as hf:
					if include_annotations:
						annotation_data[cur_example,:] = np.array(hf.get(args.tensor_map))
					if args.window_size > 0:
						tensor[cur_example,:,:] = np.array(hf.get('pileup_tensor'))
				
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator shuffled & looped over:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		if args.window_size > 0 and include_annotations:
			yield ({'pileup_tensor':tensor, args.annotation_set:annotation_data}, label_matrix)
		elif args.window_size > 0:
			yield ({'pileup_tensor':tensor}, label_matrix)
		else:
			yield (annotation_data, label_matrix)



def bqsr_tensor_generator(args, train_paths):
	"""Data generator of tensors with read and reference pair.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each


	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""	
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels) ) 
	tensor_counts = Counter()
	tensors = {}

	tensor = np.zeros((args.batch_size, args.window_size, len(args.input_symbols)))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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
					e = sys.exc_info()
					print('\nError', e, ' \n could be corrupt tensor at:', tensor_path )
					del tensors[label][tensor_counts[label]]
					continue
					
				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator shuffled & looped over:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield ({'read_tensor_input':tensor}, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))


def bqsr_tensor_annotation_generator(args, train_paths):
	"""Data generator of tensors with read and reference pair plus array of annotations.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each


	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""	
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels) ) 
	tensor_counts = Counter()
	tensors = {}

	annotation_data = np.zeros((args.batch_size, len(args.annotations)))
	tensor = np.zeros((args.batch_size, args.window_size, len(args.input_symbols)))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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
						tensor[cur_example] = np.array(hf.get(args.tensor_map))
						annotation_data[cur_example] = np.array(hf.get(args.annotation_set))
				except:
					e = sys.exc_info()
					print('\nError', e, ' \n could be corrupt tensor at:', tensor_path)
					del tensors[label][tensor_counts[label]]
					continue
					
				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator shuffled & looped over:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield ({args.tensor_map:tensor, args.annotation_set:annotation_data}, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))


def calling_tensors_generator(args, train_paths):
	'''Data generator of read tensors for calling variants and site labels for segmentation ground truth.

	Loops over all examples yielding args.batch_size examples.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: directory with hd5 calling tensors made with write_calling_tensors()

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	'''	
	from keras.utils import to_categorical # Lazy import because we don't want this file to be keras dependent

	tensors = {}
	stats = Counter()

	in_channels = defines.total_input_channels_from_args(args)
	if args.channels_last:
		tensor_shape = (args.read_limit, args.window_size, in_channels)
	else:
		tensor_shape = (in_channels, args.read_limit, args.window_size) 

	tensor = np.zeros(((args.batch_size,)+tensor_shape))
	label_matrix = np.zeros((args.batch_size, args.window_size, len(args.labels)))
	while True:
		
		for tp in train_paths:
			try: 
				with h5py.File(tp, 'r') as hf:
					tensor[stats['batch_index']] = np.array(hf.get('read_tensor'))
					label_matrix[stats['batch_index']] = to_categorical(np.array(hf.get('site_labels')), len(args.labels))

			except Exception as e:
				print('Exception for tensor at:', tp, '\n\n\nError is:', str(e))
				print('Expected tensor shape:', tensor_shape) #, 'but received shape:', np.array(hf.get('read_tensor')).shape)
				print('Expected site labels shape:',(args.window_size, len(args.labels))) #, 'received:', np.array(hf.get('site_labels')).shape)
				continue

			stats['batch_index'] += 1
			if stats['batch_index'] == args.batch_size:
				yield ({'read_tensor':tensor}, label_matrix)
				stats['batch_index'] = 0

		print('\n\nGenerator looped over all ', len(train_paths),' tensors, now shuffle them. Last tensor was:', train_paths[-1])
		np.random.shuffle(train_paths)


def calling_pileup_tensors_generator(args, train_paths):
	'''Data generator of pileup tensors for calling variants and site labels for segmentation ground truth.

	Loops over all examples yielding args.batch_size examples.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: directory with hd5 calling tensors made with write_calling_tensors()

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	'''	
	from keras.utils import to_categorical # Lazy import because we don't want this file to be keras dependent

	tensors = {}
	stats = Counter()

	channels = defines.get_reference_and_read_channels(args)
	in_shape = (args.window_size, channels)

	tensor = np.zeros(((args.batch_size,)+in_shape))
	label_matrix = np.zeros((args.batch_size, args.window_size, len(args.labels)))
	
	while True:	
		for tp in train_paths:
			try: 
				with h5py.File(tp, 'r') as hf:
					tensor[stats['batch_index']] = np.array(hf.get('pileup_tensor'))
					label_matrix[stats['batch_index']] = to_categorical(np.array(hf.get('site_labels')), len(args.labels))

			except Exception as e:
				print('\n\n\nException for tensor at:\n', tp, '\nError is:', str(e))
				print('Expected pileup tensor shape:', in_shape) #, 'but received shape:', np.array(hf.get('read_tensor')).shape)
				print('Expected site labels shape:',(args.window_size, len(args.labels))) #, 'received:', np.array(hf.get('site_labels')).shape)
				continue

			stats['batch_index'] += 1
			if stats['batch_index'] == args.batch_size:
				yield ({'pileup_tensor':tensor}, label_matrix)
				stats['batch_index'] = 0

		print('\n\nGenerator looped over all ', len(train_paths),' tensors, now shuffle them. Last tensor was:', train_paths[-1])
		np.random.shuffle(train_paths)


def tensor_generator(args, train_paths, tensor_shape):
	"""Data generator of tensors with reads, and flags.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each
		tensor_shape: Shape of the input data tensor

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""	
	debug = False
	per_batch_per_label = (args.batch_size // len(args.labels) ) 
	tensor_counts = Counter()
	tensors = {}

	tensor = np.zeros(((args.batch_size,)+tensor_shape))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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
				except Exception as e:
					print('Delete corrupt tensor at:', tensor_path)
					print('Error is:', str(e), 'Expected shape:', tensor_shape)
					print('in shape:', np.array(hf.get('read_tensor')).shape)
					del tensors[label][tensor_counts[label]]
					continue
					
				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator shuffled & looped over:', tensor_counts[label], 'examples of label:', label, '\n\nLast tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield ({'read_tensor':tensor}, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))


def tensor_annotation_generator(args, train_paths, tensor_shape):
	"""Data generator of tensors with reads, and annotations.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each
		tensor_shape: Shape of the input data tensor

	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""	
	debug = False
	eps = 1e-6
	per_batch_per_label = (args.batch_size // len(args.labels) ) 
	tensor_counts = Counter()
	tensors = {}

	tensor = np.zeros(((args.batch_size,)+tensor_shape))
	annotations = np.zeros((args.batch_size,len(args.annotations)))
	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 
		tensors[label] = [os.path.join(tp, t) for t in os.listdir(tp) if os.path.splitext(t)[1] in tensor_exts]
		tensor_counts[label] = 0
		
	if args.normalize_annotations:
		means_and_stds = np.zeros((len(args.annotations),2))
		norm_file = os.path.join(args.data_dir, 'means_and_stds.hd5')

		if not os.path.exists(norm_file):
			print('Normalization requested, but no means and stds file:', norm_file, '.  Inspecting dataset...')
			inspect_dataset(args)

		with h5py.File(norm_file, 'r') as hf:
			means_and_stds = np.array(hf.get('means_and_stds'))

	while True:
		cur_example = 0
		for label in tensors.keys():
			for i in range(per_batch_per_label):
				tensor_path = tensors[label][tensor_counts[label]]

				try:
					with h5py.File(tensor_path, 'r') as hf:
						tensor[cur_example] = np.array(hf.get(args.tensor_map))
						annotations[cur_example] = np.array(hf.get(args.annotation_set))
					if args.normalize_annotations:
						for i,a in enumerate(args.annotations):
							if annotations[cur_example][i] == 0:
								continue
							annotations[cur_example][i] -= means_and_stds[i,0]
							annotations[cur_example][i] /= (eps+means_and_stds[i,1])
						#print('Annotations are:', annotations[cur_example])

				except Exception as e:
					print('Delete corrupt tensor at:', tensor_path)
					print('Error is:', str(e), 'Expected shape:', tensor_shape)
					del tensors[label][tensor_counts[label]]
					continue
					
				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator looped over:', tensor_counts[label], 'examples of label:', label, '\n\nShuffled them. Last tensor was:', tensor_path)
					tensor_counts[label] = 0
				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)

		yield ({args.tensor_map:tensor, args.annotation_set:annotations}, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))		
		annotations = np.zeros((args.batch_size,len(args.annotations)))


def tensor_generator_from_label_dirs_and_args(args, train_paths, with_positions=False):
	"""Data generator of tensors with reads, and annotations.

	Assumes train paths contains example in labelled directories.
	Loops over all examples sampling args.batch_size examples
	uniformly from each label.

	Arguments:
		args: args object needed for batch_size, labels, and annotations
		train_paths: array of label directories with hd5 tensors within each
		with_positions: boolean if True will include a position string 
			(i.e. "1_1234_0" for tensor from contig one base 1234 and first allele)
			as the last element in each tensor tuple.
	Returns:
		A tuple with a dict of the input tensors 
		and a 1-Hot matrix (2D numpy array) of the labels.
	"""	
	debug = False

	batch = {}
	tensors = {}
	tensor_counts = Counter()
	per_batch_per_label = (args.batch_size // len(args.labels) ) 

	tm = defines.get_tensor_channel_map_from_args(args)
	if tm:
		tensor_shape = defines.tensor_shape_from_args(args)
		batch[args.tensor_map] = np.zeros(((args.batch_size,)+tensor_shape))
	
	if defines.annotations_from_args(args):
		batch[args.annotation_set] = np.zeros((args.batch_size, len(args.annotations)))
	
	if with_positions:
		positions = []

	label_matrix = np.zeros((args.batch_size, len(args.labels)))

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
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

				with h5py.File(tensor_path, 'r') as hf:
					for key in batch.keys():
						hf_tensor = hf.get(key)
						if hf_tensor:
							batch[key][cur_example] = np.array(hf_tensor)
						else:
							raise ValueError('Could not find tensor with key:'+key+ '\nAt hd5 path:'+tensor_path) 

				label_matrix[cur_example, label] = 1.0
				tensor_counts[label] += 1
				if tensor_counts[label] == len(tensors[label]):
					np.random.shuffle(tensors[label])
					print('\n\nGenerator looped over:', tensor_counts[label], 'examples of label:', label, '\n\nShuffled them. Last tensor was:', tensor_path)
					tensor_counts[label] = 0
				
				if with_positions:
					positions.append(position_string_from_tensor_name(tensor_path))

				cur_example += 1
				if cur_example == args.batch_size:
					break

		if debug:
			print('Tensor counts are:', tensor_counts, ' cur example:', cur_example, ' per b per label:', per_batch_per_label)
			print('batch keys:', batch.keys())

		if with_positions:
			yield (batch, label_matrix, positions)
			positions = []
		else:
			yield (batch, label_matrix)
		label_matrix = np.zeros((args.batch_size, len(args.labels)))		
		if with_positions and tm:
			tensor_shape = defines.tensor_shape_from_args(args)
			batch[args.tensor_map] = np.zeros(((args.batch_size,)+tensor_shape))
		
		if with_positions and defines.annotations_from_args(args):
			batch[args.annotation_set] = np.zeros((args.batch_size, len(args.annotations)))		


def big_batch_from_minibatch_generator(args, generator, with_positions=False):
	labels = []
	input_data = {}
	minibatches = args.samples // args.batch_size

	tm = defines.get_tensor_channel_map_from_args(args)
	if tm:
		input_data[args.tensor_map] = []

	annotations = defines.annotations_from_args(args)
	if annotations:
		input_data[args.annotation_set] = []	

	if with_positions:
		positions = []

	for _ in range(minibatches):
		next_batch = next(generator)
		if tm:
			input_data[args.tensor_map].extend(next_batch[0][args.tensor_map])
		if annotations:
			input_data[args.annotation_set].extend(next_batch[0][args.annotation_set])
		labels.extend(next_batch[1])
		if with_positions:
			positions.extend(next_batch[-1])

	for key in input_data:
		input_data[key] = np.array(input_data[key])
		print('Input tensor:', key, 'has shape:', input_data[key].shape)

	if with_positions:
		return input_data, np.array(labels), positions
	else:
		return input_data, np.array(labels)


def load_images_from_class_dirs(args, train_paths, shape=(224,224), per_class_max=2500, position_dict=None):
	import cv2
	count = 0
	train_set = []
	t_labels = []
	positions = []
	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 

		imgs = os.listdir(tp)
		count += 1
		this_t = 0
		for im in imgs:		
			fn, file_extension = os.path.splitext(im)
			if file_extension.lower() == '.gif':
				continue

			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			gpos = im.split('-')[-1]
			chrom = gpos.split('_')[0]
			pos = os.path.splitext(gpos.split('_')[1])[0]
			pos_str = chrom + '_' + pos
			if position_dict and pos_str not in position_dict:
				continue

			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			
			img = image_as_matrix(tp+'/'+im, shape=shape, channels_last=args.channels_last)

			positions.append(pos_str)
			train_set.append(img)
			t_labels.append(y_vector)
			this_t += 1

		
		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs), 'Loaded:', this_t)

	return (np.asarray(train_set), np.asarray(t_labels), np.asarray(positions))


def load_tensors_from_class_dirs(args, train_paths, per_class_max=2500, dataset_id='read_tensor'):
	count = 0

	positions = []
	tensors = []
	labels = []

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 
		imgs = os.listdir(tp)
		count += 1
		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs))
		this_t = 0
		for t in imgs:	
			this_t += 1
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			fn, file_extension = os.path.splitext(t)
			if not file_extension.lower() in tensor_exts:
				continue

			with h5py.File(tp+'/'+t, 'r') as hf:
				tensors.append(np.array(hf.get(dataset_id)))
				
			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0

			labels.append(y_vector)
			positions.append(position_string_from_tensor_name(t))

	return (np.asarray(tensors), np.asarray(labels), np.asarray(positions))


def load_tensors_and_annotations_from_class_dirs(args, train_paths, per_class_max=2500, position_dict=None):
	annotations = []
	positions = []
	tensors = []
	labels = []
	count = 0

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue

		label = args.labels[label_key] 
		imgs = os.listdir(tp)
		count += 1
		this_t = 0
		for t in imgs:	
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			fn, file_extension = os.path.splitext(t)
			if not file_extension.lower() in tensor_exts:
				continue

			with h5py.File(tp+'/'+t, 'r') as hf:
				tensors.append(np.array(hf.get(args.tensor_map)))
				annotations.append(np.array(hf.get(args.annotation_set)))

			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			labels.append(y_vector)
			positions.append(position_string_from_tensor_name(t))
			this_t += 1

		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs), 'Loaded:', this_t)

	return (np.asarray(tensors), np.asarray(annotations), np.asarray(labels), np.asarray(positions))


def load_bqsr_tensors_from_class_dirs(args, train_paths, per_class_max=4000):
	count = 0

	tensors = []
	labels = []

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 
		imgs = os.listdir(tp)
		count += 1
		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs))
		this_t = 0
		for t in imgs:	
			this_t += 1
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			fn, file_extension = os.path.splitext(t)
			if not file_extension.lower() in tensor_exts:
				continue

			with h5py.File(tp+'/'+t, 'r') as hf:
				tensors.append(np.array(hf.get(args.tensor_map)))
				
			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			labels.append(y_vector)

	return (np.asarray(tensors), np.asarray(labels))


def load_bqsr_tensors_annotations_from_class_dirs(args, train_paths, per_class_max=4000):
	count = 0

	annotations = []
	tensors = []
	labels = []

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key] 
		imgs = os.listdir(tp)
		count += 1
		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs))
		this_t = 0
		for t in imgs:	
			this_t += 1
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			fn, file_extension = os.path.splitext(t)
			if not file_extension.lower() in tensor_exts:
				continue

			with h5py.File(tp+'/'+t, 'r') as hf:
				tensors.append(np.array(hf.get(args.tensor_map)))
				annotations.append(np.array(hf.get(args.annotation_set)))
				
			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			labels.append(y_vector)

	return (np.asarray(tensors), np.asarray(annotations), np.asarray(labels))


def load_dna_annotations_positions_from_class_dirs(args, train_paths, per_class_max=4000, include_dna=True, include_annotations=True):
	count = 0

	annotation_data = []
	reference_data = []
	labels_data = []
	positions = []
	
	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			print('Skipping label directory:', label_key, ' which is not in args label set:', args.labels.keys())
			continue
		label = args.labels[label_key]		
		imgs = os.listdir(tp)
		count += 1
		print(count, " dir out of:", len(train_paths), tp, "has:", len(imgs))
		this_t = 0
		for t in imgs:	
			this_t += 1
			if this_t > per_class_max:
				print('Per class max reached. bailing at', this_t)
				break

			fn, file_extension = os.path.splitext(t)
			if not file_extension.lower() in tensor_exts:
				continue

			with h5py.File(tp+'/'+t, 'r') as hf:
				if include_annotations:
					annotation_data.append(np.array(hf.get(args.annotation_set)))
				if include_dna:
					reference_data.append(np.array(hf.get(args.tensor_map)))
				
			y_vector = np.zeros(len(args.labels)) # One hot Y vector of size labels, correct label is 1 all others are 0
			y_vector[label] = 1.0
			labels_data.append(y_vector)
			positions.append(position_string_from_tensor_name(t))

	if include_dna and include_annotations:
		return (np.asarray(reference_data), np.asarray(annotation_data), np.asarray(labels_data), np.asarray(positions))
	elif include_annotations:
		return (np.asarray(annotation_data), np.asarray(labels_data), np.asarray(positions))
	elif include_dna:
		return (np.asarray(reference_data), np.asarray(labels_data), np.asarray(positions))


def position_string_from_tensor_name(tensor_name):
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


def force_symlink(file1, file2):
	try:
		if not os.path.exists(os.path.dirname(file2)):
			os.makedirs(os.path.dirname(file2))
		os.symlink(file1, file2)
	except OSError as e:
		if e.errno == errno.EEXIST:
			os.remove(file2)
			os.symlink(file1, file2)


def split_and_symlink_images(args, valid_ratio=0.1):
	new_image_path = os.path.split(args.data_dir)[0] + '_split_symlinked/'
	train_path = os.path.join(new_image_path, 'train')
	valid_path = os.path.join(new_image_path, 'valid')

	count = 0
	for label in os.listdir(args.data_dir):
		if not os.path.isdir(args.data_dir+label):
			continue
		for img in [os.path.join(args.data_dir, label, img) for img in os.listdir(args.data_dir+label)]:
			dice = np.random.rand()
			if dice < valid_ratio:
				force_symlink(img, os.path.join(valid_path, label, os.path.basename(img)))
			else:
				force_symlink(img, os.path.join(train_path, label, os.path.basename(img)))
			count += 1
			if count%1000 == 0:
				print('symlinked:', count)


def get_path_to_train_valid_or_test(path, valid_ratio=0.1, test_ratio=0.2, valid_contig='-19_', test_contig='-20_'):
	dice = np.random.rand() 
	if dice < valid_ratio or valid_contig in path:
		return os.path.join(path, 'valid/')
	elif dice < valid_ratio+test_ratio or test_contig in path:	
		return os.path.join(path, 'test/')
	else:	
		return os.path.join(path, 'train/')


def get_train_valid_test_paths(args):
	train_dir = args.data_dir + 'train/'
	valid_dir = args.data_dir + 'valid/'
	test_dir = args.data_dir + 'test/'
	train_paths = [train_dir + tp for tp in sorted(os.listdir(train_dir)) if os.path.isdir(train_dir + tp)]
	valid_paths = [valid_dir + vp for vp in sorted(os.listdir(valid_dir)) if os.path.isdir(valid_dir + vp)]
	test_paths = [test_dir + vp for vp in sorted(os.listdir(test_dir)) if os.path.isdir(test_dir + vp)]		

	assert(len(train_paths) == len(valid_paths) == len(test_paths))

	return train_paths, valid_paths, test_paths


def get_train_valid_test_paths_all(args):
	train_dir = args.data_dir + 'train/'
	valid_dir = args.data_dir + 'valid/'
	test_dir = args.data_dir + 'test/'
	train_paths = [train_dir + tp for tp in sorted(os.listdir(train_dir))]
	valid_paths = [valid_dir + vp for vp in sorted(os.listdir(valid_dir))]
	test_paths = [test_dir + vp for vp in sorted(os.listdir(test_dir))]		

	return train_paths, valid_paths, test_paths


def split_and_move_data(args, valid_ratio=0.1, test_ratio=0.2):
	new_image_path = os.path.split(args.data_dir)[0] + '_split/'
	train_path = os.path.join(new_image_path, 'train')
	valid_path = os.path.join(new_image_path, 'valid')
	test_path = os.path.join(new_image_path, 'test')
	count = 0
	for label in os.listdir(args.data_dir):
		if not os.path.isdir(args.data_dir+label):
			continue
		for img in [os.path.join(args.data_dir, label, img) for img in os.listdir(args.data_dir+label)]:
			dice = np.random.rand()
			if dice < valid_ratio:
				if not os.path.exists(os.path.join(valid_path, label)):
					os.makedirs(os.path.join(valid_path, label))
				os.rename(img, os.path.join(valid_path, label, os.path.basename(img)))
			elif dice < valid_ratio+test_ratio:
				if not os.path.exists(os.path.join(test_path, label)):
					os.makedirs(os.path.join(test_path, label))
				os.rename(img, os.path.join(test_path, label, os.path.basename(img)))
			else:
				if not os.path.exists(os.path.join(train_path, label)):
					os.makedirs(os.path.join(train_path, label))				
				os.rename(img, os.path.join(train_path, label, os.path.basename(img)))
			count += 1
			if count%1000 == 0:
				print('Moved:', count)


def flag_to_array(flag):
	flags = []
	
	for i in range(defines.read_flags):
		flags.append((flag>>i)&1)

	return np.array(flags)


def add_flags_to_read_tensor(args, tensor, tensor_channel_map, flags):
	for k in tensor_channel_map.keys():
		if 'flag' in k:
			flag_bit = int(k.split('_')[-1])
			for read_idx in range(flags.shape[1]):
				if args.channels_last:
					tensor[read_idx, :, tensor_channel_map[k]] = flags[flag_bit, read_idx]
				else:
					tensor[tensor_channel_map[k], read_idx, :] = flags[flag_bit, read_idx]


def add_mq_to_read_tensor(args, tensor, tensor_channel_map, mapping_qualities):
	if not 'mapping_quality' in tensor_channel_map:
		return
		
	for read_idx, mq in enumerate(mapping_qualities):
		if args.channels_last:
			tensor[read_idx, :, tensor_channel_map['mapping_quality']] = float(mq) / defines.mapping_quality_max
		else:
			tensor[tensor_channel_map['mapping_quality'], read_idx, :] = float(mq) / defines.mapping_quality_max


def base_quality_to_phred_array(base_quality, base, base_dict):
	phred = np.zeros((4,))
	exponent = float(-base_quality) / 10.0
	p = 1.0-(10.0**exponent) # Convert to probability
	not_p = (1.0-p) / 3.0 # Error could be any of the other 3 bases
	not_base_quality = -10 * np.log10(not_p) # Back to Phred
	
	for b in base_dict.keys():
		if b == defines.indel_char:
			continue
		elif b == base:
			phred[base_dict[b]] = base_quality
		else:
			phred[base_dict[b]] = not_base_quality
	return phred


def base_quality_to_p_hot_array(base_quality, base, base_dict):
	phot = np.zeros((4,))
	exponent = float(-base_quality) / 10.0
	p = 1.0-(10.0**exponent)
	not_p = (1.0-p)/3.0

	for b in base_dict.keys():
		if b == base:
			phot[base_dict[b]] = p
		elif b == defines.indel_char:
			continue
		else:
			phot[base_dict[b]] = not_p

	return phot


def quality_from_mode(args, base_quality, base, base_dict):
	if args.base_quality_mode == 'phot':
		return base_quality_to_p_hot_array(base_quality, base, base_dict)
	elif args.base_quality_mode == 'phred':
		return base_quality_to_phred_array(base_quality, base, base_dict)
	elif args.base_quality_mode == '1hot':
		one_hot = np.zeros((4,))
		one_hot[base_dict[base]] = 1.0
		return one_hot
	else:
		raise ValueError('Error! Unknown base quality mode:', args.base_quality_mode)


def quality_from_mode_2bit(args, base_quality, base, base_dict):
	if args.base_quality_mode == 'phot':
		return base_quality_to_p_hot_array_2bit(base_quality, base, base_dict)
	elif args.base_quality_mode == 'phred':
		return base_quality_to_phred_array_2bit(base_quality, base, base_dict)
	elif args.base_quality_mode == '1hot':
		one_hot = np.array(base_dict[base])
		return one_hot
	else:
		raise ValueError('Error! Unknown base quality mode:', args.base_quality_mode)


def reads_to_tensor(args, sequences, qualities=None, reference_seq=None):
	debug = False

	in_channels = defines.get_reference_and_read_channels(args)
	if args.channels_last:
		tensor = np.zeros( (args.read_limit, args.window_size, in_channels) )
	else:
		tensor = np.zeros( (in_channels, args.read_limit, args.window_size) )

	for j,l in enumerate(sequences):
		for i,b in enumerate(l):
			if i == args.window_size:
				break
			if b in args.input_symbols:
				if b == defines.indel_char or qualities is None:
					if args.channels_last:
						tensor[j, i, args.input_symbols[b]] = 1.0
					else:
						tensor[args.input_symbols[b], j, i] = 1.0
				else:
					hot_array = quality_from_mode(args, qualities[j][i], b, args.input_symbols)
					if args.channels_last:
						tensor[j, i, :4] = hot_array
					else:
						tensor[:4, j, i] = hot_array

			elif b in defines.ambiguity_codes:
				if args.channels_last:
					tensor[j, i, :4] = defines.ambiguity_codes[b]
				else:
					tensor[:4, j, i] = defines.ambiguity_codes[b]
			elif b == defines.skip_char:
				continue
			else:
				raise ValueError('Error! Unknown symbol in seq block:', b)
				

	if reference_seq:
		reference_sequence_into_tensor(args, reference_seq, tensor)

	if debug:
		np.set_printoptions(threshold=np.inf)
		print(reference_seq, '<- reference sequence')
		print(sequences, '\n\n\n ~~~~~~ Becomes Tensor: ~~~~~~~ \n\n')
		print(tensor)

	return tensor


def reference_sequence_into_tensor(args, reference_seq, tensor):
	ref_offset = len(set(args.input_symbols.values()))
	for i,b in enumerate(reference_seq):
		if i == args.window_size:
			break
		if b in args.input_symbols:
			if args.channels_last:
				tensor[:, i, ref_offset+args.input_symbols[b]] = 1.0
			else:
				tensor[ref_offset+args.input_symbols[b], :, i] = 1.0
		elif b in defines.ambiguity_codes:
			if args.channels_last:
				tensor[:, i, ref_offset:ref_offset+4] = np.tile(defines.ambiguity_codes[b], (args.read_limit, 1))
			else:
				tensor[ref_offset:ref_offset+4, :, i] = np.transpose(np.tile(defines.ambiguity_codes[b], (args.read_limit, 1)))		



def reads_to_2bit_tensor(args, sequences, qualities=None, reference_seq=None):
	debug = False

	in_channels = defines.get_reference_and_read_channels(args)
	if args.channels_last:
		tensor = np.zeros( (args.read_limit, args.window_size, in_channels) )
	else:
		tensor = np.zeros( (in_channels, args.read_limit, args.window_size) )
	
	for j,l in enumerate(sequences):
		for i,b in enumerate(l):
			if i == args.window_size:
				break
			if b in args.input_symbols:
				if b == defines.indel_char or qualities is None:
					if args.channels_last:
						tensor[j, i, args.input_symbols[b]] = 1.0
					else:
						tensor[args.input_symbols[b], j, i] = 1.0
				else:
					hot_array = quality_from_mode_2bit(args, qualities[j][i], b, args.input_symbols)
					if args.channels_last:
						tensor[j, i, :2] = hot_array
					else:
						tensor[:2, j, i] = hot_array

			elif b in defines.ambiguity_codes or b == defines.skip_char:
				continue
			else:
				raise ValueError('Error! Unknown symbol in seq block:', b)
				

	if reference_seq:
		for i,b in enumerate(reference_seq):
			if i == args.window_size:
				break
			ref_offset = 3
			if b in args.input_symbols:
				if args.channels_last:
					tensor[:, i, ref_offset:ref_offset+2] = args.input_symbols[b]
				else:
					if b == defines.indel_char:
						tensor[ref_offset+args.input_symbols[b], :, i] = 1.0
					else:
						bcast = np.zeros((2, args.read_limit))
						bcast2 = np.zeros((2, 1))
						bcast2 = np.array(args.input_symbols[b])
						bcast[0,:] = bcast2[0]
						bcast[1,:] = bcast2[1]
						tensor[ref_offset:ref_offset+2, :, i] = bcast
			elif b in defines.ambiguity_codes:
				continue		

	if debug:
		np.set_printoptions(threshold=np.inf)
		print(reference_seq, '<- reference sequence')
		print(sequences, '\n\n\n ~~~~~~ Becomes Tensor: ~~~~~~~ \n\n')
		print(tensor, tensor.shape)

	return tensor


def read_tensor_to_pileup(args, read_tensor):
	tensor_map = defines.get_tensor_channel_map_from_args(args)
	channels = defines.get_reference_and_read_channels(args)
	pileup_tensor = np.zeros((args.window_size, channels))

	for i in range(args.window_size):
		for key in tensor_map:
			if 'read' not in key and 'reference' not in key:
				continue

			if 'read' in key and args.channels_last:
				pileup_tensor[i, tensor_map[key]] = np.sum(read_tensor[:, i, tensor_map[key]]) / args.window_size
			elif 'read' in key:
				pileup_tensor[i, tensor_map[key]] = np.sum(read_tensor[tensor_map[key], :, i]) / args.window_size
			elif 'reference' in key and args.channels_last:
				pileup_tensor[i, tensor_map[key]] = np.amax(read_tensor[:, i, tensor_map[key]])
			elif 'reference' in key:
				pileup_tensor[i, tensor_map[key]] = np.amax(read_tensor[tensor_map[key], :, i])
			else:
				raise ValueError('Error unexpected key:'+key)

			
	return pileup_tensor


def seq_block_to_tensor(args, seq_block, qualities=None):
	debug = False
	tensor = np.zeros( (len(args.input_symbols), args.read_limit+1, args.window_size) )

	lines = seq_block.split('\n')

	for j,l in enumerate(lines):

		for i,b in enumerate(l):
			if b in args.input_symbols:
				if b == defines.indel_char or j == 0 or qualities is None: # Ignore reference should be parameter
					tensor[ args.input_symbols[b], j, i] = 1.0
				else:
					qs = qualities[j-1]
					tensor[ args.input_symbols[b], j, :4] = float(qs[i])
			elif b in defines.ambiguity_codes:
				tensor[:4, j, i] = defines.ambiguity_codes[b]
			elif b == defines.skip_char:
				continue
			else:
				raise ValueError('Error! Unknown symbol in seq block:', b)
				

	if debug:
		print(seq_block, '\n\n\n ~~~~~~ Becomes Tensor: ~~~~~~~ \n\n')
		print(tensor)

	return tensor


def base_string_to_tensor(args, bases, qualities=None):
	assert(len(bases) == args.window_size)
	tensor = np.zeros( (args.window_size, len(args.input_symbols)) )

	for i,b in enumerate(bases):
		if b in args.input_symbols:
			if b == defines.indel_char or qualities is None:
				tensor[i, args.input_symbols[b]] = 1.0
			else:
				tensor[i, :4] = quality_from_mode(args, qualities[i], b, args.input_symbols)
		elif b in defines.ambiguity_codes:
			tensor[i, :4] = defines.ambiguity_codes[b]
		elif b == defines.skip_char:
			continue
		else:
			raise ValueError('Error! Unknown symbol in seq block:', b)
	
	return tensor


def scores_from_heng_li_filters(args, positions, mean_coverage=60):
	"""Get Heng Li hard filter status and truth status for given positions.

	Arguments
	args: args object needed for vcf paths and high-confidence bed file
	positions: array of strings where each string specifies genomic position
		e.g. 12_9999999 means contig 12 site 9999999

	Returns
		Two dicts one maps SNP positions to a tuple containing score and truth status
		And a similar dict for INDELs.

	"""
	stats = Counter()
	
	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.negative_vcf, 'r'))
	sam_file = pysam.AlignmentFile(args.bam_file, "rb")	
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))
		
	indel_data = {}
	snp_data = {}
	
	# Get the hard filter scores for each position
	for p in positions:
		chrom = p.split('_')[0]
		pos = int(p.split('_')[1])
		variants = vcf_ram.fetch(chrom, pos-1, pos+1)
		
		variant = None
		for v in variants:
			if v.POS == pos and v.CHROM == chrom:
				variant = v
		if not variant:
			stats['not in negative vcf'] += 1
			continue

		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['in ignore vcf'] += 1
			continue

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if variant_in_vcf(variant, vcf_nist) and in_bed:
			truth = 1
		elif in_bed:
			truth = 0
		else:
			stats['not in high confidence region'] += 1
			continue

		filtered = 1
		pileup_depth = 0
		supports_call = 0
		got_reverse = False
		got_forward = False

		pileup_columns = sam_file.pileup(chrom, pos-2, pos+2)
		for pileup_column in pileup_columns:
			if pileup_column.pos == pos-1:

				# Over coverage cutoff is 4 standard deviations from the Poisson mean
				pileup_depth = pileup_column.n
				if pileup_depth > mean_coverage + (4*np.sqrt(mean_coverage)):
					filtered = 0
					stats['SNP:'+str(variant.is_snp)+' Filtered for over coverage'] += 1

				for pileupread in pileup_column.pileups:

					# Query position is None if is_del or is_refskip is set.
					if not pileupread.is_del and not pileupread.is_refskip:
						read_base = pileupread.alignment.query_sequence[pileupread.query_position]
						if read_base in variant.ALT:
							supports_call += 1
							if pileupread.alignment.is_reverse:
								got_reverse = True
							if not pileupread.alignment.is_reverse:
								got_forward = True

		if pileup_depth == 0 or 0.3 > supports_call / pileup_depth:
			filtered = 0
			stats['SNP:'+str(variant.is_snp)+' Filtered by ratio of supporting reads'] += 1
		if not got_forward or not got_reverse:
			filtered = 0
			stats['SNP:'+str(variant.is_snp)+' Filtered by both strands support'] += 1
		if variant.QUAL < 30.0:
			filtered = 0
			stats['SNP:'+str(variant.is_snp)+' Filtered by qual < 30'] += 1
		if 0.001 > norm.cdf(float(variant.INFO['FS'])):
			filtered = 0
			stats['SNP:'+str(variant.is_snp)+' Filtered by Fisher Strand P-Value'] += 1

		if variant.is_snp:
			snp_data[p] = (filtered, truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			indel_data[p] = (filtered, truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['Not SNP or INDEL'] += 1

	for k in stats.keys():
		print(k, 'has:', stats[k])

	return snp_data, indel_data


def apply_hard_filters(filter_dict, annotations):
	passes = 1
	for k in filter_dict.keys():
		if k not in annotations:
			#print('missing annotation for', k)
			continue
		if filter_dict[k][0] == 'greater_than':
			if annotations[k] > filter_dict[k][1]:
				passes = 0
		elif filter_dict[k][0] == 'less_than':
			if annotations[k] < filter_dict[k][1]:
				passes = 0
	return passes


def score_from_hard_filters(filter_dict, annotations, means, variances):
	"""Return a signed distance in normalized annotation space.

	If a variant passes all the filters return the distance
	to the nearest threshold hyperplane.
	If a variant is filtered return the maximum distance to
	the failing theshold times -1. 
	This way worse variants have more negative scores.

	Arguments
		filter_dict: dict mapping annotation names to a tuple 
			specifying threshold and its direction e.g. 'QD':('lt', 2.0)
			mean a variant is filtered if QD is less than 2.0
		annotations: dict of a variants annotation values (variant.INFO)
		means: dict of mean values for each annotation
		variances: dict of variances for each annotation

	Return 
		float, the hard filter score
	"""
	score = 9e9
	eps = 1e-5
	
	for k in filter_dict.keys():
		if k not in annotations:
			#print 'missing annotation for', k
			continue

		normed_annotation = (annotations[k]-means[k]) / (np.sqrt(variances[k])+eps)
		normed_thresh = (filter_dict[k][1]-means[k]) / (np.sqrt(variances[k])+eps)
		
		t_minus_a = normed_thresh-normed_annotation
		if filter_dict[k][0] == 'greater_than':
			score = min(score, t_minus_a)
		elif filter_dict[k][0] == 'less_than':
			score = min(score, -t_minus_a)

	return score	


def scores_from_gatk_hard_filters(args, positions, distance_score=False):
	"""Get Hard filter scores and truth status for given positions.

	For more on hard filtering variants see: 
	https://software.broadinstitute.org/gatk/documentation/article.php?id=3225
	
	Arguments:
	args: args object needed for vcf paths and high-confidence bed file
	positions: array of strings where each string specifies genomic position
		e.g. 12_9999999 means contig 12 site 9999999

	Returns:
		Two dicts one maps SNP positions to a tuple containing score and truth status
		And a similar dict for INDELs.

	"""

	stats = Counter()
	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.negative_vcf, 'r'))
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))

	snp_data = {}
	indel_data = {}

	# Filters are specified by a tuple: the first element indicates which side of the threshold passes 
	# (i.e. greater_than or less_than). The second element is numerical value of the cuttoff.
	snp_filters = {'QD':('less_than', 2.0), 'FS':('greater_than', 60.0), 'SOR':('greater_than', 3.0),
					'ReadPosRankSum':('less_than', -8.0), 'MQ':('less_than', 40.0), 'MQRankSum':('less_than',-12.5),}
	
	indel_filters = {'QD':('less_than', 2.0), 'FS':('greater_than', 200.0), 'SOR':('greater_than', 10.0), 
					'ReadPosRankSum':('less_than', -20.0), 'InBreedingCoeff':('less_than', -20.0)}
	
	count = 0
	annotation_sum = {'QD':0, 'MQ':0, 'FS':0, 'SOR':0, 'MQRankSum':0, 'ReadPosRankSum':0}
	annotation_shifts = {'QD':0, 'MQ':0, 'FS':0, 'SOR':0, 'MQRankSum':0, 'ReadPosRankSum':0}
	annotation_sqr_sum = {'QD':0, 'MQ':0, 'FS':0, 'SOR':0, 'MQRankSum':0, 'ReadPosRankSum':0}

	# First loop collects statistics
	for p in positions:
		chrom = p.split('_')[0]
		pos = int(p.split('_')[1])
		variants = vcf_ram.fetch(chrom, pos-1, pos+1)
		
		variant = None
		for v in variants:
			if v.POS == pos and v.CHROM == chrom:
				variant = v
		if not variant:
			continue
		if not in_bed_file(bed_dict, variant.CHROM, variant.POS):
			continue

		for k in variant.INFO:
			if k in annotation_sum:
				if variant.INFO[k] == 0:
					continue
				if annotation_shifts[k] == 0:
					annotation_shifts[k] = variant.INFO[k]
				shifted_value = variant.INFO[k] - annotation_shifts[k]
				annotation_sum[k] += shifted_value
				annotation_sqr_sum[k] += shifted_value * shifted_value
				count += 1.0


	# Compute means and variances needed to normalize annotation space
	means = {}
	variances = {}
	for k in annotation_sum.keys():
		means[k] = (annotation_sum[k]/count) + annotation_shifts[k]
		variances[k] = (annotation_sqr_sum[k] - (annotation_sum[k]*annotation_sum[k]) / count) / count
		print('Annotation:', k, ' Has mean:', means[k], 'variance:', variances[k], 'std:', np.sqrt(variances[k]))


	# Now get the hard filter scores for each position
	for p in positions:
		chrom = p.split('_')[0]
		pos = int(p.split('_')[1])
		variants = vcf_ram.fetch(chrom, pos-1, pos+1)
		
		variant = None
		for v in variants:
			if v.POS == pos and v.CHROM == chrom:
				variant = v
		if not variant:
			stats['not in negative vcf'] += 1
			continue

		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['in ignore vcf'] += 1
			continue

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if variant_in_vcf(variant, vcf_nist) and in_bed:
			truth = 1
		elif in_bed:
			truth = 0
		else:
			stats['not in high confidence region'] += 1
			continue

		if variant.is_snp:
			if distance_score:
				score = score_from_hard_filters(snp_filters, variant.INFO, means, variances)
			else:
				score = apply_hard_filters(snp_filters, variant.INFO)
			snp_data[p] = (score, truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			if distance_score:
				score = score_from_hard_filters(indel_filters, variant.INFO, means, variances)
			else:
				score = apply_hard_filters(indel_filters, variant.INFO)
			indel_data[p] = (score, truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['not SNP or INDEL'] += 1

	for k in stats.keys():
		print(k, 'has:', stats[k])

	return snp_data, indel_data


def scores_from_positions(args, positions, score_key='VQSLOD', override_vcf=None):
	"""Get score and truth status for given positions.

	Arguments:
	args: args object needed for vcf paths and high-confidence bed file
	positions: array of strings where each string specifies genomic position
		e.g. 12_9999999 means contig 12 site 9999999
	score_key: The vcf annotation containing the score, 
		VQSLOD for VQSR AS_RF for random forests.

	Returns:
		Two dicts one maps SNP positions to a tuple containing score and truth status
		And a similar dict for INDELs.
	"""
	stats = Counter()

	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_truth = vcf.Reader(open(args.train_vcf, 'rb'))

	if override_vcf:
		vcf_ram = vcf.Reader(open(override_vcf, 'rb'))
		vcf_negative = vcf.Reader(open(args.negative_vcf, 'rb'))
	else:
		vcf_ram = vcf.Reader(open(args.negative_vcf, 'rb'))

	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'rb'))
	if args.include_vcf:
		vcf_include = vcf.Reader(open(args.include_vcf, 'rb'))

	snp_data = {}
	indel_data = {}

	for p in positions:
		gpos_parts = p.split('_')
		chrom = gpos_parts[0]
		pos = int(gpos_parts[1])
		allele_idx = None if len(gpos_parts) < 3 else int(gpos_parts[2])
		variants = vcf_ram.fetch(chrom, pos-1, pos+1)
		
		variant = None
		for v in variants:
			if v.POS == pos and v.CHROM == chrom:
				variant = v

		if not variant:
			stats['Not in negative vcf'] += 1
			continue

		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['In ignore vcf'] += 1
			continue

		if args.include_vcf and not variant_in_vcf(variant, vcf_include):
			stats['Not in include vcf'] += 1
			continue

		if score_key not in variant.INFO and score_key != 'QUAL':
			stats['No score key '+score_key] += 1
			continue	

		if score_key == 'AS_RF':
			if len(score) > 1: # Not worrying about multi allelics now...
				stats['Skipping Random Forest multi allelic'] += 1
				continue
			score = variant.INFO[score_key][0] 
		elif score_key == 'QUAL':
			score = variant.QUAL
		else:
			score = variant.INFO[score_key]

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if allele_idx and override_vcf:
			v_negative = variant_in_vcf(variant, vcf_negative)
			if not v_negative or allele_idx >= len(v_negative.ALT):
				stats['Variant allele missing from override_vcf VCF, wrong VCF perhaps?'] += 1
				continue
			allele = v_negative.ALT[allele_idx]
		elif allele_idx:
			if allele_idx >= len(variant.ALT):
				stats['Variant allele missing from negative VCF, wrong VCF perhaps?'] += 1
				continue
			allele = variant.ALT[allele_idx]

		if allele_idx and allele_in_vcf(allele, variant, vcf_truth):
			truth = 1
		elif variant_in_vcf(variant, vcf_truth) and in_bed:
			truth = 1
		elif in_bed:
			truth = 0
		else:
			stats['Not in high confidence region'] += 1
			continue

		if variant.is_snp:
			snp_data[p] = (score, truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			indel_data[p] = (score, truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['Not SNP or INDEL'] += 1

	for k in stats.keys():
		print(k, 'has:', stats[k])

	return snp_data, indel_data


def scores_from_vcf(args, score_keys=['VQSLOD'], override_vcf=None):
	'''Get score and truth status for given vcf.

	Arguments:
	args: args object needed for vcf paths and high-confidence bed file
	args.samples: Max number of examples per class
	score_keys: List of vcf annotations with variant score, (e.g. VQSLOD, or AS_RF for random forests.)

	Returns:
		snp_scores: Dict maps score keys to array of scores for each SNP
		snp_truth: Array of truth status for each SNP
		indel_scores: Dict maps score keys to array of scores for each INDEL
		indel_truth: Array of truth status for each INDEL
	'''
	stats = Counter()
	show_top_scores = False
	allele_specific_score_keys = ['AS_RF']

	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_omni = vcf.Reader(open(defines.omni_vcf, 'r'))
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r')) 

	if override_vcf:
		vcf_reader = vcf.Reader(open(override_vcf, 'r'))
	else:
		vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))
	if args.include_vcf:
		vcf_include = vcf.Reader(open(args.include_vcf, 'r'))
	if args.output_vcf:
		vcf_writer = vcf.Writer(open(args.output_vcf, 'w'), vcf_reader)


	snp_scores = {key:[] for key in score_keys}
	snp_positions = []
	snp_truth = []

	indel_scores = {key:[] for key in score_keys}
	indel_truth = []

	done = False
	for contig_key, np_intervals in sorted(bed_dict.items(), key=operator.itemgetter(0)):
		for start,stop in zip(np_intervals[0], np_intervals[1]):
			try:
				variants = vcf_reader.fetch(contig_key, start, stop)
			except ValueError as e:
				stats['Value Error on fetch'] += 1
				continue
			for variant in variants:
				for allele_idx, allele in enumerate(variant.ALT):
					stats['allele_count'] += 1
					if args.ignore_vcf and allele_in_vcf(allele, variant, vcf_ignore):
						stats['In ignore vcf'] += 1
						continue	
					
					if args.include_vcf and not allele_in_vcf(allele, variant, vcf_include):
						stats['Not in include vcf'] += 1
						continue

					in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
					if allele_in_vcf(allele, variant, vcf_nist) and in_bed:
						truth = 1
					elif in_bed:
						truth = 0
					else:
						stats['not in high confidence region'] += 1
						continue

					scores = []
					got_all_scores = True
					for score_key in score_keys:
						if score_key not in variant.INFO and 'QUAL' not in score_key:
							stats['No score key '+score_key] += 1
							got_all_scores = False
							break

						if score_key in allele_specific_score_keys:
							try:
								score = float(variant.INFO[score_key][allele_idx])
								if math.isnan(score):
									stats['NaN for score key '+score_key] += 1
									got_all_scores = False
									break
							except IndexError as e:
								stats['Missing allele specific score '+score_key] += 1
								got_all_scores = False
								break								 
							else:
								scores.append(float(variant.INFO[score_key][allele_idx]))
						elif score_key == 'QUAL':
							scores.append(float(variant.QUAL))
						else:
							scores.append(float(variant.INFO[score_key]))
					
					if got_all_scores:
						if variant.is_snp:
							for score_key, score in zip(score_keys, scores):
								snp_scores[score_key].append(score)
							snp_truth.append(truth)
							snp_positions.append(variant.CHROM+'_'+str(variant.POS)+'_'+str(allele_idx))
							stats['snp_'+str(truth)] += 1
						elif variant.is_indel:
							for score_key, score in zip(score_keys, scores):			
								indel_scores[score_key].append(score)
							indel_truth.append(truth)
							stats['indel_'+str(truth)] += 1			
						else:
							stats['not SNP or INDEL'] += 1
				
					if args.samples < len(snp_truth) and args.samples < len(indel_truth):
						done = True
			if done:
				break
		if done:
			break
			
	for k in stats.keys():
		print(k, 'has:', stats[k])
	print('Last variant was:', str(variant))

	if show_top_scores:
		idxsort = sorted(range(len(snp_scores['VQSLOD'])), key=snp_scores['VQSLOD'].__getitem__)
		score_sort = [snp_scores['VQSLOD'][i] for i in idxsort]
		truth_sort = [snp_truth[i] for i in idxsort]
		pos_sort = [snp_positions[i] for i in idxsort]
		for i in range(40):
			print(pos_sort[-(i+1)],(i+1),':',score_sort[-(i+1)],'truth:',truth_sort[-(i+1)])
	
	return snp_scores, snp_truth, indel_scores, indel_truth


def concordance_scores_from_vcf(args, score_key='VQSLOD', override_vcf=None):


	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_omni = vcf.Reader(open(defines.omni_vcf, 'r'))
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r')) 

	vcf_1 = vcf.Reader(open(args.negative_vcf, 'r')) 
	snp_data = {'scores':[], 'truth':[]}
	indel_data = {'scores':[], 'truth':[]}

	vcf_2 = vcf.Reader(open(args.negative_vcf_2, 'r'))
	snp_data_2 = {'scores':[], 'truth':[]}
	indel_data_2 = {'scores':[], 'truth':[]}

	vcf_3 = vcf.Reader(open(args.negative_vcf_3, 'r')) 
	snp_data_3 = {'scores':[], 'truth':[]}
	indel_data_3 = {'scores':[], 'truth':[]}

	vcf_writer = vcf.Writer(open('/dsde/working/mduran/modelFP.vcf', 'w'), vcf_1)
	
	for variant in vcf_1:

		#is this variant in the other two?
		v2 = variant_in_vcf(variant, vcf_2)
		v3 = variant_in_vcf(variant, vcf_3)

		if not (v2 and v3): #if its not in both of the others, continue
			continue

		score1 = float(variant.INFO[score_key])
		score2 = float(v2.INFO[score_key])
		score3 = float(v3.INFO[score_key])

		#get the truth status of each 

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		in_bed_2 = in_bed_file(bed_dict, v2.CHROM, v2.POS)
		in_bed_3 = in_bed_file(bed_dict, v3.CHROM, v3.POS)

		if variant_in_vcf(variant, vcf_nist) and in_bed:
			v1_truth = 1
		elif in_bed:
			v1_truth = 0
		else:
			continue

		if variant_in_vcf(v2, vcf_nist) and in_bed_2:
			v2_truth = 1
		elif in_bed:
			v2_truth = 0
		else:
			continue

		if variant_in_vcf(v3, vcf_nist) and in_bed_3:
			v3_truth = 1
		elif in_bed:
			v3_truth = 0
		else:
			continue

		if score1 > score2 and score1 > score3 and v1_truth==0:
			#print(score1, score2, score3, variant.POS)
			vcf_writer.write_record(variant)



		if variant.is_snp and v2.is_snp and v3.is_snp:

			snp_data['scores'].append(score1)
			snp_data['truth'].append(v1_truth)

			snp_data_2['scores'].append(score2)
			snp_data_2['truth'].append(v2_truth)

			snp_data_3['scores'].append(score3)
			snp_data_3['truth'].append(v3_truth)


		elif variant.is_indel and v2.is_indel and v3.is_indel:


			indel_data['scores'].append(score1)
			indel_data['truth'].append(v1_truth)

			indel_data_2['scores'].append(score2)
			indel_data_2['truth'].append(v2_truth)

			indel_data_3['scores'].append(score3)
			indel_data_3['truth'].append(v3_truth)

	vcf_writer.close()
		
	return snp_data, indel_data, snp_data_2, indel_data_2, snp_data_3, indel_data_3


def scores_from_gnomad_vcf(args, score_keys=['VQSLOD']):
	'''Get score and truth status for given vcf.

	Arguments:
	args: args object needed for vcf paths and high-confidence bed file
	score_keys: List of vcf annotations with a score, VQSLOD for VQSR, AS_RF for random forests.

	Returns:
		snp_scores: Dict maps score keys to array of scores for each SNP
		snp_truth: Array of truth status for each SNP
		indel_scores: Dict maps score keys to array of scores for each INDEL
		indel_truth: Array of truth status for each INDEL
	'''
	stats = Counter()

	gnomads = gnomads_to_dict()
	bed_dict = bed_file_to_dict(args.bed_file)
	
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_omni = vcf.Reader(open(defines.omni_vcf, 'r'))
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r')) 
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))
	if args.include_vcf:
		vcf_include = vcf.Reader(open(args.include_vcf, 'r'))
	
	snp_scores = {key:[] for key in score_keys}
	snp_truth = []

	indel_scores = {key:[] for key in score_keys}
	indel_truth = []

	for variant in vcf_reader:	
		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['In ignore vcf'] += 1
			continue	
		
		if args.include_vcf and not variant_in_vcf(variant, vcf_include):
			stats['Not in include vcf'] += 1
			continue

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if variant_in_vcf(variant, vcf_nist) and in_bed:
			truth = 1
		elif in_bed:
			truth = 0
		else:
			stats['Not in high confidence region'] += 1
			continue

		if variant.CHROM not in gnomads:
			stats['gnomAD missing chrom:'+variant.CHROM] += 1
			continue

		variants = gnomads[variant.CHROM].fetch(variant.CHROM, variant.POS-1, variant.POS+1)
		gnomad_variant = None
		for v in variants:
			if v.POS == variant.POS and v.CHROM == variant.CHROM:
				gnomad_variant = v
		if not gnomad_variant:
			stats['Variant not in gnomAD'] += 1
			continue

		missing_key = False
		for score_key in score_keys:
			if score_key == 'VQSR Single Sample':
				if 'VQSLOD' not in variant.INFO:
					stats['No single sample vqslod'] += 1
					missing_key = True
			elif score_key not in gnomad_variant.INFO:
				stats['No score key:'+score_key] += 1
				missing_key = True
		if missing_key:
			continue	

		scores = {}
		for score_key in score_keys:
			if score_key == 'VQSLOD':
				scores[score_key] = gnomad_variant.INFO[score_key]
			elif score_key == 'VQSR Single Sample':
				scores[score_key] = variant.INFO['VQSLOD']
			elif score_key == 'AS_RF':
				scores[score_key] = score_from_gnomad_site(args, gnomad_variant, variant, vcf_mills, vcf_omni, stats)
				if scores[score_key] is None:
					missing_key = True
					stats["Missing AS_RF"] += 1
		if missing_key:		
			continue

		if variant.is_snp:
			for score_key in scores:
				snp_scores[score_key].append(scores[score_key])
			snp_truth.append(truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			for score_key in scores:
				indel_scores[score_key].append(scores[score_key])
			indel_truth.append(truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['Not SNP or INDEL'] += 1

		if len(snp_truth)%200 == 0:
			for k in stats.keys():
				print(k, 'has:', stats[k])			
			print('last variant was:', str(variant))

		if args.samples < len(snp_truth) and args.samples < len(indel_truth):
			break

	for k in stats.keys():
		print(k, 'has:', stats[k])
	print('last variant was:', str(variant))
	
	return snp_scores, snp_truth, indel_scores, indel_truth


def scores_from_gnomad_like_vcf(args, score_keys=['VQSLOD']):
	'''Get score and truth status for given vcf.

	Arguments:
	args: args object needed for vcf paths and high-confidence bed file
	args.samples: Max number of examples per class
	score_key: The vcf annotation with the score, VQSLOD for VQSR AS_RF for random forests.

	Returns:
		Two dicts one maps SNP positions to a tuple containing score and truth status
		And a similar dict for INDELs.
	'''
	stats = Counter()

	bed_dict = bed_file_to_dict(args.bed_file)
	
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_omni = vcf.Reader(open(defines.omni_vcf, 'r'))
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r')) 
	vcf_reader = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_gnomad = vcf.Reader(open(args.negative_vcf_2, 'r'))
	
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))
	if args.include_vcf:
		vcf_include = vcf.Reader(open(args.include_vcf, 'r'))
	
	snp_scores = {key:[] for key in score_keys}
	snp_truth = []

	indel_scores = {key:[] for key in score_keys}
	indel_truth = []

	for variant in vcf_reader.fetch(args.chrom):
		if not variant_in_vcf(variant, vcf_gnomad):
			stats['Not gnomad-like vcf'] += 1
			continue				

		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['In ignore vcf'] += 1
			continue	
		
		if args.include_vcf and not variant_in_vcf(variant, vcf_include):
			stats['Not in include vcf'] += 1
			continue

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if variant_in_vcf(variant, vcf_nist) and in_bed:
			truth = 1
		elif in_bed:
			truth = 0
		else:
			stats['Not in high confidence region'] += 1
			continue


		variants = vcf_gnomad.fetch(variant.CHROM, variant.POS-1, variant.POS+1)
		gnomad_variant = None
		for v in variants:
			if v.POS == variant.POS and v.CHROM == variant.CHROM:
				gnomad_variant = v
		if not gnomad_variant:
			stats['Variant not in gnomAD'] += 1
			continue

		missing_key = False
		for score_key in score_keys:
			if score_key == 'VQSR Single Sample' and 'VQSLOD' not in variant.INFO:
				stats['No single sample vqslod'] += 1
				missing_key = True
			elif score_key not in gnomad_variant.INFO:
				stats['No score key:'+score_key] += 1
				missing_key = True
		if missing_key:
			continue	

		scores = {}
		for score_key in score_keys:
			if score_key == 'VQSLOD':
				scores[score_key] = gnomad_variant.INFO[score_key]
			elif score_key == 'VQSR Single Sample':
				scores[score_key] = variant.INFO['VQSLOD']
			elif score_key == 'CNN_SCORE':
				scores[score_key] = float(gnomad_variant.INFO[score_key][0])
			else:
				scores[score_key] = score_from_gnomad_site(args, gnomad_variant, variant, vcf_mills, vcf_omni, stats)
				if scores[score_key] is None:
					missing_key = True
		if missing_key:
			continue

		if variant.is_snp:
			for score_key in scores:
				snp_scores[score_key].append(scores[score_key])
			snp_truth.append(truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			for score_key in scores:
				indel_scores[score_key].append(scores[score_key])
			indel_truth.append(truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['Not SNP or INDEL'] += 1

		if len(snp_truth)%500 == 0:
			for k in stats.keys():
				print(k, 'has:', stats[k])
			print('last variant was:', str(variant))
			for score_key in score_keys:	
				print('last snp ',score_key,' was:', snp_scores[score_key][-1])					

		if args.samples < len(snp_truth) and args.samples < len(indel_truth):
			break

	for k in stats.keys():
		print(k, 'has:', stats[k])
	for score_key in score_keys:	
		print('last snp ',score_key,' was:', snp_scores[score_key][-1])					
	print('last variant was:', str(variant), 'last score was:')

	
	return snp_scores, snp_truth, indel_scores, indel_truth



def gnomads_to_dict():
	"""Open and load gnomAD autosomal vcfs into a dict.

	Returns:
		A dicts of vcf readers for each contig vcf from the gnomAD callset.
	"""		
	gnomads = {}

	for i in range(1,23):
		gnomads[str(i)] = vcf.Reader(open(defines.gnomad_prefix+str(i)+'.vcf.gz', 'r'))
	gnomads['X'] = vcf.Reader(open(defines.gnomad_prefix+'X.vcf.gz', 'r'))
	
	return gnomads


def score_from_gnomad_site(args, variant, v_negative, vcf_mills, vcf_omni, stats):
	'''gnomAD Random Forest Allele Specific score for a given variant.
	
	Skip positive training sites included in the training VCFs (mills and omni)
	And negative training sites according to hard filter cutoffs.
	See https://macarthurlab.org/2017/02/27/the-genome-aggregation-database-gnomad/

	Arguments
		args.multiallelics: How to treat multi allelic sites
		variant: the variant of interest from gnomAD VCF
		v_negative: A called variant, could be positive or negative
		vcf_mills: Mills truth VCF
		vcf_omni: Omni truth VCF
		stats: Counter to keep track of what we've seen
	
	Returns:
		None if site was used for training or is filtered, otherwise the allele specific score
	'''

	# Handle Random Forest training sites
	if args.random_forest_training_sites == 'ignore' or args.random_forest_training_sites == 'only':
		is_training_site = False

		# Positive training sites
		if variant_in_vcf(variant, vcf_mills):
			stats[args.random_forest_training_sites+' rf training example, mills'] += 1
			is_training_site = True
		if variant_in_vcf(variant, vcf_omni):
			stats[args.random_forest_training_sites+' rf training example, omni'] += 1
			is_training_site = True

		# Negative training sites	
		if variant.INFO['QD'] < 2:
			stats[args.random_forest_training_sites+' rf negative training example, QD'] += 1
			is_training_site = True
		if variant.INFO['FS'] > 60:
			stats[args.random_forest_training_sites+' rf negative training example, FS'] += 1
			is_training_site = True			
		if variant.INFO['MQ'] < 30:
			stats[args.random_forest_training_sites+' rf negative training example, MQ'] += 1
			is_training_site = True

		if args.random_forest_training_sites == 'ignore' and is_training_site:
			return
		if args.random_forest_training_sites == 'only' and not is_training_site:
			return

	got_score = False
	for s, a, ac in zip(variant.INFO['AS_RF'], variant.ALT, variant.INFO['AC']):
		for v_alt in v_negative.ALT:
			if a == v_alt:
				score = s
				got_score = True

				if ac > args.gnomad_ac_max:
					stats['AC too high (above '+ str(args.gnomad_ac_max) +') filter'] += 1
					return	

				if ac < args.gnomad_ac_min:
					stats['AC too low (below '+ str(args.gnomad_ac_min) +') filter'] += 1
					return


	if len(variant.INFO['AS_RF']) > 1:
		stats['rf multi allelic, ' +  args.multiallelics] += 1
		if args.multiallelics == 'ignore':
			return
	elif args.multiallelics == 'only':
		return	

	if not got_score: 
		#print('No RF Score:', variant.INFO['AS_RF'], 'alts:', variant.ALT, ' negative alts:', v_negative.ALT)
		stats['RF could NOT get score'] += 1
		return

	return score	


def gnomad_scores_from_positions(args, positions, score_key='VQSLOD'):
	"""Get gnomAD score and truth status for given positions.

	Similar to above except scores from the gnomad callset rather than a VCF
	Currently this function skips multi-allelic sites.

	Arguments:
		args: args object needed for vcf paths and high-confidence bed file
		positions: array of strings where each string specifies genomic position
			e.g. 12_9999999 means contig 12 site 9999999
		score_key: The vcf annotation containing the score, 
			VQSLOD for VQSR, AS_RF for random forests.

	Returns:
		Two dicts one maps SNP positions to a tuple containing score and truth status
		And a similar dict for INDELs.
	"""	
	stats = Counter()
	gnomads = gnomads_to_dict()
	bed_dict = bed_file_to_dict(args.bed_file)
	vcf_nist = vcf.Reader(open(args.train_vcf, 'r'))
	vcf_omni = vcf.Reader(open(defines.omni_vcf, 'r'))
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r')) 	
	vcf_negative = vcf.Reader(open(args.negative_vcf, 'r'))
	
	if args.ignore_vcf:
		vcf_ignore = vcf.Reader(open(args.ignore_vcf, 'r'))
	if args.include_vcf:
		include_vcf = vcf.Reader(open(args.include_vcf, 'r'))

	snp_data = {}
	indel_data = {}

	for p in positions:
		p_split = p.split('_')
		chrom = p_split[0]
		pos = int(p_split[1])

		allele_idx = None
		if len(p_split) > 2:
			allele_idx = int(p_split[2])

		variant = None
		variants = gnomads[chrom].fetch(chrom, pos-1, pos)
		for v in variants:
			if v.POS == pos and v.CHROM == chrom:
				variant = v
				v_negative = variant_in_vcf(variant, vcf_negative)

		if not variant:
			stats['Not in gnomad'] += 1
			continue

		if not v_negative:
			stats['Not in negative VCF'] += 1
			continue

		if args.ignore_vcf and variant_in_vcf(variant, vcf_ignore):
			stats['In ignore vcf'] += 1
			continue

		if args.include_vcf and not variant_in_vcf(variant, include_vcf):
			stats['Not in include vcf'] += 1
			continue	

		if score_key not in variant.INFO:
			stats['No score key '+score_key] += 1
			continue	
		
		if score_key == 'VQSLOD':
			score = variant.INFO[score_key]
		elif score_key == 'AS_RF':

			score = score_from_gnomad_site(args, variant, v_negative, vcf_mills, vcf_omni, stats)
			if score is None:
				continue

		in_bed = in_bed_file(bed_dict, variant.CHROM, variant.POS)
		if not in_bed:
			stats['Not in high confidence region'] += 1
			continue
		

		if allele_idx:
			if allele_idx >= len(v_negative.ALT):
				stats['Allele not in negative VCF'] += 1
				continue	
			elif allele_in_vcf(v_negative.ALT[allele_idx], v_negative, vcf_nist):
				truth = 1
			else:
				truth = 0
		elif not allele_idx and variant_in_vcf(variant, vcf_nist):
			truth = 1			
		else:
			truth = 0

		if variant.is_snp:
			snp_data[p] = (score, truth)
			stats['snp_'+str(truth)] += 1
		elif variant.is_indel:
			indel_data[p] = (score, truth)
			stats['indel_'+str(truth)] += 1			
		else:
			stats['Not SNP or INDEL'] += 1
			
	for k in stats.keys():
		print(k, 'has:', stats[k])

	return snp_data, indel_data


def inspect_read_tensors(args):
	train_dir = args.data_dir + 'train/'
	train_paths = [train_dir + tp for tp in sorted(os.listdir(train_dir)) if os.path.isdir(train_dir + tp)]
	
	tensors = {}
	tensor_counts = Counter()

	for tp in train_paths:
		label_key = os.path.basename(tp)
		if label_key not in args.labels:
			continue
		label = args.labels[label_key] 
		tensors[label] = [os.path.join(tp, t) for t in os.listdir(tp) if os.path.splitext(t)[1] in tensor_exts]
		tensor_counts[label] = 0

	cur_example = 0

	for label in tensors.keys():
		tensor_path = tensors[label][tensor_counts[label]]
		with h5py.File(tensor_path,'r') as hf:
			tensor = np.array(hf.get('read_tensor'))
			plots.read_tensor_to_image(args, tensor)

			
		tensor_counts[label] += 1
		if tensor_counts[label] == len(tensors[label]):
			print('\nGenerator looped over all:', tensor_counts[label], 'examples of label:', label, '\nLast tensor was:', tensor_path)
			tensor_counts[label] = 0
			

def inspect_dataset(args):
	stats = Counter()
	purines = ['A', 'G']
	pyrimidines = ['T', 'C']
	data_paths = get_train_valid_test_paths(args)
	vcf_ram = vcf.Reader(open(args.negative_vcf, 'r'))

	if args.normalize_annotations:
		norms = {a:[0,0,0,0] for a in args.annotations} # X, X^2, count, k for shifted variance calculation
		maxed_out = False

	for dp in data_paths:
		for tp in dp:
			cur_label = os.path.basename(tp)
			cur_tensors = os.listdir(tp)
			stats[cur_label] += len(cur_tensors)
			stats['total'] += len(cur_tensors)
			for t in cur_tensors:
				gpos = t.split('-')[-1]
				chrom = gpos.split('_')[0]
				pos = int(os.path.splitext(gpos.split('_')[1])[0])

				variants = vcf_ram.fetch(chrom, pos-1, pos)
				for v in variants:
					if v.POS == pos and v.is_snp and 'SNP' in cur_label: 
						# Check transition or transverion
						if (v.REF in purines and v.ALT[0] in purines) or (v.REF in pyrimidines and v.ALT[0] in pyrimidines):
							stats[cur_label+' transitions'] += 1
						else:
							stats[cur_label+' transversions'] += 1
					elif v.POS == pos and v.is_deletion:
						stats[cur_label+' deletion'] += 1 
					elif v.POS == pos and v.is_indel:
						stats[cur_label+' insertion'] += 1

					if args.normalize_annotations and v.POS == pos and not maxed_out:
						with h5py.File(os.path.join(tp,t),'r') as hf:
							annotation_data = np.array(hf.get(args.anotation_set))
							for i,a in enumerate(args.annotations):
								if annotation_data[i] == 0:
									continue
								if norms[a][3] == 0:
									norms[a][3] = annotation_data[i]
								norms[a][0] += annotation_data[i]-norms[a][3]
								norms[a][1] += (annotation_data[i]-norms[a][3])*(annotation_data[i]-norms[a][3])
								norms[a][2] += 1
								if norms[a][2] == args.max_normalize_sites:
									maxed_out = True


	for k in ['SNP', 'NOT_SNP']:
		stats[k+' Ti/Tv'] = stats[k+' transitions'] / (float(stats[k+' transversions']) + 1e-7)
	for k in ['INDEL', 'NOT_INDEL']:
		stats[k+' Insertion/Deletion'] = stats[k+' insertion'] / (float(stats[k+' deletion']) + 1e-7)
	
	for k, v in sorted(stats.items()):		
		if k in args.labels:
			print('%s has: %d tensors %2.0f percent' % (k, stats[k], (100*stats[k] / (float(stats['total']) + 1e-7))))
		else:
			print('%s has: %.2f' % (k, stats[k]))
	
	dataset_summary_latex_table_line(stats)
	if args.normalize_annotations:
		means_and_stds = np.zeros((len(args.annotations), 2))
		for i,a in enumerate(args.annotations):
			means_and_stds[i,0] = (norms[a][0] / norms[a][2]) + norms[a][3]
			var = (norms[a][1] - (norms[a][0]*norms[a][0])/norms[a][2]) / norms[a][2]
			means_and_stds[i,1] = np.sqrt(var)
			print('Annotation:', a, ' Has mean:', means_and_stds[i,0], 'variance:', var, 'std:', means_and_stds[i,1])

		with h5py.File(os.path.join(args.data_dir, 'means_and_stds.hd5'), 'w') as hf:
			hf.create_dataset('means_and_stds', data=means_and_stds)


def dataset_summary_latex_table_line(stats):
	print('\n\n')
	lkeys = ['SNP', 'INDEL', 'NOT_SNP', 'NOT_INDEL']
	for k in lkeys:
		print(" & %d (%2.0f\\%%)" % (stats[k]//1000, (100*stats[k] / (stats['total'] + 1e-7))), end='')
	skeys = ['SNP Ti/Tv', 'NOT_SNP Ti/Tv', 'INDEL Insertion/Deletion', 'NOT_INDEL Insertion/Deletion']
	for k in skeys:
		print(' & %.2f ' % stats[k], end='')
	print('\n\n')


def write_tranches(args):
	tranches = [0.9, 0.95, 0.99]
	score_key = 'VQSLOD'
	vcf_mills = vcf.Reader(open(defines.mills_vcf, 'r'))
	vcf_hapmap = vcf.Reader(open(defines.hapmap_vcf, 'r'))	
	vcf_negative = vcf.Reader(open(args.negative_vcf, 'r'))

	scores = []

	# print('Iterate over mills.')

	# for variant in vcf_mills.fetch(args.chrom):
	# 	v_scored = variant_in_vcf(variant, vcf_negative)
	# 	if not v_scored:
	# 		continue
	# 	scores.append(v_scored.INFO[score_key])

	print('Iterate over hapmap.')

	for variant in vcf_hapmap.fetch(args.chrom):
		v_scored = variant_in_vcf(variant, vcf_negative)
		if not v_scored:
			continue
		scores.append(v_scored.INFO[score_key])

	print('got', len(scores), ' in mills.')

	scores = sorted(scores)
	for t in tranches:
		t_score = scores[int(clamp(t*len(scores), 0, len(scores)-1))]
		print('At tranch:', t, ' score key:', score_key, ' cutoff is:', t_score)


def inspect_gnomad_low_ac(args):
	stats = Counter()
	gnomads = gnomads_to_dict()
	for variant in gnomads['1']:
		for i,a in enumerate(variant.ALT):
			if int(variant.INFO['AC'][i]) < 2:
				stats['low_ac'] += 1
			stats['total'] += 1
		if stats['total'] > 300000:
			break

	for k, v in sorted(stats.items()):
		print(k, 'has:', stats[k])
	print('ratio: %0.2f' % (stats['low_ac']/ stats['total']))


def combine_vcfs(args):
	stats = Counter()
	vcf_negative = vcf.Reader(open(args.negative_vcf, 'r'))
	vcf_ram = vcf.Reader(open(args.negative_vcf_2, 'r'))
	vcf_writer = vcf.Writer(open(args.output_vcf, 'w'), vcf_negative)

	for variant in vcf_negative:
		vqual = variant_in_vcf(variant, vcf_ram)
		if not vqual:
			stats['Variant not in '+args.negative_vcf_2] += 1
			continue

		variant.QUAL = vqual.QUAL
		variant.FILTER = vqual.FILTER
		vcf_writer.write_record(variant)

	vcf_writer.close()

	for k, v in sorted(stats.items()):
		print(k, 'has:', stats[k])


def split_data(datasets, valid_ratio=0.1, test_ratio=0.4):
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
		test.append(  d[valid_idx : valid_idx+test_idx] )
		train.append( d[valid_idx+test_idx: ] )

	return train, valid, test


def simple_vcf_writer():
	vcf_reader = vcf.Reader(open('path/to/vcf.gz', 'r'))
	vcf_writer = vcf.Writer(open('path/to/out.vcf', 'w'), vcf_reader)
	for variant in vcf_reader:
		variant.INFO['AF'] = 1.0
		vcf_writer.write_record(variant)


def shuffle_in_unison(a, b):
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)


def shuffle_all_in_unison(data):
	rng_state = np.random.get_state()
	for a in data:
		np.random.set_state(rng_state)
		np.random.shuffle(a)


# Samples must be in first axis
def concat_and_shuffle(data_label_tuple1, data_label_tuple2):
	concat_data = np.concatenate((data_label_tuple1[0], data_label_tuple2[0]))
	concat_labels = np.concatenate((data_label_tuple1[1], data_label_tuple2[1]))
	shuffle_in_unison(concat_data, concat_labels)
	return (concat_data, concat_labels)

def concat_and_shuffle_all(tuple1, tuple2):
	concat = []
	for a,b in zip(tuple1, tuple2):
		concat.append(np.concatenate((a, b)))

	shuffle_all_in_unison(concat)
	return concat


def sample_from_fasta(record_dict):
	c_idx = str(np.random.randint(1,20))
	contig = record_dict[c_idx]
	p_idx = np.random.randint(len(contig))
	return c_idx, p_idx


def sample_from_bed(bed_dict, contig_key_prefix=''):
	contig_key = contig_key_prefix + str(np.random.randint(1,20))
	lowers = bed_dict[contig_key][0]
	uppers = bed_dict[contig_key][1]

	idx = np.random.randint(len(lowers))
	mid_pos = (lowers[idx] + uppers[idx]) // 2
	return contig_key, mid_pos


def sample_from_bed_labels(labels, label_dict, contig_key_prefix=''):
	while True:
		contig_key = contig_key_prefix + str(np.random.randint(1,20))
		label = random.choice(labels)
		if contig_key in label_dict[label]:
			break

	lowers = label_dict[label][contig_key][0]
	uppers = label_dict[label][contig_key][1]
	label2 = label_dict[label][contig_key][2]
	label3 = label_dict[label][contig_key][3]

	idx = np.random.randint(len(lowers))
	mid_pos = (lowers[idx] + uppers[idx]) // 2
	
	return contig_key, mid_pos, label2[idx], label3[idx]


def sample_from_vcf(record_dict, vcf_reader):
	variant_window = 5000
	c_idx = np.random.randint(1,20)
	contig = record_dict[str(c_idx)]
	p_idx = np.random.randint(len(contig)-variant_window)
	return vcf_reader.fetch(str(c_idx), p_idx, p_idx + variant_window)


def intersect_vcfs(vcf1, vcf2):
	shared_variants = {}

	vcf_reader = vcf.Reader(open(vcf1, 'r'))
	vcf_ram =  vcf.Reader(open(vcf2, 'r'))


	for variant in vcf_reader:
		idx_offset = (args.window_size//2)
		start = variant.POS-idx_offset
		end = variant.POS+idx_offset

		if variant_in_vcf(variant, vcf_ram):
			shared_variants[v2.CHROM + str(v2.POS)] = variant  # This returns just 1 of the variants per site

	return shared_variants


def variant_in_vcf(variant, vcf_ram):
	''' Check if variant is in a VCF file.

	Arguments
		variant: the variant we are looking for
		vcf_ram: the VCF we look in, must have an index (tbi, or idx)

	Returns
		variant if it is found otherwise None
	'''	
	start = variant.POS-1
	end = variant.POS

	variants = vcf_ram.fetch(variant.CHROM, start, end)
	
	for v in variants:
		same_allele = any([a1 == a2 for a1 in v.ALT for a2 in variant.ALT]) 
		if v.CHROM == variant.CHROM and v.POS == variant.POS and same_allele:
			return v
	
	return None 


def allele_in_vcf(allele, variant, vcf_ram):
	''' Check if variant's allele is in a VCF file.

	Arguments
		allele: the allele from the provided variant that we are checking
		variant: the variant whose allele we are looking for
		vcf_ram: the VCF we look in, must have an index (tbi, or idx)

	Returns
		variant if it is found otherwise None
	'''	
	variants = vcf_ram.fetch(variant.CHROM, variant.POS-1, variant.POS)

	for v in variants:
		if v.CHROM == variant.CHROM and v.POS == variant.POS and allele in v.ALT:
			return v
	
	return None 


def interval_file_to_dict(interval_file, shift1=0, skip=['@']):
	''' Create a dict to store intervals from a interval list file.

	Arguments:
		interval_file: the file to load either a bed file -> shift1 should be 1
			or a picard style interval_list file -> shift1 should be 0
		shift1: Shift the intervals 1 position over to align with 1-indexed VCFs
		skip: Comment character to ignore
	Returns:
		intervals: dict where keys in the dict are contig ids
			values are a tuple of arrays the first array 
			in the tuple contains the start positions
			the second array contains the end positions.
	'''
	intervals = {}

	with open(interval_file)as f:
		for line in f:
			if line[0] in skip:
				continue

			parts = line.split()
			contig = parts[0]
			lower = int(parts[1])+shift1
			upper = int(parts[2])+shift1

			if contig not in intervals:
				intervals[contig] = ([], [])

			intervals[contig][0].append(lower)
			intervals[contig][1].append(upper)

	for k in intervals.keys():
		intervals[k] = (np.array(intervals[k][0]), np.array(intervals[k][1]))		

	return intervals


def bed_file_to_dict(bed_file, shift1=1):
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
			lower = int(parts[1])+shift1
			upper = int(parts[2])+shift1

			if contig not in bed:
				bed[contig] = ([], [])

			bed[contig][0].append(lower)
			bed[contig][1].append(upper)

	for k in bed.keys():
		bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]))		

	return bed


def bed_file_labels_to_dict(bed_file):
	bed = {}

	with open(bed_file)as f:
		for line in f:
			parts = line.split()
			contig = parts[0]
			lower = int(parts[1])
			upper = int(parts[2])
			label = parts[3]

			if contig not in bed:
				bed[contig] = ([], [], [])

			bed[contig][0].append(lower)
			bed[contig][1].append(upper)
			bed[contig][2].append(label)

	for k in bed:
		bed[k] = (np.array(bed[k][0]), np.array(bed[k][1]), bed[k][2])		

	return bed


def bed_file_sum_bases(bed_dict):
	total = 0
	for k in bed_dict.keys():
		sums = bed_dict[k][1] - bed_dict[k][0]
		total += np.sum(sums)
	return total


def in_bed_file(bed_dict, contig, pos):
	# Exclusive
	lows = bed_dict[contig][0]
	ups = bed_dict[contig][1]
	return np.any((lows <= pos) & (pos < ups))


def bed_file_label(bed_dict, contig, pos, label_i=2):

	if in_bed_file(bed_dict, contig, pos):
		lows = bed_dict[contig][0]
		ups = bed_dict[contig][1]

		i = np.argmax((lows < pos) & (pos < ups))
		label = bed_dict[contig][label_i][i]

		return label


def plain_name(full_name):
	name = os.path.basename(full_name)
	return name.split('.')[0]


def is_insertion(variant):
	return any( map(lambda x: x and len(x) > len(variant.REF), variant.ALT) )


def is_deletion(variant):
	return any( map(lambda x: x and len(x) < len(variant.REF), variant.ALT) )

# Back to the top!
if "__main__" == __name__:
	run_training_data()
