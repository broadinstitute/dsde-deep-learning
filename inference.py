#!/usr/bin/env python
# inference.py
#
# Load pre-trained model from HD5 files and apply to command line supplied tensors.
#
# September 2017
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
import h5py
import time
import pysam
import models
import defines
import operator
import arguments
import numpy as np
import training_data as td
from Bio import Seq, SeqIO
from collections import Counter, defaultdict


def run():
	args = arguments.parse_args()
	annotate_vcf_with_inference(args)


def annotate_vcf_with_inference(args):
	cnns = {}
	stats = Counter()
	vcf_reader = pysam.VariantFile(args.negative_vcf, "r")

	for a in args.architectures:	
		print('Annotating with architecture:', a)		
		cnns[a] = models.set_args_and_get_model_from_semantics(args, a)
		vcf_reader.header.info.add(score_key_from_json(a), '1', 'Float', 'Site-level score from Convolutional Neural Net named '+a+'.')

	vcf_writer = pysam.VariantFile(args.output_vcf, 'w', header=vcf_reader.header)
	print('got vcfs.')

	reference = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	print('got ref.')

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	print('got sam.')

	tensor_channel_map = defines.get_tensor_channel_map_from_args(args)
	if args.channels_last:
		tensor_shape = (args.read_limit, args.window_size, len(tensor_channel_map))
	else:
		tensor_shape = (len(tensor_channel_map), args.read_limit, args.window_size) 

	print('iterate over vcf')

	positions = []
	variant_batch = []
	time_batch = time.time()
	tensor_batch = np.zeros(((args.batch_size,) + tensor_shape))
	annotation_batch = np.zeros((args.batch_size, len(args.annotations)))
		
	if args.chrom:
		variants = vcf_reader.fetch(args.chrom, args.start_pos, args.end_pos)
	else:
		variants = vcf_reader

	for variant in variants:
		idx_offset, ref_start, ref_end = get_variant_window(args, variant)

		contig = reference[variant.contig]	
		record = contig[ ref_start : ref_end ]
		v = pysam_variant_to_pyvcf(variant)
		
		annotation_data = td.get_annotation_data(args, v, stats)
		annotation_batch[stats['annotations_in_batch']] = annotation_data
		stats['annotations_in_batch'] += 1

		read_tensor = td.make_reference_and_reads_tensor(args, v, samfile, record.seq, ref_start, stats)
		if read_tensor is None:
			continue

		tensor_batch[stats['read_tensors_in_batch']] = read_tensor
		stats['read_tensors_in_batch'] += 1

		positions.append(variant.contig + '_' + str(variant.pos))
		variant_batch.append(variant)

		if stats['read_tensors_in_batch'] == args.batch_size or stats['annotations_in_batch'] == args.batch_size:
			batch = [tensor_batch, annotation_batch]
			apply_cnns_to_batch(args, cnns, batch, positions, variant_batch, vcf_writer, stats)
			
			# Reset the batch
			positions = []		
			variant_batch = []
			tensor_batch = np.zeros(((args.batch_size,)+tensor_shape))
			annotation_batch = np.zeros((args.batch_size, len(args.annotations)))		
			stats['read_tensors_in_batch'] = stats['annotations_in_batch'] = 0
			stats['batches processed'] += 1
			if stats['batches processed'] % 10 == 0:
				print('Processed:', stats['batches processed'], 'batches.  Last variant:', variant)

		if stats['batches processed']*args.batch_size > args.samples:
			break

	if stats['read_tensors_in_batch'] > 0 or stats['annotations_in_batch'] > 0:
		batch = [tensor_batch, annotation_batch]
		apply_cnns_to_batch(args, cnns, batch, positions, variant_batch, vcf_writer, stats)

	for s in stats.keys():
		print(s, 'has:', stats[s])	


def pysam_variant_to_pyvcf(v):
	alts = [vcf.model._Substitution(a) for a in v.alts]
	return vcf.model._Record(v.contig, v.pos, v.id, v.ref, alts, v.qual, v.filter, v.info, [], None)

def score_key_from_json(json_file):
	return td.plain_name(json_file).upper() 

def is_snp(variant):  # SNP means ref and alt both are length 1
	return len(variant.ref) == 1 and any(map(lambda x: len(x) == 1, variant.alts))

def get_variant_window(args, variant):
	index_offset = (args.window_size//2)
	reference_start = variant.pos-(index_offset+1)
	reference_end = variant.pos+index_offset

	return index_offset, reference_start, reference_end


def apply_cnns_to_batch(args, cnns, input_data, positions, variant_batch, vcf_writer, stats):
	snp_dicts = {}
	indel_dicts = {}
	predictions = {}
	for a in cnns:
		predictions[a] = cnns[a].predict(input_data, batch_size=args.batch_size)
		snp_dicts[a] = models.predictions_to_snp_scores(args, predictions[a], positions)
		indel_dicts[a] = models.predictions_to_indel_scores(args, predictions[a], positions)

	# loop over the batch of variants and write them out with a score
	for v_out in variant_batch:
		position = v_out.contig + '_' + str(v_out.pos)
		
		for a in cnns:
			if is_snp(v_out):
				v_out.info[score_key_from_json(a)] = float(snp_dicts[a][position])
			elif is_indel(v_out):
				v_out.info[score_key_from_json(a)] = float(indel_dicts[a][position])
			else:
				stats['Not SNP or INDEL'] += 1
				break

		vcf_writer.write(v_out)
		stats['variants_written'] += 1



def load_tensors_and_annotations_from_class_dirs(args):
	tensors = []
	positions = []
	annotations = []

	for tp in os.listdir(args.tensors):

		fn, file_extension = os.path.splitext(tp)
		if not file_extension.lower() in tensor_exts:
			continue

		gpos = tp.split('-')[2]
		chrom = gpos.split('_')[0]
		pos = os.path.splitext(gpos.split('_')[1])[0]

		positions.append(chrom + '_' + pos)

		try:
			with h5py.File(args.tensors+'/'+tp, 'r') as hf:
				tensors.append(np.array(hf.get('read_tensor')))
				annotations.append(np.array(hf.get('annotations')))
		except ValueError as e:
			print(str(e), '\nValue error at:', tp)

	return (np.asarray(tensors), np.asarray(annotations), np.asarray(positions))



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


def is_snp(variant):
	return len(variant.ref) == 1 and any(map(lambda x: len(x) == 1, variant.alts))

def is_indel(variant):
	return len(variant.ref) > 1 or any(map(lambda x: len(x) > 1, variant.alts))

if __name__=='__main__':
	run()

