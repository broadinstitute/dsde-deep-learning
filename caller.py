# caller.py

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
import inference as inf
from Bio import Seq, SeqIO
from collections import Counter, defaultdict


def run():
	args = arguments.parse_args()
	args.labels = defines.calling_labels
	if args.mode == 'tensors':
		infer_tensor(args)
	elif args.mode == 'vcf':
		infer_vcf(args)


def infer_tensor(args):
	stats = Counter()
	model = models.load_model(args.weights_hd5, custom_objects=models.get_all_custom_objects(args.labels))
	vcf_reader = pysam.VariantFile(args.negative_vcf, 'r')
	vcf_writer = pysam.VariantFile(args.output_vcf, 'w', header=vcf_reader.header)
	print('got vcfs.')
	
	tensor_paths = [args.data_dir+tp for tp in sorted(os.listdir(args.data_dir))]
	print('found tensors: ', len(tensor_paths))
	tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
	gpos_batch = []

	for tp in tensor_paths:
		with h5py.File(tp, 'r') as hf:
			tensor_batch[stats['cur_tensor']] = np.array(hf.get(args.tensor_map))
			gpos_batch.append(td.position_string_from_tensor_name(tp).split('_'))
			stats['cur_tensor'] += 1
			if stats['cur_tensor'] == args.batch_size:
				## Evaluate the model
				predictions = model.predict(tensor_batch) # predictions is a numpy arra	
				predictions_to_variants(args, predictions, gpos_batch, tensor_batch, vcf_writer)
				stats['cur_tensor'] = 0
				gpos_batch = []


def infer_vcf(args):
	stats = Counter()
	model = models.load_model(args.weights_hd5, custom_objects=models.get_all_custom_objects(args.labels))

	vcf_reader = pysam.VariantFile(args.negative_vcf, 'r')
	vcf_writer = pysam.VariantFile(args.output_vcf, 'w', header=vcf_reader.header)
	print('got vcfs.')

	reference = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	print('got ref.')

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	print('got sam.')	

	intervals = td.bed_file_to_dict(args.bed_file)
	print('got intervals.')

	tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
	gpos_batch = []
	for k in intervals:
		contig = reference[k]
		for start,stop in zip(intervals[k][0], intervals[k][1]):
			cur_pos = start
			for cur_pos in range(start, stop, args.window_size):		
				
				record = contig[cur_pos: cur_pos+args.window_size]
				print(str(record))
				tensor_batch[stats['cur_tensor']] = td.make_calling_tensor(args, samfile, record, cur_pos, stats)				
				gpos_batch.append((k, cur_pos, record))
				stats['cur_tensor'] += 1

				if stats['cur_tensor'] == args.batch_size:
					## Evaluate the model
					predictions = model.predict(tensor_batch) # predictions is a numpy arra
					predictions_to_variants(args, predictions, gpos_batch, tensor_batch, vcf_writer)
					tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
					stats['cur_tensor'] = 0
					gpos_batch = []


def predictions_to_variants(args, predictions, gpos_batch, tensor_batch, vcf_writer):
	index2labels = {v:k for k,v in defines.calling_labels.items()}
	for i,gpos in enumerate(gpos_batch):
		guess = np.argmax(predictions[i], axis=1)
		cur_tensor = tensor_batch[i]
		print('Prediction shape:', predictions.shape)
		print('Guess:', guess)
		print('gpos:', gpos)
		for j in range(guess.shape[0]):
			gpos_int = int(gpos[1])
			ref_snp = reference_snp_allele_from_tensor(args, cur_tensor, j)
			alt_snps = alt_snp_allele_from_tensor(args, cur_tensor, j)
			alt = alt_snps[0] if alt_snps[1] == ref_snp else alt_snps[1]
			if index2labels[guess[j]] == 'HET_SNP':
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=gpos_int+j,
									  alleles=[ref_snp, alt],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
			elif index2labels[guess[j]] == 'HOM_SNP':
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=gpos_int+j,
									  alleles=[ref_snp, alt],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
			elif index2labels[guess[j]] == 'HET_DELETION':
				pass	
			elif index2labels[guess[j]] == 'HOM_DELETION':
				pass		
			elif index2labels[guess[j]] == 'HOM_INSERTION':
				pass	
			elif index2labels[guess[j]] == 'HET_INSERTION':
				pass


def reference_snp_allele_from_tensor(args, tensor, gpos):
	channels = defines.get_tensor_channel_map_from_args(args)
	for c in channels:
		if 'reference' in c:
			if args.channels_last and tensor[ 0, gpos, channels[c]] > 0:
				return c[-1].upper() # reference channels are strings like reference_A or reference_C
				# Here we want just the nucleic acid. 
			elif tensor[channels[c], 0, gpos] > 0:
				return c[-1].upper()


def alt_snp_allele_from_tensor(args, tensor, gpos):
	channels = defines.get_tensor_channel_map_from_args(args)
	counts = {}
	for c in channels:
		if 'read' in c:
			if args.channels_last:
				counts[c[-1].upper()] = np.sum(tensor[:, gpos, channels[c]])
			else:
				counts[c[-1].upper()] = np.sum(tensor[channels[c], :, gpos])
	
	counts = sorted(counts.items(), key=operator.itemgetter(1))
	return counts[-1][0], counts[-2][0]



if __name__=='__main__':
	run()