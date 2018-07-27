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
	else:
		raise ValueError('Unknown calling mode:', args.mode)		


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
	print('Loaded reference FASTA:', args.reference_fasta)

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	print('got sam.')	

	if args.chrom:
		intervals = { args.chrom : [int(args.start_pos), int(args.end_pos)] }
	elif args.bed_file:
		intervals = td.bed_file_to_dict(args.bed_file)
	else:
		raise ValueError('What do you want to iterate over? Use arguments --bed_file or --chrom --start_pos --end_pos')

	tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
	gpos_batch = []

	print(len(intervals), 'intervals to iterate over, contigs:', intervals.keys())
	start_time = time.time()
	for k in intervals:
		contig = reference[k]
		args.chrom = k
		for start,stop in zip(intervals[k][0], intervals[k][1]):
			cur_pos = start
			for cur_pos in range(start, stop, args.window_size):		
				record = contig[cur_pos: cur_pos+args.window_size]
				t = td.make_calling_tensor(args, samfile, record, cur_pos, stats)
				
				if not t is None:
					tensor_batch[stats['cur_tensor']] = t	
					gpos_batch.append((k, cur_pos, record))
					stats['cur_tensor'] += 1

				if stats['cur_tensor'] == args.batch_size:
					predictions = model.predict(tensor_batch) # predictions is a numpy arra
					predictions_to_variants(args, predictions, gpos_batch, tensor_batch, vcf_writer, record)
					tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
					stats['cur_tensor'] = 0
					stats['batches_processed'] += 1
					gpos_batch = []

					if stats['batches_processed'] % 100 == 0:
						elapsed = time.time() - start_time
						v_per_minute = stats['batches processed']*args.batch_size / (elapsed/60)
						print('At genomic position:', cur_pos, 'Tensors per minute:', ,'Batches processed:', stats['batches_processed'])
						for s in stats.keys():
							print(s, 'has:', stats[s])	
	
	for s in stats.keys():
		print(s, 'has:', stats[s])	


def predictions_to_variants(args, predictions, gpos_batch, tensor_batch, vcf_writer, record=None):
	index2labels = {v:k for k,v in defines.calling_labels.items()}
	indel_start = -1
	ref_offset = 0
	for i,gpos in enumerate(gpos_batch):
		guess = np.argmax(predictions[i], axis=1)
		cur_tensor = tensor_batch[i]

		for j in range(guess.shape[0]):
			if index2labels[guess[j]] == 'REFERENCE':
				continue

			ref_start = int(gpos[1])-ref_offset
			# Does NOT properly handle multiallelics
			if record and j < len(record):
				ref_allele = record[j]
			else:
				ref_allele = reference_base_from_tensor(args, cur_tensor, j)
			alt = strongest_alt_allele_from_tensor(args, cur_tensor, j, ref_allele)

			is_indel = 'DELETION' in index2labels[guess[j]] or 'INSERTION' in index2labels[guess[j]]
			if is_indel and indel_start == -1:
				indel_start = j	
			
			if index2labels[guess[j]] == 'HET_SNP':
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=ref_start+j,
									  stop=ref_start+j+1,
									  alleles=[ref_allele, alt],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
			elif index2labels[guess[j]] == 'HOM_SNP':
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=ref_start+j,
									  stop=ref_start+j+1,
									  alleles=[ref_allele, alt],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
			elif index2labels[guess[j]] == 'HET_DELETION' and variant_edge(index2labels, guess, j):
				d = get_deleted(args, cur_tensor, indel_start, j)
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=ref_start+indel_start-1,
									  stop=ref_start+indel_start+(j-indel_start)+2,
									  alleles=[d, d[0]],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
				indel_start = -1
			elif index2labels[guess[j]] == 'HOM_DELETION' and variant_edge(index2labels, guess, j):
				d = get_deleted(args, cur_tensor, indel_start, j)
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=ref_start+indel_start-1,
									  stop=ref_start+indel_start+(j-indel_start)+2,
									  alleles=[d, d[0]],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
				indel_start = -1
			elif index2labels[guess[j]] == 'HOM_INSERTION' and variant_edge(index2labels, guess, j):
				if record and ref_offset+(indel_start-1) < len(record):
					ref = record[ref_offset+(indel_start-1)]
				else:
					ref = reference_base_from_tensor(args, cur_tensor, j)
					if ref == defines.indel_char:
						print('Looked for reference but found only insertions at:', (ref_offset+(indel_start-1)))
						#if (ref_offset+(indel_start-1))  > 0:
						#	print('at t-1 we have:', reference_base_from_tensor(args, cur_tensor, ref_offset+(indel_start-2)))

				insert = get_inserted(args, cur_tensor, indel_start, j)
				v = vcf_writer.new_record(contig=gpos[0], 
									  start=ref_start+indel_start,
									  stop=ref_start+indel_start+1,
									  alleles=[ref, insert],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
				ref_offset += j-indel_start
				indel_start = -1	
			elif index2labels[guess[j]] == 'HET_INSERTION' and variant_edge(index2labels, guess, j):
				if record and ref_offset+(indel_start-1) < len(record):
					ref = record[ref_offset+(indel_start-1)]
				else:
					ref = reference_base_from_tensor(args, cur_tensor, j)
					if ref == defines.indel_char:
						print('Looked for reference but found only insertions at:', (ref_offset+(indel_start-1)))
						#if (ref_offset+(indel_start-1)) > 0:
						#	print('at t-1 we have:', reference_base_from_tensor(args, cur_tensor, ref_offset+(indel_start-2)))

				insert = get_inserted(args, cur_tensor, indel_start, j)
				v = vcf_writer.new_record(contig=gpos[0],
									  start=ref_start+indel_start,
									  stop=ref_start+indel_start+1,
									  alleles=[ref, insert],
									  qual=predictions[i][j][guess[j]])
				vcf_writer.write(v)
				ref_offset += j-indel_start
				indel_start = -1


def variant_edge(index2labels, guess, j):
	return (j == guess.shape[0]-1) or (index2labels[guess[j]] != index2labels[guess[j+1]])


def get_deleted(args, tensor, indel_start, j):			
	return ''.join([reference_base_from_tensor(args, tensor, deleted_i) for deleted_i in range(indel_start-1, j+1)])


def get_inserted(args, tensor, indel_start, j):			
	return ''.join([strongest_allele_from_tensor(args, tensor, tensor_site) for tensor_site in range(indel_start-1, j+1)])


def reference_base_from_tensor(args, tensor, tensor_site):
	channels = defines.get_tensor_channel_map_from_args(args)
	for c in channels:
		if c[-1] != defines.indel_char and 'reference' in c:
			if args.channels_last and tensor[0, tensor_site, channels[c]] > 0:
				return c[-1].upper() # reference channels are strings like reference_A or reference_C
				# Here we want just the nucleic acid. 
			elif not args.channels_last and tensor[channels[c], 0, tensor_site] > 0:
				return c[-1].upper()


	return defines.indel_char # No evidence of reference, insertion perhaps 


def strongest_allele_from_tensor(args, tensor, tensor_site):
	channels = defines.get_tensor_channel_map_from_args(args)
	max_count = -1
	strongest_allele = 'N'
	for c in channels:
		if c[-1] != defines.indel_char and 'read' in c:
			if args.channels_last:
				cur_count = np.sum(tensor[:, tensor_site, channels[c]])
			else:
				cur_count = np.sum(tensor[channels[c], :, tensor_site])
			
			if cur_count > max_count:
				max_count = cur_count
				strongest_allele = c[-1].upper()
	
	return strongest_allele


def strongest_alt_allele_from_tensor(args, tensor, tensor_site, ref_allele):
	channels = defines.get_tensor_channel_map_from_args(args)
	max_count = -1
	strongest_allele = 'N'
	
	for c in channels:
		if c[-1] != defines.indel_char and 'read' in c:
			if args.channels_last:
				cur_count = np.sum(tensor[:, tensor_site, channels[c]])
			else:
				cur_count = np.sum(tensor[channels[c], :, tensor_site])
			
			if cur_count > max_count and c[-1].upper() != ref_allele:
				max_count = cur_count
				strongest_allele = c[-1].upper()
	
	return strongest_allele


if __name__=='__main__':
	run()