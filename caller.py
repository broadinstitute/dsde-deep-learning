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
	infer_vcf(args)


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

	## LOAD a model
	tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
	variant_batch = []
	for k in intervals:
		contig = reference[k]
		for start,stop in zip(intervals[k][0], intervals[k][1]):
			cur_pos = start
			for cur_pos in range(start, stop, args.window_size):			
				
				record = contig[cur_pos: cur_pos+args.window_size]
				tensor_batch[stats['cur_tensor']] = td.make_calling_tensor(args, samfile, record, cur_pos, stats)
				variant_batch.append((k, cur_pos, record))
				stats['cur_tensor'] += 1

				if stats['cur_tensor'] == args.batch_size:
					## Evaluate the model
					predictions = model.predict(tensor_batch) # predictions is a numpy arra
					tensor_batch = np.zeros((args.batch_size,)+defines.tensor_shape_from_args(args))
					variant_batch = []



def predictions_to_variants(args, predictions, variant_batch, tensor_batch, vcf_writer):
	index2labels = {v:k for k,v in defines.calling_labels.items()}
	for i,gpos in enumerate(variant_batch):
		guess = np.argmax(predictions[i])
		for j in guess.shape[0]:
			if index2labels[guess[j]] == 'HET_SNP':
				vcf.model._Record(gpos[0], gpos[1], '.', record[j], alts, v.qual, v.filter, v.info, [], None)
			elif index2labels[guess[j]] == 'HOM_SNP':
				pass	
			elif index2labels[guess[j]] == 'HET_DELETION':
				pass	
			elif index2labels[guess[j]] == 'HOM_DELETION':
				pass		
			elif index2labels[guess[j]] == 'HOM_INSERTION':
				pass	
			elif index2labels[guess[j]] == 'HET_INSERTION':
				pass



if __name__=='__main__':
	run()