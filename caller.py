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
from Bio import Seq, SeqIO
from collections import Counter, defaultdict


def run():
	args = arguments.parse_args()
	infer_vcf(args)


def infer_vcf(args):
	stats = Counter()

	vcf_reader = pysam.VariantFile(args.negative_vcf, 'r')
	vcf_writer = pysam.VariantFile(args.output_vcf, 'w', header=vcf_reader.header)
	print('got vcfs.')

	reference = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
	print('got ref.')

	samfile = pysam.AlignmentFile(args.bam_file, "rb")	
	print('got sam.')	

	intervals = bed_file_to_dict(args.bed_file)
	print('got intervals.')

	## LOAD a model
	model = models.load_model(args.weights_hd5)

	for k in intervals:
		for start,stop in zip(intervals[k][0], intervals[k][1]):
			cur_pos = start
			for cur_pos in range(start, stop, args.window_size):
				
				contig = reference[variant.contig]
				record = contig[cur_pos: cur_pos+args.window_size]
				read_tensor = td.make_calling_tensor(args, samfile, record, cur_pos, stats)

				## Evaluate the model
				## From model predictions make a Variant
				## write variant to output vcf





if __name__=='__main__':
	run()


