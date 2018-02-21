#!/usr/bin/env python
# unit_tests.py
#
# Units tests for Variant Filtration.
# Test tensor generation, model building and training.
#
# May 2017
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
import pysam
import plots
import pickle
import models
import defines
import unittest
import arguments
import numpy as np
import training_data as td

from Bio import Seq, SeqIO
from collections import Counter


def run_tests():
	suite = unittest.TestLoader().loadTestsFromTestCase(TestModels)
	unittest.TextTestRunner(verbosity=2).run(suite)	
	suite = unittest.TestLoader().loadTestsFromTestCase(TestVariants)
	unittest.TextTestRunner(verbosity=2).run(suite)


class TestVariants(unittest.TestCase):

	def setUp(self):
		self.record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))
		self.vcf_ram = vcf.Reader(open(args.negative_vcf, 'r'))
		self.vcf_train = vcf.Reader(open(args.train_vcf, 'r'))
		self.bed_dict = td.bed_file_to_dict(args.bed_file)	

	def test_vcf_and_bed_lookup(self):
		v1 = self.vcf_train.next()
		v2 = self.vcf_ram.next()

		self.assertTrue(td.in_bed_file(self.bed_dict, v1.CHROM, v1.POS))
		self.assertFalse(td.in_bed_file(self.bed_dict, v2.CHROM, v2.POS))
		self.assertFalse(td.variant_in_vcf(v2, self.vcf_train))
		self.assertTrue(td.variant_in_vcf(v1, self.vcf_train))
		self.assertTrue(td.variant_in_vcf(v2, self.vcf_ram))

	def test_vcf_and_reference(self):
		self.check_vcf_and_reference(self.vcf_ram)
		self.check_vcf_and_reference(self.vcf_train)		

	def test_vcf_truth_on_platinum_pinhole_bed(self):
		if 'platinum' in args.bed_file:
			chrom = '1'
			vpos = 142560386

			variants = self.vcf_train.fetch(chrom, vpos-2, vpos+2)
			for v in variants:
				if v.POS == vpos:
					print('Got variant at %d' % vpos)
					print(v)
					self.assertTrue(td.variant_in_vcf(v, self.vcf_train))
					self.assertFalse(td.in_bed_file(self.bed_dict, v.CHROM, v.POS))

	def check_vcf_and_reference(self, my_vcf, max_samples=1000):
		count = 0
		for v in my_vcf:
			contig = self.record_dict[v.CHROM]	
			self.assertTrue(td.variant_in_vcf(v, my_vcf))
			self.assertEquals(v.REF[0], contig[v.POS-1])
			count += 1
			if count > max_samples:
				break


class TestModels(unittest.TestCase):
	
	def setUp(self):
		in_channels = defines.total_input_channels_from_args(args)
		if args.channels_last:
			self.tensor_shape = (args.read_limit, args.window_size, in_channels)
		else:
			self.tensor_shape = (in_channels, args.read_limit, args.window_size)

	def test_baseline_2d(self):
		m = models.build_read_tensor_2d_and_annotations_model(args)
		self.assertEquals(m.output_shape[1], len(args.labels))
		self.assertEquals(m.input_shape[0][1:], self.tensor_shape)
		self.assertEquals(m.input_shape[1][1], len(args.annotations))


# Back to the top!
if '__main__' == __name__:
	args = arguments.parse_args()
	run_tests()

