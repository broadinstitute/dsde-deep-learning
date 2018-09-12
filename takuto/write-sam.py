import numpy as np
import keras
import matplotlib.pyplot as plt
import pysam
import argparse
import h5py
from array import array

from keras.models import load_model


#### Helper Functions

def qc_read(read):
    ''' Given a read decide whether it's a good read '''
    read_group = read.get_tag('RG')
    if 'artificial' in read_group.lower():
        return False
    elif not read.is_proper_pair or not read.is_paired:
        return False
    elif read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_qcfail or read.is_unmapped:
        return False

    return True

def write_bam():
    #### Filepaths
    grch38 = "/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta"
    cram = "/dsde/data/datasets/SnapShotExperiment2015/CEUTrio/Alignments/G94982.NA12878/NA12878.cram"
    samfile = pysam.AlignmentFile(cram, "rc", reference_filename=grch38)  # "rb" for bam, "rc" for cram

    # compress and write a new sam file
    new_samfile_path = "/dsde/data/deep/takutoencoder/output/NA12878.cram"
    new_samfile = pysam.AlignmentFile(new_samfile_path, "wc", template=samfile, reference_filename=grch38)

    model_path = "/Users/tsato/workspace/dsde-deep-learning/takutoencoder/vanilla.h5"
    autoencoder = load_model(model_path)

    read_length = 151

    i = 0
    for read in samfile.fetch('chr22'):
        if qc_read(read):
            orig_bqs = np.reshape(np.array(read.query_qualities), (1, read_length))
            new_bqs = autoencoder.predict(orig_bqs).reshape(read_length).astype(int).clip(min=2)
            read.query_qualities = array('b', new_bqs)
            i += 1
        new_samfile.write(read)

if __name__ == '__main__':
	write_bam()