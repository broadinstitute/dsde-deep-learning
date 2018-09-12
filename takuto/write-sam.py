import numpy as np
import keras
import matplotlib.pyplot as plt
import pysam
import argparse
import h5py
from array import array

from keras.models import load_model

def read_passes_qc(read):
    ''' Given a read decide whether it's a good read '''
    read_group = read.get_tag('RG')
    if 'artificial' in read_group.lower():
        return False
    elif not read.is_proper_pair or not read.is_paired:
        return False
    elif read.is_duplicate or read.is_secondary or read.is_supplementary or read.is_qcfail or read.is_unmapped:
        return False

    return True

def write_bam(args):
    sam = args.input
    input_type = args.input_type
    reference = args.reference
    read_length = args.read_length
    output_path = args.output
    model = args.model

    io_read_flag = "rc" if input_type == "cram" else "rb"
    io_write_flag = "wc" if input_type == "cram" else "wb"

    samfile = pysam.AlignmentFile(sam, io_read_flag, reference_filename=reference)

    # compress and write a new sam file
    # new_samfile_path = "/dsde/data/deep/takutoencoder/output/NA12878.cram"
    new_samfile = pysam.AlignmentFile(output_path, "wc", template=samfile, reference_filename=reference)

    # model_path = "/Users/tsato/workspace/dsde-deep-learning/takutoencoder/vanilla.h5"
    autoencoder = load_model(model)

    i = 0
    for read in samfile.fetch('chr22'):
        if read_passes_qc(read):
            orig_bqs = np.reshape(np.array(read.query_qualities), (1, read_length))
            new_bqs = autoencoder.predict(orig_bqs).reshape(read_length).astype(int).clip(min=2)
            read.query_qualities = array('b', new_bqs)
            i += 1

        new_samfile.write(read)

        if args.debug and i > 10:
            break

def parse_args():
    #### Filepaths
    default_model = "/Users/tsato/workspace/dsde-deep-learning/takutoencoder/vanilla.h5"
    grch38 = "/seq/references/Homo_sapiens_assembly38/v0/Homo_sapiens_assembly38.fasta"
    # cram = "/dsde/data/datasets/SnapShotExperiment2015/CEUTrio/Alignments/G94982.NA12878/NA12878.cram"

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default=default_model)
    parser.add_argument('--reference', default = grch38)
    parser.add_argument('--input')
    parser.add_argument('--input_type', default="bam")
    parser.add_argument('--read_length', type=int, default=151)
    parser.add_argument('--output')
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()
    print('Arguments are', args)
    return args

if __name__ == '__main__':
    args = parse_args()
    write_bam(args)