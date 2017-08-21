# DSDE Deep Learning
#
# Train 1D Convolutional Neural Network from exome bait
# Input is a small window of reference sequence, a bait, 
#
# Output is the normalized coverage of the bait
#
# July 2017
# Yossi Farjoun
# farjoun@broadinstitute.org

from __future__ import print_function
from __future__ import division

import argparse
import math
import os

import matplotlib
import numpy as np

matplotlib.use('Agg')
from keras import metrics
import keras.backend as K
from Bio import SeqIO
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.initializers import RandomNormal
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Input, Dense, Dropout, Flatten, Activation
from keras.layers.merge import Concatenate
# import tensorflow as tf

data_path = '/Users/farjoun/exomeData/'
reference_fasta = data_path + 'Homo_sapiens_assembly19.fasta'
bait_bed_file = data_path + 'coverage.singleton.bait.bed'


def run():
    args = parse_args()

    if 'small' == args.model:
        make_small_model(args)
    elif 'large' == args.model:
        make_large_model(args)
    elif 'plot' == args.model:
        make_plots(args)
    else:
        print('Unknown model argument')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='small')
    parser.add_argument('--window_size', default=500, type=int)
    parser.add_argument('--samples', default=10000, type=int)
    parser.add_argument('--reference_fasta', default=reference_fasta)
    parser.add_argument('--bed_file', default=bait_bed_file)
    parser.add_argument('--inputs', default={'A': 0, 'C': 1, 'T': 2, 'G': 3, 'Bait': 4})
    parser.add_argument('--annotations', default=('%gc',))
    parser.add_argument('--weights')

    args = parser.parse_args()
    print('Arguments are', args)
    return args


def make_small_model(args):
    model = build_small_sequential_bait_model(args)

    train_data = load_dna_and_bait_coverage(args)
    train, valid, test = split_data(train_data)

    weight_path = weight_path_from_args(args)
    model = train_bait_model(model, train, valid, weight_path)

    title = weight_path_to_title(weight_path)

    plot_scatter(model, {"test": ([test.baitdata, test.gc], test.coverage),
                         "validation": ([valid.baitdata, valid.gc], valid.coverage),
                         "train": ([train.baitdata, train.gc], train.coverage)}, title)


def make_large_model(args):
    model = build_small_functional_bait_model(args)

    train_data = load_dna_and_bait_coverage(args)
    train, valid, test = split_data(train_data)

    weight_path = weight_path_from_args(args)
    model = train_bait_model(model, train, valid, weight_path)

    title = weight_path_to_title(weight_path)

    plot_scatter(model, {"test": ([test.baitdata, test.gc], test.coverage),
                         "validation": ([valid.baitdata, valid.gc], valid.coverage),
                         "train": ([train.baitdata, train.gc], train.coverage)}, title)


def make_plots(args):
    model = build_small_functional_bait_model(args)
    model.load_weights(args.weights, by_name=True)
    print('Loaded model weights from:', args.weights)

    train_data = load_dna_and_bait_coverage(args)
    train, valid, test = split_data(train_data)

    title = weight_path_to_title(args.weights)

    plot_scatter(model, {"test": ([test.baitdata, test.gc], test.coverage),
                         "validation": ([valid.baitdata, valid.gc], valid.coverage),
                         "train": ([train.baitdata, train.gc], train.coverage)}, title)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def build_small_sequential_bait_model(args):
    model = Sequential()
    model.add(Conv1D(input_dim=len(args.inputs),
                     input_length=args.window_size,
                     filters=40,
                     filter_length=16,
                     border_mode='valid',
                     activation="relu",
                     init='normal'))

    model.add(Dropout(0.3))
    model.add(Conv1D(filters=64, kernel_size=16, activation="relu", init='normal', padding='valid'))
    model.add(Dropout(0.3))
    model.add(MaxPooling1D(pool_size=3, strides=3))
    model.add(Flatten())

    model.add(Dense(units=32, kernel_initializer='normal'))
    model.add(Activation('relu'))

    model.add(Dense(units=1, kernel_initializer=RandomNormal(mean=1.0, stddev=0.5, seed=None)))
    model.add(Activation("relu"))

    adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
    my_metrics = [metrics.mean_squared_error, rmse_log]

    model.compile(loss='mean_squared_error', optimizer=adamo, metrics=my_metrics)
    model.summary()

    return model


def build_small_functional_bait_model(args):
    bait_shape = (args.window_size, len(args.inputs),)
    annotation_shape = (len(args.annotations),)

    print("Bait shape: %s" % (bait_shape,))
    input_baits = Input(name="bait", shape=bait_shape)

    x = Conv1D(activation='relu',
               padding='valid',
               filters=100,
               kernel_size=8,
               kernel_initializer='normal', name='conv1')(input_baits)

    x = Dropout(0.3)(x)

    x = Conv1D(kernel_initializer="normal", activation="relu", padding="valid", filters=64, kernel_size=8)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=3, strides=3)(x)

    x = Conv1D(kernel_initializer="normal", activation="relu", padding="valid", filters=40, kernel_size=8)(x)
    x = Dropout(0.3)(x)
    x = MaxPooling1D(pool_size=3, strides=3)(x)

    x = Flatten()(x)

    x = Dense(units=32, kernel_initializer="normal", activation="relu")(x)

    print("Annotation shape: %s" % (annotation_shape,))

    input_annotations = Input(name="annotation", shape=annotation_shape)

    xy = Concatenate(axis=-1)([x, input_annotations])

    xy = Dense(units=32, kernel_initializer="normal", activation="relu")(xy)

    predictions = Dense(units=1, kernel_initializer=RandomNormal(mean=1.0, stddev=0.5, seed=None), activation="relu")(
        xy)
    my_metrics = [metrics.mean_squared_error, rmse_log, gme]

    # this creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=[input_baits, input_annotations], outputs=predictions)

    # # add some TensorBoard annotations
    # conv1d_1 = filter(lambda y: y.name == "conv1d_1", model.layers)[0]
    # conv1d_1_shape = map(lambda x: x.value, conv1d_1.kernel.get_shape())
    # conv1d_1_shape.insert(0, 1)
    #
    # reshaped = tf.reshape(conv1d_1.kernel, conv1d_1_shape)
    #
    # filters=put_kernels_on_grid(reshaped, 2)
    #
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter("./log/" + '/train')
    #
    adamo = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.)
    model.compile(loss=metrics.mean_squared_error, optimizer=adamo, metrics=my_metrics)
    model.summary()
    return model


def train_bait_model(model, train, valid, save_weight):
    # type: (Model, Data, Data, unicode) -> Model

    checkpointer = ModelCheckpoint(filepath=save_weight, verbose=1, save_best_only=True)
    earlystopper = EarlyStopping(monitor='val_loss', patience=100, verbose=1)
    tensorboard = TensorBoard(log_dir='./logs/' + save_weight, histogram_freq=1, write_graph=False, write_images=False)

    history = model.fit([train.baitdata, train.gc], train.coverage,
                        batch_size=32, epochs=150, shuffle=True,
                        validation_data=([valid.baitdata, valid.gc], valid.coverage),
                        callbacks=[checkpointer, earlystopper, tensorboard])

    plot_metric_history(history, weight_path_to_title(save_weight))

    return model


# data class
class Data:
    def __init__(self, samples, window, depth):
        self.samples = samples  # type: int
        self.window = window  # type: int
        self.depth = depth  # type: int
        self.baitdata = np.zeros((samples, window, depth))  # type: np.array
        self.coverage = np.zeros((samples, 1))  # type: np.array
        self.gc = np.zeros((samples, 1))  # type: np.array

    def __repr__(self):
        return "Data(sample=%r,window=%r,depth=%r,baitdata=%r,coverage=%r,gc=%r)" % (self.samples,
                                                                                     self.window,
                                                                                     self.depth,
                                                                                     self.baitdata,
                                                                                     self.coverage,
                                                                                     self.gc)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training Data ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def load_dna_and_bait_coverage(args, only_labels=None):
    record_dict = SeqIO.to_dict(SeqIO.parse(args.reference_fasta, "fasta"))

    baits_and_coverages = bed_file_to_coverage(args.bed_file)
    print('Loaded baits and annotations from: %s' % (args.bed_file,))

    train_data = Data(args.samples, args.window_size, len(args.inputs))

    idx_offset = (args.window_size // 2)

    amiguity_codes = {'K': [0, 0, 0.5, 0.5],
                      'M': [0.5, 0.5, 0, 0],
                      'R': [0.5, 0, 0, 0.5],
                      'Y': [0, 0.5, 0.5, 0],
                      'S': [0, 0.5, 0, 0.5],
                      'W': [0.5, 0, 0.5, 0],
                      'B': [0, 0.333, 0.333, 0.334],
                      'V': [0.333, 0.333, 0, 0.334],
                      'H': [0.333, 0.333, 0.334, 0],
                      'D': [0.333, 0, 0.333, 0.334],
                      'X': [0.25, 0.25, 0.25, 0.25],
                      'N': [0.25, 0.25, 0.25, 0.25]}

    count = 0
    np.random.seed(0)
    while count < args.samples:
        contig_key, start, end, coverage, gc = sample_from_bed(baits_and_coverages)
        mid = (start + end) // 2
        contig = record_dict[contig_key]

        record = contig[mid - idx_offset: mid + idx_offset]

        train_data.coverage[count, 0] = coverage
        train_data.gc[count] = gc

        for i, b in enumerate(record.seq):
            B = b.upper()
            if B in args.inputs.keys():
                train_data.baitdata[count, i, args.inputs[B]] = 1.0
            elif B in amiguity_codes.keys():
                train_data.baitdata[count, i, 0:4] = amiguity_codes[B]
            else:
                print('Error! Unknown code:', b)
                return

            ref_pos = i + mid - idx_offset
            if start <= ref_pos <= end:
                train_data.baitdata[count, i, 4] = 1

        count += 1

    print('Train data shape: %s' % (train_data.baitdata.shape,))
    print('Coverage data shape: %s' % (train_data.coverage.shape,))

    # should be in the form ( 5xsize x samples tensor, 1x samples tensor)
    return train_data


def bed_file_to_coverage(bed_file):
    bed_with_cov_and_gc = {}

    total_reads = 0L  # type: long
    total_baits = 0L  # type: long

    with open(bed_file) as f:
        for line in f:
            parts = line.split()
            contig = parts[0]
            lower = int(parts[1])
            upper = int(parts[2])
            gc = float(parts[3])
            reads = int(parts[4])
            total_reads += reads
            total_baits += 1

            if contig not in bed_with_cov_and_gc.keys():
                bed_with_cov_and_gc[contig] = ([], [], [], [],)

            bed_with_cov_and_gc[contig][0].append(lower)
            bed_with_cov_and_gc[contig][1].append(upper)
            bed_with_cov_and_gc[contig][2].append(reads)
            bed_with_cov_and_gc[contig][3].append(gc)

    reads_per_bait = float(total_reads) / total_baits
    print("Reads per bait: %s" % reads_per_bait)
    for contig in bed_with_cov_and_gc.keys():
        bed_with_cov_and_gc[contig] = (
            np.array(bed_with_cov_and_gc[contig][0]),
            np.array(bed_with_cov_and_gc[contig][1]),
            np.array([x / reads_per_bait for x in bed_with_cov_and_gc[contig][2]]),
            bed_with_cov_and_gc[contig][3],)
        print('key is: %s len %s' % (contig, len(bed_with_cov_and_gc[contig][0])))

    return bed_with_cov_and_gc


def in_bed_file(bed_dict, contig, pos):
    lows = bed_dict[contig][0]
    ups = bed_dict[contig][1]

    return np.any((lows <= pos) & (pos <= ups))


def split_data(data, valid_ratio=0.1, test_ratio=0.4):
    # type: (Data, float, float) -> (Data, Data, Data)

    np.random.seed(0)

    samples = data.samples
    indices = range(samples)
    np.random.shuffle(indices)

    valid_idx = int(valid_ratio * float(samples))
    test_idx = int(test_ratio * float(samples))

    train = Data(samples - valid_idx - test_idx, data.window, data.depth)
    valid = Data(valid_idx, data.window, data.depth)
    test = Data(test_idx, data.window, data.depth)

    valid.coverage = data.coverage[:valid_idx]
    valid.baitdata = data.baitdata[:valid_idx]
    valid.gc = data.gc[:valid_idx]

    test.coverage = data.coverage[valid_idx:(valid_idx + test_idx)]
    test.baitdata = data.baitdata[valid_idx:(valid_idx + test_idx)]
    test.gc = data.gc[valid_idx:(valid_idx + test_idx)]

    train.coverage = data.coverage[(valid_idx + test_idx):]
    train.baitdata = data.baitdata[(valid_idx + test_idx):]
    train.gc = data.gc[(valid_idx + test_idx):]

    return train, valid, test


def sample_from_bed(bed_dict):
    contig_sizes = {key: len(bed_dict[key]) for key in bed_dict.keys()}
    total_size = sum(contig_sizes.values())

    contig_key = np.random.choice(bed_dict.keys(), 1, p=[x / total_size for x in contig_sizes.values()])[0]

    lowers = bed_dict[contig_key][0]
    uppers = bed_dict[contig_key][1]
    coverage = bed_dict[contig_key][2]
    gcs = bed_dict[contig_key][3]

    idx = np.random.randint(len(lowers))
    return contig_key, lowers[idx], uppers[idx], coverage[idx], gcs[idx]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Metrics ~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def gme(y_true, y_pred):
    """calculates the root (geometric) mean squared error of the values."""
    return K.exp(K.mean(K.log(K.abs(np.divide(y_true + .001, y_pred + .001) - 1))))


def rmse_log(y_true, y_pred):
    """calculates the root mean squared error of the log (base e) of the values."""
    return K.sqrt(K.mean(K.square(K.log(np.divide(y_true + 1, y_pred + 1)))))


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Plots ~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def plot_scatter(model, data_dict, title):
    # type: (Model, dict, str) -> None

    fig, ax = plt.subplots()
    plt.figure(figsize=[4, 4])

    # Compute metrics:
    for key in data_dict.keys():
        y_pred = model.predict(data_dict[key][0], verbose=1)
        y_truth = data_dict[key][1]

        ax.plot(y_truth, y_pred, linestyle='', marker='.', label=key)

    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlabel('True (normalized) Coverage ')
    ax.set_ylabel('Predicted (normalized) Coverage')
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')
    ax.set_title("scatter" + str(title))

    max_xy = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, max_xy], [0, max_xy], ls="--", c=".3")

    fig.savefig("./scatter_" + title + ".jpg")


def plot_history(history, title):
    # list all data in history
    print("History keys: " + history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('Accuracy: ' + title)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss: ' + title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("./plot_history_" + title + ".jpg")


def plot_metric_history(history, title):
    # list all data in history
    print("History keys: " + history.history.keys())

    row = 0
    col = 0
    num_plots = len(history.history) / 2.0  # valid and train plot together
    rows = 4
    cols = int(math.ceil(num_plots / float(rows)))

    f, axes = plt.subplots(rows, cols, sharex=True, figsize=(36, 24))

    if cols > 1:
        for k in history.history.keys():

            if 'val' not in k:
                axes[row, col].plot(history.history[k])
                axes[row, col].plot(history.history['val_' + k])

                axes[row, col].set_ylabel(str(k))
                axes[row, col].legend(['train', 'valid'], loc='upper left')
                axes[row, col].set_xlabel('epoch')

                row += 1
                if row == rows:
                    row = 0
                    col += 1
                    if row * col >= rows * cols:
                        break

        axes[0, 1].set_title(title)
    else:
        for k in history.history.keys():

            if 'val' not in k:
                axes[row].plot(history.history[k])
                axes[row].plot(history.history['val_' + k])

                axes[row].set_ylabel(str(k))
                axes[row].legend(['train', 'valid'], loc='upper left')
                axes[row].set_xlabel('epoch')

                row += 1
        axes[0].set_title(title)

    plt.savefig("./metric_history_" + title + ".jpg")


def weight_path_from_args(args):
    save_weight = './bait_performance_cnn_model'

    ignore = ['inputs', 'labels', 'bed_file', 'reference_fasta']
    for arg in vars(args):
        if arg in ignore:
            continue

        attr = getattr(args, arg)

        if os.path.isdir(str(attr)) or os.path.isfile(str(attr)):
            continue

        if os.path.isabs(str(attr)):
            attr = os.path.splitext(os.path.basename(attr))[0]

        save_weight += '__' + str(arg) + '_' + str(attr)

    save_weight += '.hd5'
    print('save weight path:', save_weight)

    return save_weight


def weight_path_to_title(wp):
    return wp.split('/')[-1].replace('__', '-')

#
# # from gist: https://gist.github.com/kukuruza/03731dc494603ceab0c5
# from math import sqrt
#
#
# def put_kernels_on_grid(kernel, pad=1):
#     """Visualize conv. filters as an image (mostly for the 1st layer).
#   Arranges filters into a grid, with some paddings between adjacent filters.
#   Args:
#     kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
#     pad:               number of black pixels around each filter (between them)
#   Return:
#     Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
#   """
#
#     # get shape of the grid. NumKernels == grid_Y * grid_X
#     def factorization(n):
#         for i in range(int(sqrt(float(n))), 0, -1):
#             if n % i == 0:
#                 if i == 1: print('Who would enter a prime number of filters')
#                 return i, int(n / i)
#
#     (grid_Y, grid_X) = factorization(kernel.get_shape()[3].value)
#     print('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))
#
#     x_min = tf.reduce_min(kernel)
#     x_max = tf.reduce_max(kernel)
#     kernel = (kernel - x_min) / (x_max - x_min)
#
#     # pad X and Y
#     x = tf.pad(kernel, tf.constant([[pad, pad], [pad, pad], [0, 0], [0, 0]]), mode='CONSTANT')
#
#     # X and Y dimensions, w.r.t. padding
#     Y = kernel.get_shape()[0] + 2 * pad
#     X = kernel.get_shape()[1] + 2 * pad
#
#     channels = kernel.get_shape()[2]
#
#     # put NumKernels to the 1st dimension
#     x = tf.transpose(x, (3, 0, 1, 2))
#     # organize grid on Y axis
#     x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))
#
#     # switch X and Y axes
#     x = tf.transpose(x, (0, 2, 1, 3))
#     # organize grid on X axis
#     x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))
#
#     # back to normal order (not combining with the next step for clarity)
#     x = tf.transpose(x, (2, 1, 3, 0))
#
#     # to tf.image_summary order [batch_size, height, width, channels],
#     #   where in this case batch_size == 1
#     x = tf.transpose(x, (3, 0, 1, 2))
#
#     # scaling to [0, 255] is not necessary for tensorboard
#     return x
#

if '__main__' == __name__:
    run()
