'''
rnn_char.py
DSDE Deep Learning RNN Example
Train character level RNNs on nietzsche, wikipedia or reference DNA
sam@broadinstitute.org
Adapted from:
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
At least 20 epochs are required before the generated text
starts sounding coherent.
It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.
If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function

from keras.layers import Dense, Activation, LSTM, GRU
from keras.models import Sequential, load_model
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop
from Bio import Seq, SeqIO

import numpy as np
import argparse
import random
import sys


def run():
	#train_from_scratch()
	load_model_and_sample()

def train_from_scratch():
	lstm_units = 128
	window_size = 40

	text = dna_from_reference()
	x,y = make_io_tensors(text, window_size)
	chars = sorted(list(set(text)))

	model = lstm_model(lstm_units, window_size, len(chars), state=True)
	train_rnn(model, x, y, text, 'lstm_dna')


def load_model_and_sample():
	text = wikipedia()
	model = load_model('rnn_model_lstm_dna.hd5')
	generate_sample_sentences(model, text, 40)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training Data ~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def dna_from_reference(chrom='21'):
	reference_hg19 = '/dsde/data/deep/vqsr/Homo_sapiens_assembly19.fasta'
	record_dict = SeqIO.to_dict(SeqIO.parse(reference_hg19, "fasta"))
	dna = str(record_dict[chrom].seq[35000000:37000000])
	return dna


def nietzsche():
	path = get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
	text = open(path).read().lower()
	return text	


def wikipedia():
	text = open('/dsde/data/deep/wikipedia/enwik8').read()
	return text[:2000000]


def make_io_tensors(text, window_size, step=3):
	print('text excerpt:', text[:30])
	print('corpus length:', len(text))
	chars = sorted(list(set(text)))
	print('total chars:', len(chars))
	char_indices = dict((c, i) for i, c in enumerate(chars))

	sentences = []
	next_chars = []
	for i in range(0, len(text) - window_size, step):
		sentences.append(text[i: i + window_size])
		next_chars.append(text[i + window_size])
	print('nb sequences:', len(sentences))

	print('Vectorization...')
	x = np.zeros((len(sentences), window_size, len(chars)), dtype=np.bool)
	y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
	for i, sentence in enumerate(sentences):
		for t, char in enumerate(sentence):
			x[i, t, char_indices[char]] = 1
		y[i, char_indices[next_chars[i]]] = 1
	return x, y


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Models ~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def lstm_model(lstm_units, input_size, alphabet_size):    
	model = Sequential()
	model.add(LSTM(lstm_units, input_shape=(input_size, alphabet_size)))
	model.add(Dense(alphabet_size, activation='softmax'))

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	print('Build model...', model.summary())
	return model


def gru_model(lstm_units, input_size, alphabet_size):    
	model = Sequential()
	model.add(GRU(128, input_shape=(input_size, alphabet_size)))
	model.add(Dense(alphabet_size, activation='softmax'))

	optimizer = RMSprop(lr=0.01)
	model.compile(loss='categorical_crossentropy', optimizer=optimizer)
	print('Build model...', model.summary())
	return model


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~ Sample with Temperature ~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def sample(preds, temperature=1.0):
	# helper function to sample an index from a probability array
	preds = np.asarray(preds).astype('float64')
	preds = np.log(preds) / temperature
	exp_preds = np.exp(preds)
	preds = exp_preds / np.sum(exp_preds)
	probas = np.random.multinomial(1, preds, 1)
	return np.argmax(probas)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~ Train and Generate  ~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_rnn(model, x, y, text, title):
	# train the model, output generated text after each iteration
	for iteration in range(1, 60):
		print()
		print('-' * 50)
		print('Iteration', iteration)
		model.fit(x, y, batch_size=512, nb_epoch=1)
		model.save('rnn_model_'+title+'.hd5')
		generate_sample_sentences(model, text, x.shape[1])


def generate_sample_sentences(model, text, window_size):
	chars = sorted(list(set(text)))
	char_indices = dict((c, i) for i, c in enumerate(chars))
	indices_char = dict((i, c) for i, c in enumerate(chars))	
	start_index = random.randint(0, len(text) - window_size - 1)

	for temperature in [0.001, 0.1, 0.2, 0.3, 0.5, 1.1]:
		print()
		print('----- temperature:', temperature)

		generated = ''
		sentence = text[start_index: start_index + window_size]
		generated += sentence
		print('----- Generating with seed: "' + sentence + '"')
		sys.stdout.write(generated)

		for i in range(400):
			gx = np.zeros((1, window_size, len(chars)))
			for t, char in enumerate(sentence):
				gx[0, t, char_indices[char]] = 1.

			preds = model.predict(gx, verbose=0)[0]
			next_index = sample(preds, temperature)
			next_char = indices_char[next_index]

			generated += next_char
			sentence = sentence[1:] + next_char

			sys.stdout.write(next_char)
			sys.stdout.flush()
		print()	

if '__main__'==__name__:
	run()
