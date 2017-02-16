# CNN with drop out relu and max pool
import cPickle
import gzip
import os
import sys
import theano
from theano import tensor as T
import theano.tensor.signal as S
import theano.tensor.signal.downsample
import numpy as np
import matplotlib.pyplot as plt

seed = 1234
rng = np.random.RandomState(seed)
srng = T.shared_randomstreams.RandomStreams(rng.randint(seed)) 
#theano.config.set_floatX(32)

def load_data(dataset):
    ''' Loads the dataset
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #whose row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    return train_set, valid_set, test_set

def floatX(X):
	return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
	return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def linear_model(X, w):
	return X * w

def logistic_model(X, w):
	return T.nnet.softmax(T.dot(X, w))

def rectify(X):
	return T.maximum(X, 0)

def softmax(X):
	e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
	return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def RMSprop(cost, params, lr=0.0001, rho=0.9, epsilon=1e-6):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		acc = theano.shared(p.get_value() * 0.)
		acc_new = rho * acc + (1 - rho) * g ** 2
		gradient_scaling = T.sqrt(acc_new + epsilon)
		g = g / gradient_scaling
		updates.append((acc, acc_new))
		updates.append((p, p - lr * g))
	return updates

def sgd(cost, params, lr=0.05):
	grads = T.grad(cost=cost, wrt=params)
	updates = []
	for p, g in zip(params, grads):
		updates.append([p, p - g *lr])
	return updates

def dropout(X, p =0.):
	if p > 0:
		retain_prob = 1- p
		X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
		X /= retain_prob
	return X

def mlp_dropout_model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden):
	X = dropout(X, p_drop_input)
	h = rectify(T.dot(X, w_h))

	h = dropout(h, p_drop_hidden)
	h2 = rectify(T.dot(h, w_h2))

	h2 = dropout(h2, p_drop_hidden)
	py_x = softmax(T.dot(h2, w_o))

	return h, h2, py_x

def cnn_model(X, w, w2, w3, w4, w_o, p_drop_conv, p_drop_hidden):
	l1a = rectify(T.nnet.conv2d(X, w, border_mode='full'))
	l1 = S.downsample.max_pool_2d(l1a, (2, 2))
	l1 = dropout(l1, p_drop_conv)

	l2a = rectify(T.nnet.conv2d(l1, w2))
	l2 = T.signal.downsample.max_pool_2d(l2a, (2, 2))
	l2 = dropout(l2, p_drop_conv)

	l3a = rectify(T.nnet.conv2d(l2, w3))
	l3b = T.signal.downsample.max_pool_2d(l3a, (2, 2))
	l3 = T.flatten(l3b, outdim=2)
	l3 = dropout(l3, p_drop_conv)

	l4 = rectify(T.dot(l3, w4))
	l4 = dropout(l4, p_drop_hidden)

	pyx = softmax(T.dot(l4, w_o))
	return l1, l2, l3, l4, pyx

def make_one_hot(y):
	ohy = np.zeros((len(y), 10))
	for i in range(0, len(y)):
		ohy[i][y[i]] = 1.0
	return ohy

def run_cnn_on_mnist():
	train, test, valid = load_data('/home/sam/Dropbox/Code/python/cnn/data/mnist.pkl.gz')

	trX = train[0]
	trY = make_one_hot(train[1])

	teX = test[0]
	teY = make_one_hot(test[1])

	print 'trx shape b4:', trX.shape
	print 'trY shape b4:', trY.shape
	trX = trX.reshape(-1, 1, 28, 28)
	teX = teX.reshape(-1, 1, 28, 28)

	print 'trx shape after:', trX.shape

	X = T.ftensor4()
	Y = T.fmatrix()

	w1 = init_weights((128, 1, 3, 3))
	w2 = init_weights((128, 128, 3, 3))
	w3 = init_weights((128, 128, 3, 3))
	w4 = init_weights((128 * 3 * 3 , 1024))
	w_o = init_weights((1024, 10))

	l1n, l2n, l3n, l4n, py_x = cnn_model(X, w1, w2, w3, w4, w_o, 0.2, 0.5)
	l1, l2, l3, l4, py_x = cnn_model(X, w1, w2, w3, w4, w_o, 0.0, 0.0)
	y_x = T.argmax(py_x, axis=1)

	cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
	params =[w1, w2, w3, w4, w_o]
	updates = RMSprop(cost, params)

	train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
	predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
	get_hidden_weights = theano.function(inputs=[], outputs=w_o)

	batch_size = 128;
	print 'start training...'
	for i in range(100):
		for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
			cost = train(trX[start:end], trY[start:end])
		print 'On training batch', i, ' training accuracy', np.mean(np.argmax(trY[start:end], axis=1) == predict(trX[start:end]))	
		print 'Average validation accuracy', np.mean(np.argmax(teY, axis=1) == predict(teX))


if __name__ == '__main__':
    run_cnn_on_mnist()
