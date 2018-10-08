#!/usr/bin/env python
# hyperparameter_optimizer.py
#
# Supports grid search, random search, Bayesian Optimization, and A/B tests over neural net architectures.
# Makes use of the models.read_tensor_2d_annotation_model_from_args() 
#
# July 2017
# Sam Friedman 
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

# Imports
import os
import h5py
import plots
import models
import defines
import operator
import arguments
import numpy as np
import training_data as td
from collections import Counter

# Bayesian optimization imports
import GPy
import GPyOpt
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# Keras imports
import keras.backend as K

bools = ['spatial_dropout', 'batch_normalization', 'batch_normalize_input', 'valid_padding', 'annotation_shortcut',
				'conv_batch_normalize', 'fc_batch_normalize', 'annotation_batch_normalize', 'kernel_single_channel']

def run():
	args = arguments.parse_args()

	cfg = K.tf.ConfigProto()
	cfg.gpu_options.allow_growth = True
	K.set_session(K.tf.Session(config=cfg))	

	if '2d' == args.mode:
		ho = HyperparameterOptimizer()
		ho.bayesian_search_2d(args, args.iterations)
	elif '1d' == args.mode:
		ho = HyperparameterOptimizer()
		ho.bayesian_search_1d(args, args.iterations)
	elif 'mlp' == args.mode:
		ho = HyperparameterOptimizer()
		ho.bayesian_search_mlp(args, args.iterations)
	elif '1d_anno' == args.mode:
		ho = HyperparameterOptimizer()
		ho.bayesian_search_1d_anno(args, args.iterations)
	elif '2d_anno' == args.mode:
		ho = HyperparameterOptimizer()
		ho.bayesian_search_2d_anno(args, args.iterations)	
	elif 'ab_spatial' == args.mode:
		ho = HyperparameterOptimizer()
		pa = ho.get_baseline_2d_params()
		pb = ho.get_baseline_2d_params()
		pb['spatial_dropout'] = False
		ho.ab_test_2d(args, pa, pb)
	else:
		raise ValueError('Error! Unknown hyperprameter optimizer mode:', args.mode)


class HyperparameterOptimizer(object):

	def __init__(self):
		self.performances = {}
		self.max_loss = 99.9
		self.paddings = ['valid']
		self.annotation_units = [12, 16, 20]
		self.conv_dropouts = [0.0, 0.2, 0.4]
		self.fc_dropouts =  [0.0, 0.2, 0.3, 0.4]
		self.batch_normalizations =  [False]
		self.kernel_initializers = ['normal', 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform']
		self.fc_initializers = ['normal', 'he_normal', 'he_uniform', 'glorot_normal', 'glorot_uniform']		
		self.conv_widths = [6, 12, 16, 20]
		self.conv_heights = [3, 6, 12, 24]
		self.conv_layers_sets = [
									[48, 64, 96], [96, 64, 48], [32, 64, 128], [128, 64, 32],
									[64,48,32,24], [24,32,48,64], [48,64,96,128],  [128,96,64,48], 
									[48, 48, 64, 64, 96, 96], [96, 96, 64, 64, 48, 32], 
									[96, 96, 64, 64, 48, 48, 32, 32], [32, 32, 48, 48, 64, 64, 96, 96],
									[96, 96, 64, 64, 48, 48, 32, 32, 24, 24], [24, 24, 32, 32, 48, 48, 64, 64, 96, 96]
								]

		self.max_pool_sets_2d = [ 
									[],
									[(1,2)], [(1,3)], [(2,1)], [(3,1)], [(4,1)], 
									[(1,2),(1,2)], [(3,1),(3,1)], [(4,1),(4,1)], 
									[(2,1),(2,1)], [(1,3), (1,3)], [(1,4),(1,4)], 
								  	[(1,2),(1,2),(1,2)],  [(3,1),(3,1),(3,1)], 
								  	[(2,1),(2,1),(2,1)], [(1,3),(1,3),(1,3)],
								  	[(1,2),(1,2),(1,2),(1,2)], [(3,1),(3,1),(3,1),(3,1)], 
								  	[(2,1),(2,1),(2,1),(2,1)], [(1,3),(1,3),(1,3),(1,3)],
								  	[(2,2)], [(3,3)], [(4,4)], [(8,8)],
								  	[(2,2),(2,2)], [(3,3), (3,3)], [(4,4), (4,4)], [(4,8), (4,8)],
								  	[(2,2),(2,2),(2,2)], [(3,3), (3,3), (3,3)]
								]

		self.max_pool_sets_1d = [ 
									[], [2], [3], [6], [8], [2,2], 
									[3,3], [2,6], [4,4], [8,8] 
								]

		self.fc_layer_sets = [
									[16], [24], [32], 
									[32, 16], [16, 32], [32, 32]
							 ]

		self.mlp_layer_sets = [
									[32], [64], [256], [32, 16], [16, 32], [32, 32], [64, 64], [128, 128], 
									[256, 256], [512,512], [256, 128, 64], [64,32,16], [128,64,32], [512,512,512],
									[256,128,64,32], [64, 128, 256, 128, 64]
								]

		self.residual_layers_sets = [
										[]
									]


	def grid_search_2d(self, args):
		'''Grid search in hyperparameter space over convolution size and max pooling tuples.
		
		Grid search is exponentially slow so this is only practical for small sets of hyperparameters.
		See random_search_2d() below for a much faster hyperparameter optimizer
		'''
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		in_channels = defines.total_input_channels_from_args(args)
		if args.channels_last:
			tensor_shape = (args.read_limit, args.window_size, in_channels)
		else:
			tensor_shape = (in_channels, args.read_limit, args.window_size) 

		generate_train = td.tensor_annotation_generator(args, train_paths, tensor_shape)
		generate_valid = td.tensor_annotation_generator(args, valid_paths, tensor_shape)
		test = td.load_tensors_and_annotations_from_class_dirs(args, test_paths, per_class_max=args.samples)

		for c in self.conv_layers_sets:
			for m in self.max_pool_sets_2d:
				args.id = 'epochs_' + str(args.epochs) + '_' + args.node
				for units in c:
					args.id += '_' + str(units)
				args.id += '_pools'
				for pools in m:
					args.id += '_' + str(pools[0]) + '-' + str(pools[1])
				weight_path = './weights/' + args.id + '.hd5'
				model = models.read_tensor_2d_annotation_model_from_args(args, 
											conv_width = 6,
											conv_layers = c,
											conv_dropout = 0.1,
											max_pools = m,
											padding='valid',
											annotation_units = 16,
											fc_layers = [64],
											fc_dropout = 0.3)
				model = models.train_model_from_generators(args, model, generate_train, generate_valid, weight_path)
				plots.plot_roc_per_class(model, [test[0], test[1]], test[2], args.labels, args.id)


	def random_search_2d(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Architectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''		
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		in_channels = defines.total_input_channels_from_args(args)
		if args.channels_last:
			tensor_shape = (args.read_limit, args.window_size, in_channels)
		else:
			tensor_shape = (in_channels, args.read_limit, args.window_size) 

		generate_train = td.tensor_annotation_generator(args, train_paths, tensor_shape)
		generate_valid = td.tensor_annotation_generator(args, valid_paths, tensor_shape)
		test = td.load_tensors_and_annotations_from_class_dirs(args, test_paths, per_class_max=args.samples)

		for i in range(iterations):
			try:
				model, params = self.get_random_architecture(args)
			except ValueError as e:
				print('value error on architecture, skipping this iteration. Error is:\n', str(e))
				continue

			param_str = 'Iteration: ' + str(i) + '\nParameter set:\n' + str(params) + '\nTotal params:' + str(model.count_params())
			print(param_str)
			weight_path = './weights/hyper_opt_' + str(i) + '.hd5'
			model = models.train_model_from_generators(args, model, generate_train, generate_valid, weight_path)
			param_str += plots.string_auc_per_class(model, [test[0], test[1]], test[2], args.labels)
			plots.print_auc_per_class(model, [test[0], test[1]], test[2], args.labels)
			self.performances[param_str] = plots.get_auc(model, [test[0], test[1]], test[2], args.labels)

		self.write_results_to_file('./param_opt_2d_' + args.id + '.txt')		
		for k, v in sorted(self.performances.items(), key=operator.itemgetter(1)):	
			print(k, '\nGot AUC:', self.performances[k])


	def bayesian_search_2d(self, args, iterations):
		'''Bayesian optimization in hyperparameter space searching for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''
		stats = Counter()	
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.tensor_generator_from_label_dirs_and_args(args, train_paths)
		generate_valid = td.tensor_generator_from_label_dirs_and_args(args, valid_paths)
		generate_test  = td.tensor_generator_from_label_dirs_and_args(args, test_paths)

		space = {
			'conv_width' : hp.quniform('conv_width', 3, 25, 2),
			'conv_height' : hp.quniform('conv_height', 3, 25, 2),
			'conv_layers' : hp.choice('conv_layers', self.conv_layers_sets),
			'kernel_single_channel' : hp.choice('kernel_single_channel', [0, 1]),
			'fc' : hp.choice('fc',self.fc_layer_sets),
			'valid_padding' : hp.choice('valid_padding', [0, 1]),
			'max_pools_2d' : hp.choice('max_pools_2d', self.max_pool_sets_2d),
		}
		
		def hp_loss_from_params_2d(x):
			max_loss = 99
			try:
				model = models.read_tensor_2d_model_from_args(args, 
										conv_width = int(x['conv_width']),
										conv_height = int(x['conv_height']),
										conv_layers = x['conv_layers'],
										max_pools = x['max_pools_2d'],
										kernel_single_channel = bool(x['kernel_single_channel']),
										padding = 'valid' if bool(x['valid_padding']) else 'same',
										fc_layers = x['fc']
										)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return max_loss 

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Current architecture: ', self.string_from_arch_dict(x))
				print('Loss ', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)
				
				del model
				#limit_mem()

				return loss_and_metrics[0]
			
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return 9e9')
				return max_loss

		samples = [ hyperopt.pyll.stochastic.sample(space) for n in range(2) ]
		print(samples)
		trials = hyperopt.Trials()
		best = fmin(hp_loss_from_params_2d, 
			space=space, 
			algo=tpe.suggest, 
			max_evals=args.iterations, 
			trials=trials)
		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(trials))


	def bayesian_search_2d_anno(self, args, iterations):
		'''Bayesian optimization in hyperparameter space searching for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''
		stats = Counter()	
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.tensor_generator_from_label_dirs_and_args(args, train_paths)
		generate_valid = td.tensor_generator_from_label_dirs_and_args(args, valid_paths)
		generate_test  = td.tensor_generator_from_label_dirs_and_args(args, test_paths)

		space = {
			'conv_width' : hp.quniform('conv_width', 3, 25, 2),
			'conv_height' : hp.quniform('conv_height', 3, 25, 2),
			'conv_layers' : hp.choice('conv_layers', self.conv_layers_sets),
			'kernel_single_channel' : hp.choice('kernel_single_channel', [0, 1]),
			'fc' : hp.choice('fc',self.fc_layer_sets),
			'valid_padding' : hp.choice('valid_padding', [0, 1]),
			'annotation_units' : hp.quniform('annotation_units', 16, 128, 16),
			'annotation_shortcut' : hp.choice('annotation_shortcut', [0, 1]),
			'max_pools_2d' : hp.choice('max_pools_2d', self.max_pool_sets_2d),
		}
		
		def hp_loss_from_params_2d_anno(x):
			max_loss = 99
			try:
				model = models.read_tensor_2d_annotation_model_from_args(args, 
										conv_width = int(x['conv_width']),
										conv_height = int(x['conv_height']),
										conv_layers = x['conv_layers'],
										max_pools = x['max_pools_2d'],
										padding = 'valid' if bool(x['valid_padding']) else 'same',
										kernel_single_channel = bool(x['kernel_single_channel']),
										annotation_units = int(x['annotation_units']),
										annotation_shortcut = bool(x['annotation_shortcut']),
										fc_layers = x['fc'])

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return max_loss

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Current architecture: ', self.string_from_arch_dict(x))
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				del model

				return loss_and_metrics[0]
			
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps?')
				return max_loss

		samples = [ hyperopt.pyll.stochastic.sample(space) for n in range(2) ]
		print(samples)
		trials = hyperopt.Trials()
		best = fmin(hp_loss_from_params_2d_anno, 
			space=space, 
			algo=tpe.suggest, 
			max_evals=args.iterations, 
			trials=trials)
		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(trials))


	def bayesian_search_1d(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''		
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.tensor_generator_from_label_dirs_and_args(args, train_paths)
		generate_valid = td.tensor_generator_from_label_dirs_and_args(args, valid_paths)
		generate_test  = td.tensor_generator_from_label_dirs_and_args(args, test_paths)

		stats = Counter()
		
		space = {
			'conv_width' : hp.quniform('conv_width', 3, 25, 2),
			'conv_layers' : hp.choice('conv_layers', self.conv_layers_sets),
			'valid_padding' : hp.choice('valid_padding', [0, 1]),
			'max_pools_1d' : hp.choice('max_pools_1d', self.max_pool_sets_1d),
			'fc' : hp.choice('fc',self.fc_layer_sets),
		}
		
		def loss_from_params_1d(x):
			try:
				model = models.build_reference_1d_model_from_args(args, 
										conv_width = int(x['conv_width']),
										conv_layers = x['conv_layers'],
										max_pools = x['max_pools_1d'],
										padding = 'valid' if bool(x['valid_padding']) else 'same',
										fc_layers = x['fc'])			


				if model.count_params() > args.max_parameters:
					print('Model too big')
					return self.max_loss

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Current architecture: ', self.string_from_arch_dict(x))
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				del model
				return loss_and_metrics[0]

			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return max loss')
				return self.max_loss       

		trials = hyperopt.Trials()
		best = fmin(loss_from_params_1d, 
			space=space, 
			algo=tpe.suggest, 
			max_evals=args.iterations, 
			trials=trials)

		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(trials))


	def bayesian_search_1d_anno(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''		
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.tensor_generator_from_label_dirs_and_args(args, train_paths)
		generate_valid = td.tensor_generator_from_label_dirs_and_args(args, valid_paths)
		generate_test  = td.tensor_generator_from_label_dirs_and_args(args, test_paths)

		stats = Counter()
		
		space = {
			'conv_width' : hp.quniform('conv_width', 3, 25, 2),
			'conv_layers' : hp.choice('conv_layers', self.conv_layers_sets),
			'fc' : hp.choice('fc',self.fc_layer_sets),
			'valid_padding' : hp.choice('valid_padding', [0, 1]),
			'annotation_units' : hp.quniform('annotation_units', 16, 128, 16),
			'annotation_shortcut' : hp.choice('annotation_shortcut', [0, 1]),
			'max_pools_1d' : hp.choice('max_pools_1d', self.max_pool_sets_1d),
		}


		def loss_from_params_1d(x):
			try:
				model = models.build_reference_annotation_1d_model_from_args(args, 
										conv_width = int(x['conv_width']),
										conv_layers = x['conv_layers'],
										max_pools = x['max_pools_1d'],
										padding = 'valid' if bool(x['valid_padding']) else 'same',
										fc_layers = x['fc'],
										annotation_units = int(x['annotation_units']),
										annotation_shortcut = bool(x['annotation_shortcut']),
									)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return self.max_loss

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Current architecture: ', self.string_from_arch_dict(x))
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				del model
				return loss_and_metrics[0]

			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps?')
				return self.max_loss

		trials = hyperopt.Trials()
		best = fmin(loss_from_params_1d, 
			space=space, 
			algo=tpe.suggest, 
			max_evals=args.iterations, 
			trials=trials)

		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(trials))


	def bayesian_search_mlp(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''
		args.tensor_map = None	
		generate_train, generate_valid, generate_test = td.train_valid_test_generators_from_args(args)

		stats = Counter()
		
		space = {
			'fc' : hp.choice('fc',self.fc_layer_sets),
			'shortcut' : hp.choice('shortcut', [0, 1]),
			'batch_normalization' : hp.choice('batch_normalization', [0, 1]),
			'batch_normalize_input' : hp.choice('batch_normalize_input', [0, 1]),
		}

		def loss_from_params_mlp(x):
			try:
				model = models.annotation_multilayer_perceptron_from_args(args,
											fc_layers = layer_set,
											#dropout = float(x['dropout']),
											skip_connection = bool(x['shortcut']),
											batch_normalization = bool(x['batch_normalization']),
											batch_normalize_input = bool(x['batch_normalize_input'])
											)
											
				if model.count_params() > args.max_parameters:
					print('Model too big')
					return self.max_loss

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Current architecture: ', self.string_from_arch_dict(x))
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				del model
				return loss_and_metrics[0]

			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return 9e9')
				return self.max_loss

				trials = hyperopt.Trials()

		best = fmin(loss_from_params_mlp, 
			space=space, 
			algo=tpe.suggest, 
			max_evals=args.iterations, 
			trials=trials)

		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(trials))


	def string_from_best_trials(self, trials):
		
		s = ''

		best_trial_idx = np.argmin(trials.losses())
		x = trials.trials[best_trial_idx]['misc']['vals']
		print('lowest loss was:', trials.losses()[best_trial_idx], ' at trial:', best_trial_idx)
		for k in x:
			
			s += '\n' + k + ' = '
			v = x[k][0]

			if k == 'fc':
				s += str(self.fc_layer_sets[int(v)])
			elif k == 'mlp_fc':
				s += str(self.mlp_layer_sets[int(v)])	
			elif k == 'conv_layers':
				s += str(self.conv_layers_sets[int(v)])
			elif k == 'max_pools_1d':
				s += str(self.max_pool_sets_1d[int(v)])
			elif k == 'max_pools_2d':
				s += str(self.max_pool_sets_2d[int(v)])
			elif k == 'residual_layers':
				s += str(self.residual_layers_sets[int(v)])					
			elif k in bools:
				s += str(bool(v))
			else:
				s += str(v)

		return s


	def string_from_arch_dict(self, x):
		s = ''
		for k in x:
			
			s += '\n' + k + ' = '
			v = x[k]

			if k in bools:
				s += str(bool(v))
			else:
				s += str(v)

		return s


	def ab_test_2d(self, args, params_a, params_b):
		'''A/B Test between two different architectures.

		Trains two different models and compares their performance.

		Arguments:
			args: the arguments namespace
			params_a: The hyperparameters for architecture A
			params_b: The hyperparameters for architecture B
		'''		
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		in_channels = defines.total_input_channels_from_args(args)
		if args.channels_last:
			tensor_shape = (args.read_limit, args.window_size, in_channels)
		else:
			tensor_shape = (in_channels, args.read_limit, args.window_size) 

		generate_train = td.tensor_annotation_generator(args, train_paths, tensor_shape)
		generate_valid = td.tensor_annotation_generator(args, valid_paths, tensor_shape)
		test = td.load_tensors_and_annotations_from_class_dirs(args, test_paths, per_class_max=args.samples)
		
		model_a = self.model_from_params_2d(args, params_a)
		weight_path = './weights/hyper_opt_a.hd5'
		model = models.train_model_from_generators(args, model_a, generate_train, generate_valid, weight_path)
		plots.print_auc_per_class(model_a, [test[0], test[1]], test[2], args.labels)
		self.performances[str(params_a)] = plots.get_auc(model_a, [test[0], test[1]], test[2], args.labels)

		model_b = self.model_from_params_2d(args, params_b)
		weight_path = './weights/hyper_opt_b.hd5'
		model = models.train_model_from_generators(args, model_b, generate_train, generate_valid, weight_path)
		plots.print_auc_per_class(model_b, [test[0], test[1]], test[2], args.labels)
		self.performances[str(params_b)] = plots.get_auc(model_b, [test[0], test[1]], test[2], args.labels)

		self.write_results_to_file('./param_ab_test_' + args.id + '.txt')		
		for k, v in sorted(self.performances.items()):	
			print(k, '\nGot AUC:', self.performances[k])

	def conv_layers_from_params(self, x):
		return [ min(350, max(1, int(x[4]*(x[3]**i)))) for i in range(int(x[2]))]


	def get_random_architecture_1d(self, args):
		'''Create a random 1d architecture.

		Draw random samples from the hyperparameter sets defined at the top of the class.
		
		Returns
			model: the random architecture as a keras model
			params: dict of the chosen parameter set
		'''
		params = self.get_random_params()
		model = models.build_reference_annotation_1d_model_from_args(args, 
									conv_width = params['conv_width'], 
									conv_layers = params['conv_layers'],
									conv_dropout = params['conv_dropout'],
									spatial_dropout = params['spatial_dropout'],
									max_pools = params['max_pools_1d'],
									padding=params['padding'],
									annotation_units = params['anno_units'],
									fc_layers = params['fc'],
									fc_dropout = params['fc_dropout'],
									batch_normalization = params['batch_normalization'],
									kernel_initializer=params['kernel_initializer'],
									fc_initializer=params['fc_initializer'])

		return model, params


	def get_random_params(self):
		params = {}

		cwi = np.random.randint(len(self.conv_widths))
		params['conv_width'] = self.conv_widths[cwi]
		chi = np.random.randint(len(self.conv_heights))
		params['conv_height'] = self.conv_heights[chi]		
		ci = np.random.randint(len(self.conv_layers_sets))
		params['conv_layers'] = self.conv_layers_sets[ci]
		mi = np.random.randint(len(self.max_pool_sets_2d))
		params['max_pools'] = self.max_pool_sets_2d[mi]
		mi = np.random.randint(len(self.max_pool_sets_1d))
		params['max_pools_1d'] = self.max_pool_sets_1d[mi]
		fci = np.random.randint(len(self.fc_layer_sets))
		params['fc'] = self.fc_layer_sets[fci]
		paddingi = np.random.randint(len(self.paddings))
		params['padding'] = self.paddings[paddingi]
		annoi = np.random.randint(len(self.annotation_units))
		params['anno_units'] = self.annotation_units[annoi]
		fc_dropouti = np.random.randint(len(self.fc_dropouts))
		params['fc_dropout'] = self.fc_dropouts[fc_dropouti]
		conv_dropouti = np.random.randint(len(self.conv_dropouts))
		params['conv_dropout'] = self.conv_dropouts[conv_dropouti]
		params['spatial_dropout'] = False
		bni = np.random.randint(len(self.batch_normalizations))
		params['batch_normalization'] = self.batch_normalizations[bni]
		ki = np.random.randint(len(self.kernel_initializers))
		params['kernel_initializer'] = self.kernel_initializers[ki]
		fci = np.random.randint(len(self.fc_initializers))
		params['fc_initializer'] = self.fc_initializers[fci]

		return params


	def get_random_architecture(self, args):
		'''Create a random 2D architecture.

		Draw random samples from the hyperparameter sets defined at the top of the class.
		Bit of a dangerous construction, catches ValueErrors thrown by invalid architectures,
		and tries again. Could certainly overflow the stack. Not production ready!

		Returns
			model: the random architecture as a keras model
			params: dict of the chosen parameter set
		'''
		params = self.get_random_params()
		
		model = self.model_from_params_2d(args, params)

		return model, params


	def get_baseline_2d_params(self):
		params = {}

		params['fc'] = [32]
		params['anno_units'] = 16
		params['conv_width'] = 16
		params['conv_height'] = 16
		params['fc_dropout'] = 0.3
		params['padding'] = 'valid'
		params['conv_dropout'] = 0.4
		params['spatial_dropout'] = False
		params['max_pools'] = [(3,1),(3,1)]
		params['batch_normalization'] = False
		params['fc_initializer'] = 'glorot_normal'
		params['conv_layers'] = [316, 160, 128, 64]
		params['kernel_initializer'] = 'glorot_normal'
		
		return params
	

	def model_from_params_2d(self, args, params):
		'''Create a 2d architecture with hyperparameters given by params.
		
		Arguments:
			params: dict of the chosen parameter set

		Returns
			model: the random architecture as a keras model
		'''
		return models.read_tensor_2d_annotation_model_from_args(args, 
									conv_width = params['conv_width'],
									conv_height = params['conv_height'], 
									conv_layers = params['conv_layers'],
									conv_dropout = params['conv_dropout'],
									spatial_dropout = params['spatial_dropout'],
									max_pools = params['max_pools'],
									padding = params['padding'],
									annotation_units = params['anno_units'],
									fc_layers = params['fc'],
									fc_dropout = params['fc_dropout'],
									batch_normalization = params['batch_normalization'],
									kernel_initializer=params['kernel_initializer'],
									fc_initializer=params['fc_initializer'])


	def write_results_to_file(self, file_name):
		with open(file_name, 'w') as f:
			for k, v in sorted(self.performances.items(), key=operator.itemgetter(1)):	
				f.write(str(k) + '\nGot AUC:' + str(self.performances[k]))		

def limit_mem():
	try:
		K.clear_session()
		#K.get_session().close()
		# cfg = K.tf.ConfigProto()
		# cfg.gpu_options.allow_growth = True
		# K.set_session(K.tf.Session(config=cfg))
	except AttributeError as e:
		print('Could not clear session. Maybe you are using Theano backend?')


# Back to the top!
if __name__ == '__main__':
	run()
