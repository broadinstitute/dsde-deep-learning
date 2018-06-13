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
									#[], 
									[(1,2)], [(1,3)], [(2,1)], [(3,1)], [(4,1)], 
									[(1,2),(1,2)], [(2,1),(2,1)], [(3,1),(3,1)], [(1,3), (1,3)],
								  	[(2,1),(6,1)],[(4,1),(4,1)], [(4,1),(8,1)], 
								  	[(3,3), (3,3)], [(4,4), (4,4)], [(4,8), (4,8)],
								  	[(1,2),(1,2),(1,2)], [(2,1),(2,1),(2,1)], [(3,1),(3,1),(3,1)]
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

		bounds = [
			{'name':'conv_width', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			{'name':'conv_height', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			{'name':'conv_layers', 'type':'discrete', 'domain':range(len(self.conv_layers_sets))},
			{'name':'kernel_single_channel', 'type':'categorical', 'domain':(0, 1)},
			{'name':'fc', 'type':'discrete', 'domain':range(len(self.fc_layer_sets))},
			{'name':'valid_padding', 'type':'categorical', 'domain':(0, 1)},
			{'name':'max_pools_2d', 'type':'discrete', 'domain':range(len(self.max_pool_sets_2d))}
		]

		space = {
			'conv_width' : hp.quniform('conv_width', 3, 19, 2),
			'conv_height' : hp.quniform('conv_height', 3, 19, 2),
			'conv_layers' : hp.choice('conv_layers', self.conv_layers_sets),
			'kernel_single_channel' : hp.choice('kernel_single_channel', [0, 1]),
			'fc' : hp.choice('fc',self.fc_layer_sets),
			'valid_padding' : hp.choice('valid_padding', [0, 1]),
			'max_pools_2d' : hp.choice('max_pools_2d', self.max_pool_sets_2d),
		}
		
		def hp_loss_from_params_2d(x):
			#print('my x is:', x)
			#conv_layers = self.conv_layers_sets[int(p[param_keys['conv_layers']])]
			#max_pools = self.max_pool_sets_2d[int(p[param_keys['max_pools_2d']])]
			#fc_layers = self.fc_layer_sets[int(p[param_keys['fc']])]
			max_loss = 9e9
			try:
				model = models.read_tensor_2d_model_from_args(args, 
										conv_width = int(x['conv_width']),
										conv_height = int(x['conv_height']),
										conv_layers = x['conv_layers'],
										max_pools = x['max_pools_2d'],
										padding = 'valid' if bool(x['valid_padding']) else 'same',
										fc_layers = x['fc']
										)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return max_loss 

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Loss ', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'init numdata:', args.patience, 'Model size', model.count_params())
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)
				
				#limit_mem()
				print('x is', x)
				print('Current architecture: ', self.string_from_arch_dict(x))
				return loss_and_metrics[0]
			
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return 9e9')
				return max_loss

		samples = [ hyperopt.pyll.stochastic.sample(space) for n in range(2) ]
		print(samples)
		trials = hyperopt.Trials()
		best = fmin(hp_loss_from_params_2d, space=space, algo=tpe.suggest, max_evals=args.iterations, trials=trials)
		print('trial dicts', trials.trials)
		print('trials.losses', trials.losses())
		print('best is:', best)
		print('best str is:', self.string_from_best_trials(best, trials.trials))


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

		bounds = [
			{'name':'conv_width', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			{'name':'conv_height', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			{'name':'conv_layers', 'type':'discrete', 'domain':range(len(self.conv_layers_sets))},
			{'name':'kernel_single_channel', 'type':'categorical', 'domain':(0, 1)},
			{'name':'valid_padding', 'type':'categorical', 'domain':(0, 1)},
			{'name':'max_pools_2d', 'type':'discrete', 'domain':range(len(self.max_pool_sets_2d))},
			#{'name':'residual_layers', 'type':'discrete', 'domain':range(len(self.residual_layers_sets))},
			#{'name':'annotation_units', 'type':'discrete', 'domain':(16,32,64)},
			{'name':'annotation_shortcut', 'type':'categorical', 'domain':(0, 1)},
			{'name':'fc', 'type':'discrete', 'domain':range(len(self.fc_layer_sets))},
		]

		param_keys = { d['name']:i for i,d in enumerate(bounds)}

		def loss_from_params_2d_anno(x):
			p = x[0]
			fc_layers = self.fc_layer_sets[int(p[param_keys['fc']])]
			conv_layers = self.conv_layers_sets[int(p[param_keys['conv_layers']])]
			max_pool_set = self.max_pool_sets_2d[int(p[param_keys['max_pools_2d']])]
			#residual_layers = self.residual_layers_sets[int(p[param_keys['residual_layers']])]

			try:
				print(self.str_from_params_and_keys(p, param_keys))
				model = models.read_tensor_2d_annotation_model_from_args(args, 
										conv_width = int(p[param_keys['conv_width']]),
										conv_height = int(p[param_keys['conv_height']]),
										conv_layers = conv_layers,
										max_pools = max_pool_set,
										padding = 'valid' if bool(p[param_keys['valid_padding']]) else 'same',
										kernel_single_channel = bool(p[param_keys['kernel_single_channel']]),
										#annotation_units = int(p[param_keys['annotation_units']]),
										annotation_shortcut = bool(p[param_keys['annotation_shortcut']]),
										fc_layers = fc_layers,
										)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'init numdata:', args.patience, 'Model size', model.count_params())
				print(self.str_from_params_and_keys(p, param_keys))

				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				limit_mem()
				return loss_and_metrics[0]
			
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps?')
				return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same

		optimizer = GPyOpt.methods.BayesianOptimization(f=loss_from_params_2d_anno, # Objective function       
                                             domain=bounds,          				# Box-constraints of the problem
                                             initial_design_numdata=args.patience, 	# Random models built before Bayesian optimization
											model_type='GP',
                                             acquisition_type='EI',        			# Expected Improvement
                                             acquisition_optimizer='DIRECT',
                                             exact_feval=True,						# Is loss exact or noisy? Noisy!
                                             verbosity=True,							# Talk to me!
                                             normalize_Y = False
                                             )           

		optimizer.run_optimization(args.iterations, max_time=6e10, eps=0, verbosity=True, report_file=args.output_dir + args.id + '.bayes_report')
		print('Best parameter set:', optimizer.x_opt)
		print(self.str_from_params_and_keys(optimizer.x_opt, param_keys))
		with open(args.output_dir + args.id + '.bayes_report', 'a') as f:
			f.write(self.str_from_params_and_keys(optimizer.x_opt, param_keys))


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
		bounds = [
			{'name':'conv_width', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			#{'name':'conv_dropout', 'type':'continuous', 'domain':(0.0, 0.4)},
			{'name':'conv_layers', 'type':'discrete', 'domain':range(len(self.conv_layers_sets))},
			#{'name':'conv_batch_normalize', 'type':'categorical', 'domain':(0, 1)},			
			#{'name':'spatial_dropout', 'type':'categorical', 'domain':(0, 1)},
			{'name':'fc', 'type':'discrete', 'domain':range(len(self.fc_layer_sets))},
			#{'name':'fc_dropout', 'type':'continuous', 'domain':(0.0, 0.4)},
			{'name':'valid_padding', 'type':'categorical', 'domain':(0, 1)},
			{'name':'max_pools_1d', 'type':'discrete', 'domain':range(len(self.max_pool_sets_1d))}
		]

		param_keys = { d['name']:i for i,d in enumerate(bounds)}

		def loss_from_params_1d(x):
			p = x[0]
			conv_layers = self.conv_layers_sets[int(p[param_keys['conv_layers']])]
			max_pool_set = self.max_pool_sets_1d[int(p[param_keys['max_pools_1d']])]
			fc_layers = self.fc_layer_sets[int(p[param_keys['fc']])]
			try:
				model = models.build_reference_1d_model_from_args(args, 
											conv_width = int(p[param_keys['conv_width']]), 
											conv_layers = conv_layers,
											#conv_dropout = float(p[param_keys['conv_dropout']]),
											#conv_batch_normalize = bool(p[param_keys['conv_batch_normalize']]),
											#spatial_dropout = bool(p[param_keys['spatial_dropout']]),
											max_pools = max_pool_set,
											padding = 'valid' if bool(p[param_keys['valid_padding']]) else 'same',
											fc_layers = fc_layers
											#fc_dropout = float(p[param_keys['fc_dropout']])
											)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'init numdata:', args.patience, 'Model size', model.count_params())
				print(self.str_from_params_and_keys(p, param_keys))
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				limit_mem()
				return loss_and_metrics[0]
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return 9e9')
				return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same


		optimizer = GPyOpt.methods.BayesianOptimization(f=loss_from_params_1d,  	# Objective function       
                                             domain=bounds,          				# Box-constraints of the problem
                                             initial_design_numdata=args.patience, 	# Random models built before Bayesian optimization
                                             model_type='GP',
                                             acquisition_type='EI',        			# Expected Improvement
                                             acquisition_optimizer='DIRECT',
                                             exact_feval=True,						# Is loss exact or noisy? Noisy!
                                             verbosity=True,							# Talk to me!
                                             normalize_Y = False
                                             )           

		optimizer.run_optimization(max_iter=args.iterations, max_time=6e10, verbosity=True, eps=0, report_file=args.output_dir + args.id + '.bayes_report')
		print('Best parameter set:', optimizer.x_opt)
		print(self.str_from_params_and_keys(optimizer.x_opt, param_keys))
		with open(args.output_dir + args.id + '.bayes_report', 'a') as f:
			f.write(self.str_from_params_and_keys(optimizer.x_opt, param_keys))


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
		bounds = [
			{'name':'conv_width', 'type':'discrete', 'domain':(3,5,7,11,15,19)},
			#{'name':'conv_dropout', 'type':'continuous', 'domain':(0.0, 0.4)},
			{'name':'conv_layers', 'type':'discrete', 'domain':range(len(self.conv_layers_sets))},
			#{'name':'spatial_dropout', 'type':'categorical', 'domain':(0, 1)},
			{'name':'annotation_units', 'type':'discrete', 'domain':(24,32,48,64)},
			{'name':'annotation_shortcut', 'type':'categorical', 'domain':(0, 1)},
			#{'name':'annotation_batch_normalize', 'type':'categorical', 'domain':(0, 1)},			
			{'name':'fc', 'type':'discrete', 'domain':range(len(self.fc_layer_sets))},
			#{'name':'fc_dropout', 'type':'continuous', 'domain':(0.0, 0.4)},
			{'name':'valid_padding', 'type':'categorical', 'domain':(0, 1)},
			{'name':'max_pools_1d', 'type':'discrete', 'domain':range(len(self.max_pool_sets_1d))},
		]

		param_keys = { d['name']:i for i,d in enumerate(bounds)}

		def loss_from_params_1d(x):
			p = x[0]
			conv_layers = self.conv_layers_sets[int(p[param_keys['conv_layers']])]
			max_pool_set = self.max_pool_sets_1d[int(p[param_keys['max_pools_1d']])]
			fc_layers = self.fc_layer_sets[int(p[param_keys['fc']])]
			try:
				model = models.build_reference_annotation_1d_model_from_args(args, 
											conv_width = int(p[param_keys['conv_width']]), 
											conv_layers = conv_layers,
											max_pools = max_pool_set,
											padding = 'valid' if bool(p[param_keys['valid_padding']]) else 'same',
											annotation_units = int(p[param_keys['annotation_units']]),
											annotation_shortcut = bool(p[param_keys['annotation_shortcut']]),
											fc_layers = fc_layers
											)

				if model.count_params() > args.max_parameters:
					print('Model too big')
					return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same

				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'init numdata:', args.patience, 'Model size', model.count_params())
				print(self.str_from_params_and_keys(x[0], param_keys))
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				limit_mem()
				return loss_and_metrics[0]
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps?')
				return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same


		optimizer = GPyOpt.methods.BayesianOptimization(f=loss_from_params_1d,  	# Objective function       
                                             domain=bounds,          				# Box-constraints of the problem
                                             initial_design_numdata=args.patience, 	# Random models built before Bayesian optimization
                                             model_type='GP',
                                             acquisition_type='EI',        			# Expected Improvement
                                             acquisition_optimizer='DIRECT',
                                             exact_feval=True,						# Is loss exact or noisy? Noisy!
                                             verbosity=True,							# Talk to me!
                                             normalize_Y = False
                                             )           

		optimizer.run_optimization(max_iter=args.iterations, max_time=6e10, verbosity=True, eps=0, report_file=args.output_dir + args.id + '.bayes_report')
		print('Best parameter set:', optimizer.x_opt)
		print(self.str_from_params_and_keys(optimizer.x_opt, param_keys))
		with open(args.output_dir + args.id + '.bayes_report', 'a') as f:
			f.write(self.str_from_params_and_keys(optimizer.x_opt, param_keys))


	def bayesian_search_mlp(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''
		args.window_size = 0	
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.dna_annotation_generator(args, train_paths)
		generate_valid = td.dna_annotation_generator(args, valid_paths)
		generate_test = td.dna_annotation_generator(args, test_paths)

		stats = Counter()
		bounds = [
			{'name':'mlp_fc', 'type':'discrete', 'domain':range(len(self.mlp_layer_sets))},
			{'name':'dropout', 'type':'continuous', 'domain':(0.0, 0.6)},
			{'name':'annotation_shortcut', 'type':'categorical', 'domain':(0, 1)},
			{'name':'batch_normalization', 'type':'categorical', 'domain':(0, 1)},
			{'name':'batch_normalize_input', 'type':'categorical', 'domain':(0, 1)}
		]

		param_keys = { d['name']:i for i,d in enumerate(bounds)}

		def loss_from_params_mlp(x):
			p = x[0]
			layer_set = self.mlp_layer_sets[int(p[param_keys['mlp_fc']])]
			try:
				model = models.annotation_multilayer_perceptron_from_args(args,
											fc_layers = layer_set,
											dropout = float(p[param_keys['dropout']]),
											skip_connection = bool(p[param_keys['annotation_shortcut']]),
											batch_normalization = bool(p[param_keys['batch_normalization']]),
											batch_normalize_input = bool(p[param_keys['batch_normalize_input']])
											)
											
				if model.count_params() > args.max_parameters:
					print('Model too big')
					return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same


				model = models.train_model_from_generators(args, model, generate_train, generate_valid, args.output_dir + args.id + '.hd5')
				loss_and_metrics = model.evaluate_generator(generate_test, steps=args.validation_steps)
				stats['count'] += 1
				print('Loss:', loss_and_metrics[0], '\nCount:', stats['count'], 'iterations', args.iterations, 'init numdata:', args.patience, 'Model size', model.count_params())
				print(self.str_from_params_and_keys(p, param_keys))
				if args.inspect_model:
					image_name = args.id+'_hyper_'+str(stats['count'])+'.png'
					image_path = image_name if args.image_dir is None else args.image_dir + image_name
					models.inspect_model(args, model, generate_train, generate_valid, image_path=image_path)

				limit_mem()
				return loss_and_metrics[0]
			except ValueError as e:
				print(str(e) + '\n Impossible architecture perhaps? return 9e9')
				return np.random.uniform(100,10000) # this is ugly but optimization quits when loss is the same

		optimizer = GPyOpt.methods.BayesianOptimization(f=loss_from_params_mlp,  	# Objective function       
                                             domain=bounds,          				# Box-constraints of the problem
                                             initial_design_numdata=args.patience, 	# Random models built before Bayesian optimization
                                             model_type='GP',
                                             acquisition_type='EI',        			# Expected Improvement
                                             acquisition_optimizer='DIRECT',
                                             exact_feval=True,						# Is loss exact or noisy? Noisy!
                                             verbosity=True,						# Talk to me!
                                             normalize_Y = False
                                             )           		           

		optimizer.run_optimization(max_iter=args.iterations,
									max_time=6e10, 
									verbosity=True, 
									eps=0, 
									report_file=args.output_dir + args.id + '.bayes_report')
		
		print('Best parameter set:', optimizer.x_opt)
		print(self.str_from_params_and_keys(optimizer.x_opt, param_keys))
		with open(args.output_dir + args.id + '.bayes_report', 'a') as f:
			f.write(self.str_from_params_and_keys(optimizer.x_opt, param_keys))


	def str_from_params_and_keys(self, x, param_keys):

		s = ''
		for k in param_keys:
			s += '\n' + k + ' = '
			if k == 'fc':
				s += str(self.fc_layer_sets[int(x[param_keys[k]])])
			elif k == 'mlp_fc':
				s += str(self.mlp_layer_sets[int(x[param_keys[k]])])	
			elif k == 'conv_layers':
				s += str(self.conv_layers_sets[int(x[param_keys[k]])])
			elif k == 'max_pools_1d':
				s += str(self.max_pool_sets_1d[int(x[param_keys[k]])])
			elif k == 'max_pools_2d':
				s += str(self.max_pool_sets_2d[int(x[param_keys[k]])])
			elif k == 'residual_layers':
				s += str(self.residual_layers_sets[int(x[param_keys[k]])])					
			elif k in bools:
				s += str(bool(x[param_keys[k]]))
			else:
				s += str(x[param_keys[k]])

		return s



	def string_from_best_trials(self, best, trials):
		
		s = ''

		best_trial_idx = np.argmin(best)
		x = trials[best_trial_idx]['misc']['vals']

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


	def random_search_1d(self, args, iterations):
		'''Random search in hyperparameter space for good architectures.
		
		Create a bunch of random architectures and test their performance.
		Archtiectures are created from within the bounds defined at the top of this class.

		Arguments:
			iterations: how many architectures to try
		'''		
		train_paths, valid_paths, test_paths = td.get_train_valid_test_paths(args)

		generate_train = td.dna_annotation_generator(args, train_paths)
		generate_valid = td.dna_annotation_generator(args, valid_paths)
		test = td.load_dna_annotations_positions_from_class_dirs(args, test_paths, per_class_max=args.samples)

		for i in range(iterations):
			try:
				model, params = self.get_random_architecture_1d(args)
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

		self.write_results_to_file('./param_opt_1d_' + args.id + '.txt')
		for k, v in sorted(self.performances.items(), key=operator.itemgetter(1)):
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
		K.get_session().close()
		cfg = K.tf.ConfigProto()
		cfg.gpu_options.allow_growth = True
		K.set_session(K.tf.Session(config=cfg))
	except AttributeError as e:
		print('Could not clear session. Maybe you are using Theano backend?')


# Back to the top!
if __name__ == '__main__':
	run()
