"""
FedAvg class as trainer
"""

import logging

from models import model
from trainer import client
from utils import read_write_data
from robust import attack, defense
from compression import compression_method

from utils.torch_utils import get_flat_params_from, set_flat_params_to

import torch
import numpy as np
from absl import flags
import time

FLAGS = flags.FLAGS


class Server(object):
	"""
	server class
	"""

	def __init__(self, dataset):

		self.logger = logging.getLogger('main.server')

		self.dataset = dataset
		self.model = model.get_model()
		self.move_model_to_gpu()
		self.logger.info('Activate a server for training.')

		self.round_num = 0

		self.client, self.clientData = self.setup_clients()
		# print(">>> Activate clients number: {}".format(self.numClients))
		self.logger.info('Activate clients number: %d', self.numClients)

		self.errorfeedback = [0]*self.numClients

		self.output_metric = read_write_data.Metrics()

	def move_model_to_gpu(self):
		if FLAGS.gpu:
			torch.cuda.set_device(FLAGS.device)
			torch.backends.cudnn.enabled = True
			self.model.cuda()
			# print('>>> Server use gpu on device {}'.format(FLAGS.device))
			self.logger.info('Server use gpu on device %d', FLAGS.device)
		else:
			self.logger.info('Server use cpu')

	def setup_clients(self):
		"""
		Instantiates clients based on given train and test data
		Returns:
			one object client and the data frame for all the clients
		"""
		users, groups, train_data, test_data = self.dataset
		self.logger.debug('users information %s', users)
		self.logger.debug('traindata type %s size %d, testdata type %s size %d',
		                  type(train_data), len(train_data), type(test_data), len(train_data))

		self.numClients = len(users)

		data_frame = []

		for i, user in enumerate(users):
			data_frame.append((i, train_data[user], test_data[user]))
		baseClient = client.Client()
		return baseClient, data_frame

	def select_clients(self, seed=1, replace=False):
		"""Selects num_clients clients weighted by number of samples from possible_clients

		Args:
			1. seed: random seed
			2. num_clients: number of clients to select; default 20
				note that within function, num_clients is set to min(num_clients, len(possible_clients))

		Return:
			list of id for each selected client
		"""
		if FLAGS.clients_per_round > self.numClients:
			raise ValueError("clients per round should be smaller than total clients")
		if FLAGS.clients_per_round == self.numClients:
			return [i for i in range(self.numClients)]
		# num_clients = min(FLAGS.clients_per_round, self)
		np.random.seed(seed)
		return np.random.choice(self.numClients, FLAGS.clients_per_round, replace=replace).tolist()

	def updateMetric(self, cid, delta, compressed_delta, compressed_rate, metric):
		'''
		update error, update statis of delta, error, compression rate.
		:param delta:
		:param compressed_delta:
		:param compressed_rate:
		:return:
		'''
		self.errorfeedback[cid] = compressed_delta - delta
		delta_dict = {"delta_norm": torch.norm(compressed_delta).item(),
		              "delta_max": compressed_delta.max().item(),
		              "delta_min": compressed_delta.min().item()}
		metric.update(delta_dict)

		error_dict = {"error_norm": torch.norm(self.errorfeedback[cid]).item(),
		              "error_max": self.errorfeedback[cid].max().item(),
		              "error_min": self.errorfeedback[cid].min().item()}
		metric.update(error_dict)

		compression_dict = {"compression_rate": compressed_rate}
		metric.update(compression_dict)

	def local_train(self, client_list, round_num, condition=None):
		"""
		local train process for each client
		:return:
		delta: (num_sampels, delta)
		metric: list of dict of performance information
		"""

		deltas = []
		metrics = []
		server_flat_params = self.get_flat_model_params()

		for cid in client_list:
			if round_num == 0:
				self.errorfeedback[cid] = torch.zeros_like(self.get_flat_model_params())
			self.client.reset(self.clientData[cid], self.errorfeedback[cid], round_num)
			delta, metric = self.client.local_train(server_flat_params)
			# compressed_delta, compressed_rate = compression_method.get_compression(delta[1], condition=condition)
			# self.updateMetric(cid, delta[1], compressed_delta, compressed_rate, metric)

			deltas.append(delta)
			metrics.append(metric)

		if FLAGS.attack == 'byzantine':
			attack.norm_attack(deltas, num_std=1.5)

		i = 0
		for cid, delta, metric in zip(client_list, deltas, metrics):
			compressed_delta, compressed_rate = compression_method.get_compression(delta[1], condition=condition)
			self.updateMetric(cid, delta[1], compressed_delta, compressed_rate, metric)
			delta[1] = compressed_delta
			i += 1
			self.logger.info("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
				    "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			        "Delta: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			        "Error: norm {:>.4f} ({:>.4f}->{:>.4f})| "
				    "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
					round_num, metric['id'], i, FLAGS.clients_per_round,
					metric['grad_norm'], metric['grad_min'], metric['grad_max'],
				    metric['delta_norm'], metric['delta_min'], metric['delta_max'],
					metric['error_norm'], metric['error_min'], metric['error_max'],
					metric['train_loss'], metric['train_acc'] * 100, metric['time']))
			print("Round: {:>2d} | CID: {: >3d} ({:>2d}/{:>2d})| "
			                 "Param: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			                 "Delta: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			                 "Error: norm {:>.4f} ({:>.4f}->{:>.4f})| "
			                 "Loss {:>.4f} | Acc {:>5.2f}% | Time: {:>.2f}s".format(
				round_num, metric['id'], i, FLAGS.clients_per_round,
				metric['grad_norm'], metric['grad_min'], metric['grad_max'],
				metric['delta_norm'], metric['delta_min'], metric['delta_max'],
				metric['error_norm'], metric['error_min'], metric['error_max'],
				metric['train_loss'], metric['train_acc'] * 100, metric['time']))


		return deltas, metrics


	def aggregate(self, deltas):
		"""
		aggregate delta from clients
		:return:
		"""
		num = 0
		previous_para = self.get_flat_model_params()
		averaged_delta = torch.zeros_like(previous_para)
		for num_samples, client_delta in deltas:
			num += num_samples
			averaged_delta += client_delta * num_samples
		averaged_delta /= num
		# new_para = previous_para + FLAGS.server_lr * averaged_delta
		return averaged_delta

	def train(self):
		"""
		train process
		:return:
		"""
		self.logger.info('sampling %d clients per communication round', FLAGS.clients_per_round)

		for round_num in range(FLAGS.num_round):
			client_list = self.select_clients(seed=round_num, replace=False)
			self.logger.debug('sample clients completed')

			t0 = time.time()
			if FLAGS.compressor == 'uniform_drop':
				condition = torch.rand(self.get_flat_model_params().size()) >= FLAGS.compress_factor
				deltas, metrics = self.local_train(client_list, round_num, condition=condition)
			else:
				deltas, metrics = self.local_train(client_list, round_num)

			self.output_metric.add_extra_stats(round_num, metrics)
			self.logger.info("training time for one communication round: %f", (time.time()-t0))


			# aggregate clients delta and update the server model
			if FLAGS.defense == 'none':
				average_delta = self.aggregate(deltas)
			elif FLAGS.defense == 'krum':
				average_delta = defense.krum(deltas)
			else:
				raise ValueError("invalid defense")


			self.update_server_state(average_delta)

			self.logger.debug('communication round %d finished.', )

			self.test_latest_model_on_traindata(round_num)
			self.test_latest_model_on_testdata(round_num)

		self.output_metric.write()


	def test_latest_model_on_traindata(self, round_num):
		"""

		:param round_num:
		:return:
		"""
		stats = self.local_test(use_eval_data=False)
		self.logger.info('>>> Training info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f}.'.format(
			round_num, stats['acc'], stats['loss']))
		print('>>> Training info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f}.'.format(
			round_num, stats['acc'], stats['loss']))
		self.output_metric.add_training_stats(round_num, stats)


	def test_latest_model_on_testdata(self, round_num):
		"""

		:param round_num:
		:return:
		"""
		stats = self.local_test(use_eval_data = True)
		self.logger.info('>>>Testing info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f}.'.format(
			round_num, stats['acc'], stats['loss']))
		print('>>>Testing info in Round: {: >4d} / Acc: {:.3%} / Loss: {:.4f}.'.format(
			round_num, stats['acc'], stats['loss']))
		self.output_metric.add_test_stats(round_num, stats)


	def local_test(self, use_eval_data=True):
		assert self.model is not None
		flat_para = self.get_flat_model_params()

		num_samples = []
		tot_corrects = []
		losses = []
		for i in range(self.numClients):
			tot_correct, num_sample, loss = self.client.local_test(flat_para, use_eval_data=use_eval_data)
			tot_corrects.append(tot_correct)
			num_samples.append(num_sample)
			losses.append(loss)

		# ids = [c.cid for c in self.clients]
		# groups = [c.group for c in self.clients]

		stats = {'acc': sum(tot_corrects) / sum(num_samples),
		         'loss': sum(losses) / sum(num_samples),
		         'num_samples': num_samples}

		return stats


	def update_server_state(self, average_delta):
		pre_para = self.get_flat_model_params()
		update_para = pre_para + FLAGS.server_lr * average_delta
		self.set_flat_model_params(update_para)


	def set_flat_model_params(self, flat_params):
		set_flat_params_to(self.model, flat_params)


	def get_flat_model_params(self):
		flat_params = get_flat_params_from(self.model)
		return flat_params.detach()


