import torch
from absl import flags
import numpy as np

FLAGS = flags.FLAGS

def norm_attack(x, num_std=1.5):
	'''
	Compute the attack vector adapted from A Little is Enough: Circumventing Defenses For Distributed Learning
	:param x: list of model parameter of workers
	:param num_std: hyper-parameters
	:return:
	'''
	xList = []
	for num_samples, x_grad in x:
		xList.append(x_grad)
	xList = torch.stack(xList)
	grad_mean = torch.mean(xList, axis=0)
	grad_std = torch.std(xList, axis=0)
	attack_vector = grad_mean - num_std * grad_std
	for i in range(int(FLAGS.clients_per_round * FLAGS.attack_percentage)):
		x[i][1] = attack_vector

def dimension_attack(x, para=1.0):
	'''

	:param x:
	:param para:
	:return:
	'''
	xList = []
	for num_samples, x_grad in x:
		xList.append(x_grad)
	xList = torch.stack(xList)
	grad_mean = torch.mean(xList, axis=0)
	for i in range(int(FLAGS.clients_per_round * FLAGS.attack_percentage)):
		attack_vector = torch.zeros_like(grad_mean)
		j = np.random.randint(0, len(grad_mean))
		attack_vector[j] = para * np.sqrt(len(grad_mean))
		x[i][1] = attack_vector

def hybrid_attack(x, para=1.0, num_std=1.5):
	'''

	:param x:
	:param para:
	:return:
	'''
	xList = []
	for num_samples, x_grad in x:
		xList.append(x_grad)
	xList = torch.stack(xList)
	grad_mean = torch.mean(xList, axis=0)
	grad_std = torch.std(xList, axis=0)
	attack_vector = grad_mean - num_std * grad_std

	for i in range(int(FLAGS.clients_per_round * FLAGS.attack_percentage)):
		if i <= int(FLAGS.clients_per_round * FLAGS.attack_percentage / 2.0):
			x[i][1] = attack_vector
		else:
			attack_vector_1 = torch.zeros_like(grad_mean)
			j = np.random.randint(0, len(grad_mean))
			attack_vector_1[j] = para * np.sqrt(len(grad_mean))
			x[i][1] = attack_vector_1


# x = []
# for i in range(10):
# 	x.append([i, torch.randn(4)])
# print(x)
# print(np.random.randint(0,5))

