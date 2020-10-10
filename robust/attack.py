import torch
from absl import flags

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



# x = []
# for i in range(10):
# 	x.append([i, torch.randn(4)])
# print(x)

