import torch

from absl import flags

FLAGS = flags.FLAGS

def krum(x):
	'''
	defense: Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent
	:param x: list of model parameter of workers
	:return: estimation of the mean
	'''
	score = []
	# index = 3
	index = FLAGS.clients_per_round - int(FLAGS.clients_per_round * FLAGS.attack_percentage) -2
	for i, (i_samples, i_grad) in enumerate(x):
		dist = []
		for j, (j_samples, j_grad) in enumerate(x):
			if i != j:
				dist.append(torch.dist(i_grad, j_grad))
		dist.sort(reverse=False)
		score.append(sum(dist[0:index]))
	min_index = score.index(min(score))
	return x[min_index][1]

def coordinate_median(x):
	'''

	:param x:
	:return:
	'''
	xList = []
	for num_samples, x_grad in x:
		xList.append(x_grad)
	xList = torch.stack(xList)
	return torch.median(xList, dim=0).values




# x = []
# for i in range(10):
# 	x.append([i, torch.randn(4)])
# print(x)
# print(coordinate_median(x))
