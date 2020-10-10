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
# x = torch.rand(3) >= 0.1
# print(x)
# y = [(i, i+1) for i in range(3)]
# print(y)
# z = {'one':1, 'two':2, 'three':3}
# print(z)
# # print((x, y, z))
# for i, j, k in zip(x, y, z):
# 	print(0)
# 	print(i, j, k)

# print((x==0).sum() / float(len(x)))
