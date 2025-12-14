"""
python implementation of Hilbert Schmidt Independence Criterion
hsic_gam implements the HSIC test using a Gamma approximation
Python 2.7.12

Gretton, A., Fukumizu, K., Teo, C. H., Song, L., Scholkopf, B., 
& Smola, A. J. (2007). A kernel statistical test of independence. 
In Advances in neural information processing systems (pp. 585-592).

Shoubo (shoubo.sub AT gmail.com)
09/11/2016

Inputs:
X 		n by dim_x matrix
Y 		n by dim_y matrix
alph 		level of test

Outputs:
testStat	test statistics
thresh		test threshold for level alpha test
"""

from __future__ import division
import numpy as np
from scipy.stats import gamma
import torch


def pairwise_distances(x):
	#x should be two dimensional
	instances_norm = torch.sum(x**2,-1).reshape((-1,1))
	return -2*torch.mm(x,x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
	pairwise_distances_ = pairwise_distances(x)
	return torch.exp(-pairwise_distances_ /sigma)

def HSIC_loss(x, y, s_x=1, s_y=1):

	# width_x = compute_width(x)
	# width_y = compute_width(y)

	m,_ = x.shape #batch size
	K = GaussianKernelMatrix(x,s_x)
	L = GaussianKernelMatrix(y,s_y)
	H = torch.eye(m) - 1.0/m * torch.ones((m,m))
	# H = H.double().cuda()
	L = L.double().cuda()
	H = H.double().cuda()
	K = K.double().cuda()


	HSIC = torch.trace(torch.mm(L,torch.mm(H,torch.mm(K,H))))/((m-1)**2)

	# 检查梯度连接性
	assert HSIC.requires_grad, "HSIC loss lost gradient connection"

	return HSIC


