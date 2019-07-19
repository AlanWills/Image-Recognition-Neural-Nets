import numpy as np
import timeit

w1 = np.array([[0.2, 0.2, 0.2], [0.4, 0.4, 0.4], [0.6, 0.6, 0.6]])
w2 = np.zeros((1, 3))
w2[0,:] = np.array([0.5, 0.5, 0.5])
b1 = np.array([0.8, 0.8, 0.8])
b2 = np.array([0.2])

def f(x):
	return 1 / (1 + np.exp(-x))

def simple_looped_nn_calc(n_layers, x, w, b):
	for l in range(n_layers - 1):
		if l == 0:
			node_in = x
		else:
			node_in = h

		h = np.zeros((w[l].shape[0],))

		for i in range(w[l].shape[0]):
			f_sum = 0

			for j in range(w[l].shape[1]):
				f_sum += w[l][i][j] * node_in[j]

			f_sum += b[l][i]
			h[i] = f(f_sum)

	return h

def matrix_feed_forward_calc(n_layers, x, w, b):
	for l in range(n_layers - 1):
		if l == 0:
			node_in = x
		else:
			node_in = h

		z = w[l].dot(node_in) + b[l]
		h = f(z)

	return h


w = [w1, w2]
b = [b1, b2]
x = [1.5, 2.0, 3.0]

print(matrix_feed_forward_calc(3, x, w, b))