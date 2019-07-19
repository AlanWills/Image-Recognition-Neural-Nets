from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt

def convert_y_to_vect(y):
	y_vect = np.zeros((len(y), 10))
	
	for i in range(len(y)):
		y_vect[i, y[i]] = 1
	return y_vect

def f(x):
	return 1 / (1 + np.exp(-x))

def f_deriv(x):
	return f(x) * (1 - f(x))

def setup_and_init_weights(nn_structure):
	W = {}
	b = {}

	#for l in range(1, len(nn_structure)):
		#W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
		#b[l] = r.random_sample((nn_structure[l],))

	#return W, b

	weights_file = open("Weights.txt")
	weights = {}

	for l in range(1, len(nn_structure)):
		weights[l] = np.zeros((nn_structure[l], nn_structure[l-1]))

		line_count = 0
		while line_count < nn_structure[l]:
			line = weights_file.readline()
			fields = line.strip().split()
			weights[l][line_count] = np.asarray([float(i) for i in fields])
			line_count += 1

	biases_file = open("Biases.txt")
	biases = {}

	for l in range(1, len(nn_structure)):
		biases[l] = np.zeros((nn_structure[l],))

		line = biases_file.readline()
		fields = line.strip().split()
		biases[l] = np.asarray([float(i) for i in fields])

	return weights, biases



def init_tri_values(nn_structure):
	tri_W = {}
	tri_b = {}

	for l in range(1, len(nn_structure)):
		tri_W[l] = np.zeros((nn_structure[l], nn_structure[l-1]))
		tri_b[l] = np.zeros((nn_structure[l],))

	return tri_W, tri_b

def feed_forward(x, W, b):
	h = {1: x}
	z = {}

	for l in range(1, len(W) + 1):
		if l == 1:
			node_in = x
		else:
			node_in = h[l]

		z[l + 1] = W[l].dot(node_in) + b[l]
		h[l + 1] = f(z[l + 1])

	return h, z

def calculate_out_later_delta(y, h_out, z_out):
	return -(y - h_out) * f_deriv(z_out)

def calculate_hidden_delta(delta_plus_1, w_l, z_l):
	return np.dot(np.transpose(w_l), delta_plus_1) * f_deriv(z_l)

def train_nn(nn_structure, X, y, iter_num=500, alpha=0.25):
	W, b = setup_and_init_weights(nn_structure)

	# Can early exit if we have trained values and we want to just evaluate the NN
	#return W, b

	cnt = 0
	m = len(y)
	one_over_m = 1.0 / m

	print('Starting gradient descent for {} iterations'.format(iter_num))

	while cnt < iter_num:
		if cnt % 100 == 0:
			print('Iteration {} of {}'.format(cnt, iter_num))

		tri_W, tri_b = init_tri_values(nn_structure)

		for i in range(len(y)):
			delta = {}

			h, z = feed_forward(X[i, :], W, b)

			for l in range(len(nn_structure), 0, -1):
				if l == len(nn_structure):
					delta[l] = calculate_out_later_delta(y[i,:], h[l], z[l])
				else:
					if l > 1:
						delta[l] = calculate_hidden_delta(delta[l + 1], W[l], z[l])

					tri_W[l] += np.dot(delta[l + 1][:, np.newaxis], np.transpose(h[l][:, np.newaxis]))
					tri_b[l] += delta[l + 1]

		for l in range(len(nn_structure) - 1, 0, -1):
			W[l] += -alpha * (one_over_m * tri_W[l])
			b[l] += -alpha * (one_over_m * tri_b[l])

		cnt += 1

	return W, b

def predict_y(W, b, X, n_layers):
	m = X.shape[0]
	y = np.zeros((m,))

	for i in range(m):
		h, z = feed_forward(X[i, :], W, b)
		y[i] = np.argmax(h[n_layers])

	return y

def write_array_of_array_of_values(array_of_array_of_values, file):
	file = open(file, 'w')

	for array_index in array_of_array_of_values:
		for array in array_of_array_of_values[array_index]:
			for value in array:
				file.write(str(value))
				file.write(" ")

			file.write('\n')

	file.close()

def write_array_of_values(array_of_values, file):
	file = open(file, 'w')

	for value_index in array_of_values:
		for value in array_of_values[value_index]:
			file.write(str(value))
			file.write(" ")

		file.write('\n')

	file.close()


digits = load_digits()
X_scale = StandardScaler()

X = X_scale.fit_transform(digits.data)
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
y_v_train = convert_y_to_vect(y_train)
y_v_test = convert_y_to_vect(y_test)

nn_structure = [64, 30, 10]

W, b = train_nn(nn_structure, X_train, y_v_train)

y_pred = predict_y(W, b, X_test, 3)
print(accuracy_score(y_test, y_pred) * 100)

write_array_of_array_of_values(W, "Weights.txt")
print("Finished writing Weights.txt")

write_array_of_values(b, "Biases.txt")
print("Finished writing Biases.txt")