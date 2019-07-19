import numpy as np
import numpy.random as r

def setup_and_init_weights(nn_structure):
	W = {}
	b = {}

	for l in range(1, len(nn_structure)):
		W[l] = r.random_sample((nn_structure[l], nn_structure[l-1]))
		b[l] = r.random_sample((nn_structure[l],))

	return W, b

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


nn_structure = [64, 30, 10]
#nn_structure = [5, 4, 3]

W, b = setup_and_init_weights(nn_structure)

write_array_of_array_of_values(W, "Weights.txt")
write_array_of_values(b, "Biases.txt")

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