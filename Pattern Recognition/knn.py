import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import time

def read_data(filename):
	"""
	This function read image data as well as labels from the file

	filename(str): The filename for image and labels

	"""
	data, label = [], []
	with open(filename, 'r') as f:
		while True:
			single = [int(c) for c in f.readline().strip()]
			if single:
				for _ in range(31):
					single += [int(c) for c in f.readline().strip()]
				data.append(single)
				label.append(int(f.readline().strip()))
			else:
				break

	return np.array(data).T, np.array(label)

def label_matrix(label, n = 10):
	"""
	This function changes the label array to one-hot format

	label(list): The array of labels
	n(int): The number of classes with default as 10

	"""
	
	m = len(label)
	matrix = np.zeros((m, n))
	for i in range(m):
		for j in range(n):
			if j == label[i]:
				matrix[i][j] = 1
			else:
				matrix[i][j] = -1

	return matrix.T

def manhattan_distance(arr1, arr2):
	"""
	This function calculates the manhattan distance between two arrays

	arr1(list): image data
	arr2(list): image data

	"""
	return np.sum(np.logical_and(arr1>0, arr2>0))

def square_distance(arr1, arr2):
	"""
	This function calculates the mean-square distance between two arrays

	arr1(list): image data
	arr2(list): image data

	"""

	return np.sqrt(np.sum(np.square(arr1 > 0 - arr2 > 0)))

def find_neighbors(data, test, dis_func):
	"""
	This function finds the sorting neighbors for every test examples

	data(list): The training image data
	test(list): The test image data
	dis_func(function): The method used to calculate the distance

	"""
	
	nei = []
	ave_time = []
	for i in range(test.shape[1]):
		dis_list = []
		start = time.time()
		for j in range(data.shape[1]):
			dis = dis_func(data[:, j], test[:, i])
			dis_list.append((j, dis))
			#assert dis[-1][1] == sum(data[:, j]!=test[:, i])
		#nei.append([0] * 10)
		nei.append([j for j, _ in sorted(dis_list, key = lambda x: x[1], reverse=True)])
		#print i, pred[-1]
		end = time.time()
		ave_time.append(end - start)
		
	print 'Time for single query is ' + str(sum(ave_time) / float(len(ave_time)))

	return nei

def get_label(test, label, nei, k):
	"""
	This function finds the label for each test example based on the nearest neighbors find from prevous function

	test(list): The test image data
	label(list): The label for training image data
	nei(list): The sorted neighbors for each test example
	k(int): The value of k nearest neighbors
	
	"""
	pred = np.zeros((test.shape[1], 10))
	for i in range(test.shape[1]):
		for j in nei[i][:k]:
			pred[i, label[j]] += 1

	return pred


def main():
	train_path = 'digitdata/optdigits-orig_train.txt'
	test_path = 'digitdata/optdigits-orig_test.txt'
	k_range = range(1, 50)

	X_train, label_train = read_data(train_path)
	X_test, label_test = read_data(test_path)

	man_func = lambda x, y: manhattan_distance(x, y)
	squ_func = lambda x, y: square_distance(x, y)

	nei = find_neighbors(X_train, X_test, man_func)
	
	acc_list = []
	for k in k_range:
		pred = get_label(X_test, label_train, nei, k)
		
		acc_list.append(sum(np.argmax(pred, axis=1)==label_test) / float(X_test.shape[1]))
		#print ('k is ' + str(k) + ' and accuray is ' + str(acc_list[-1]))
	
	# Best choice of k is 3
	k = 3
	confusion_matrix = np.zeros((10, 10))
	pred = get_label(X_test, label_train, nei, k)
	print 'The test accuracy at k = ' + str(k) + ' is ' + str(sum(np.argmax(pred, axis=1)==label_test) / float(X_test.shape[1]))
	
	for i in range(len(label_test)):
		confusion_matrix[label_test[i], np.argmax(pred[i, :])] += 1

	for row in range(10):
		s = np.sum(confusion_matrix[row, :])
		for col in range(10):
			confusion_matrix[row, col] /= float(s)

	for line in confusion_matrix:
		for val in line:
			print "{0:.2f} ".format(val),
		print ''

	fig = plt.figure()
	plt.plot(k_range, acc_list, 'r')
	plt.xlabel('k')
	plt.ylabel('accuracy')
	fig = plt.figure()
	res = plt.imshow(np.array(confusion_matrix), cmap=plt.cm.jet, interpolation='nearest')
	plt.colorbar(res)
	plt.show()
	
if __name__ == '__main__':
	main()
