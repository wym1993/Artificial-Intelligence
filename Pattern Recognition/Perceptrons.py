import numpy as np
import matplotlib.pyplot as plt

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

def log_loss(label, output):
	"""
	This function calcualte the log loss of the predictions

	label(list): The label for image data
	output(list): The prediction for image data

	"""

	loss = 0.0
	for i in range(output.shape[0]):
		loss += np.log(np.exp(output[i, label[i]]) / np.sum(np.exp(output[i, :])) + 10 ** -12)
	return loss / (-output.shape[0])

def train(data, label_train, test_data, label_test, learning_rate=0.001, weight_type='zeros', data_type='fixed', epoch=30, decay=0.0):
	"""
	This function is the training phase of the perceptron algorithm

	data(list): The training image data
	label_train(list): The label for training image data
	test_data(list): The test image data
	label_test(list): The label for test image data
	learning_rate(float): The learning rate with default 0.001
	weight_type(str): The type of weight matrix initialization
	data_type(str): The variable indicating whether to shuffle the image dataset
	epoch(int): Number of epoches to train
	decay(float): learning rate decay variable

	"""


	if weight_type == 'zeros':
		weight = np.zeros((data.shape[0], 10))
	elif weight_type == 'random':
		weight = np.random.rand(data.shape[0], 10)
	else:
		raise ValueError('Wrong weight type, input onle from [zeros, random]')

	train_loss, test_loss = [], []
	train_acc, test_acc = [], []
	
	for epo in range(epoch):

		if data_type == 'shuffle':
			idx_list = np.array(range(data.shape[1]))
			np.random.shuffle(idx_list)
			data, label_train = data[:, idx_list], label_train[idx_list]
		elif data_type != 'fixed':
			raise ValueError('Wrong data type, input only from [fixed, shuffle]')

		learning_rate *= (1.0 / (1.0 + decay * epo))
		output = []
		for i in range(data.shape[1]):
			one_out = np.dot(weight.T, data[:, i])
			pred = np.argmax(one_out)
			
			if pred != label_train[i]:
				weight[:, label_train[i]] += learning_rate * data[:, i]
				weight[:, pred] -= learning_rate * data[:, i]
			output.append(one_out)

		_, loss, acc = test(data, label_train, weight)
		train_loss.append(loss)
		train_acc.append(acc)

		_, loss, acc = test(test_data, label_test, weight)
		test_loss.append(loss)
		test_acc.append(acc)
		print 'finish ' + str(epo) + ' epochs with train accuracy ' + str(train_acc[-1]) + ' and test accuracy ' + str(test_acc[-1])
	
	result = {'train_loss':train_loss, 'test_loss':test_loss, 'train_acc':train_acc, 'test_acc':test_acc}
		
	return weight, result

def test(data, label, weight):
	"""
	This function tests the image data after the training phase

	data(list): The test image data
	label(list): The test image data label
	weight(list): The weight matrix got from the training phase
	
	"""
	output = np.dot(weight.T, data)
	pred = np.argmax(output, axis=0)
	
	#loss = get_loss(output, label)
	loss = log_loss(label, output.T)
	acc = sum(pred == label) / float(data.shape[1])
	return output, loss, acc


def main():
	use_bias = True
	learning_rate = 0.01
	weight_type = 'zeros' # [zeros, random]
	data_type = 'fixed' # [fixed, shuffle]
	epoch = 30
	decay = 0.1
	train_path = 'digitdata/optdigits-orig_train.txt'
	test_path = 'digitdata/optdigits-orig_test.txt'

	X_train, label_train = read_data(train_path)
	y_train = label_matrix(label_train, 10)

	X_test, label_test = read_data(test_path)
	y_test = label_matrix(label_test, 10)

	assert X_train.shape[1] == y_train.shape[1] and X_test.shape[1] == y_test.shape[1] and X_train.shape[0] == X_test.shape[0]
	#print X_train.shape, y_train.shape, X_test.shape, y_test.shape
	
	if use_bias:
		bias_train, bias_test = np.ones((1, X_train.shape[1])), np.ones((1, X_test.shape[1]))
		X_train = np.concatenate((X_train, bias_train), axis = 0)
		X_test = np.concatenate((X_test, bias_test), axis=0)

	weight, result = train(X_train, label_train, X_test, label_test, weight_type=weight_type, data_type=data_type, epoch=epoch, decay=decay)
	
	plt.figure()
	plt.plot([2,1])
	plt.subplot(211)
	line1 = plt.plot(range(1, epoch + 1), result['train_loss'], 'b', label='train loss')
	line2 = plt.plot(range(1, epoch + 1), result['test_loss'], 'r', label='test loss')
	plt.legend(loc=1)
	plt.ylabel('Train/Test Loss')

	plt.subplot(212)
	line1 = plt.plot(range(1, epoch + 1), result['train_acc'], 'b', label='train acc')
	line2 = plt.plot(range(1, epoch + 1), result['test_acc'], 'r', label='test acc')
	plt.legend(loc=1)
	plt.xlabel('Epoch')
	plt.ylabel('Train/Test Accuracy')
	

	pred, _, acc = test(X_test, label_test, weight)
	print('The overall accuracy is ' + str(acc))
	
	confusion_matrix = np.zeros((10, 10))
	for i in range(len(label_test)):
		confusion_matrix[label_test[i], np.argmax(pred[:, i])] += 1

	for row in range(10):
		s = np.sum(confusion_matrix[row, :])
		for col in range(10):
			confusion_matrix[row, col] /= float(s)

	for line in confusion_matrix:
		for val in line:
			print "{0:.2f} ".format(val),
		print ''

	fig = plt.figure()
	res = plt.imshow(np.array(confusion_matrix), cmap=plt.cm.jet, interpolation='nearest')
	plt.colorbar(res)

	#plt.legend(loc=1)
	plt.show()

if __name__ == '__main__':
	main()



