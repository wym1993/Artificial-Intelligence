import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy


class DNN(object):
	def __init__(self, learning_rate=0.05, shuffle=True, epochs=600, batch_size=32, plot_loss=True):
		self.learning_rate = learning_rate
		self.shuffle = shuffle
		self.epochs = epochs
		self.batch_size = batch_size
		self.W, self.b = None, None
		self.plot_loss = plot_loss
		#self.hidden_units = [256, 256, 256]
		self.hidden_units = [256, 256]
		self.loss_list, self.acc_list = [], []

	def set_hidden_units(self, layer, num):
		if layer >= len(self.hidden_units):
			raise Exception('Wrong layer changes')
		self.hidden_units[layer] = num

	def initial_weight(self, n_x, n_y):
		self.W, self.b = [], []

		weight_units = [n_x] + self.hidden_units + [n_y]
		for i in range(len(weight_units) - 1):
			self.W.append(np.random.rand(weight_units[i], weight_units[i+1])*0.01)
			self.b.append(np.zeros((1, weight_units[i+1])))

	def relu(self, Z):
		return Z*(Z>0)

	def relu_derivative(self, dA, Z):
		dZ = dA * (Z > 0)

		return dZ

	def softmax(self, F):
		A = np.zeros(F.shape)
		for i in range(F.shape[0]):
			s = np.sum(np.exp(F[i, :]))
			for j in range(F.shape[1]):
				A[i, j] = np.exp(F[i, j]) / s
		return A

	def cross_entropy_loss(self, F, Y):
		pred = self.softmax(F)

		L = 0.0
		for i in range(Y.shape[0]):
			L += np.sum(np.multiply(np.log(pred[i, :]), Y[i, :]))
		return L / float(-1 * Y.shape[0])
	
	def cross_entropy_derivative(self, F, Y):
		
		dF = np.zeros(F.shape)
		for i in range(F.shape[0]):
			idx = list(Y[i, :]).index(1)
			s = np.sum(np.exp(F[i, :]))
			dF[i, :] = (Y[i, :] - np.exp(F[i, :])) / float(-1 * s * F.shape[0])
		
		return dF

	def model_forward(self, X):

		Z1 = np.dot(X, self.W[0]) + self.b[0]
		A1 = self.relu(Z1)

		Z2 = np.dot(A1, self.W[1]) + self.b[1]
		A2 = self.relu(Z2)

		Z3 = np.dot(A2, self.W[2]) + self.b[2]
		A3 = self.relu(Z3)

		#F = np.dot(A3, self.W[3]) + self.b[3]
		F = np.dot(A2, self.W[2]) + self.b[2]
		
		#A_list = [X, A1, A2, A3]
		#Z_list = [Z1, Z2, Z3, F]
		A_list = [X, A1, A2]
		Z_list = [Z1, Z2, F]

		return F, Z_list, A_list

	def model_backward(self, Y, Z_list, A_list):
		dW_list, db_list = [], []
		
		#dF = self.cross_entropy_derivative(Z_list[3], Y)
		#dA3 = np.dot(dF, self.W[3].T)
		#dW4 = np.dot(A_list[3].T, dF)
		#db4 = np.sum(dF, axis=0, keepdims=True)

		#dZ3 = self.relu_derivative(dA3, Z_list[2])
		#dA2 = np.dot(dZ3, self.W[2].T)
		#dW3 = np.dot(A_list[2].T, dZ3)
		#db3 = np.sum(dZ3, axis=0, keepdims=True)
		dF = self.cross_entropy_derivative(Z_list[2], Y)
		dA2 = np.dot(dF, self.W[2].T)
		dW3 = np.dot(A_list[2].T, dF)
		db3 = np.sum(dF, axis=0, keepdims=True)

		dZ2 = self.relu_derivative(dA2, Z_list[1])
		dA1 = np.dot(dZ2, self.W[1].T)
		dW2 = np.dot(A_list[1].T, dZ2)
		db2 = np.sum(dZ2, axis=0, keepdims=True)

		dZ1 = self.relu_derivative(dA1, Z_list[0])
		dX = np.dot(dZ1, self.W[0].T)
		dW1 = np.dot(A_list[0].T, dZ1)
		db1 = np.sum(dZ1, axis=0, keepdims=True)

		#dW_list = [dW1, dW2, dW3, dW4]
		#db_list = [db1, db2, db3, db4]
		dW_list = [dW1, dW2, dW3]
		db_list = [db1, db2, db3]

		return dW_list, db_list

	def update_parameters(self, dW_list, db_list):
		for i in range(len(dW_list)):
			self.W[i] -= dW_list[i] * self.learning_rate
			self.b[i] -= db_list[i] * self.learning_rate

	def fit(self, X, Y):

		# Initial weight
		self.initial_weight(X.shape[1], Y.shape[1])
		

		for epoch in range(self.epochs):

			# Shuffle the data/label matrix
			if self.shuffle:
				idx_list = np.array(range(X.shape[0]))
				np.random.shuffle(idx_list)
				X = X[idx_list]
				Y = Y[idx_list]

			# Do forwarding one batch at a time
			num_batch = X.shape[0] // self.batch_size + 1
			for batch_idx in range(num_batch):
				start, end = batch_idx*self.batch_size, (batch_idx+1)*self.batch_size
				batch_X, batch_Y = X[start:end, :], Y[start:end, :]
				if batch_X.shape[0] == 0:
					break
				
				F, Z_list, A_list = self.model_forward(batch_X)
				dW_list, db_list = self.model_backward(batch_Y, Z_list, A_list)
				self.update_parameters(dW_list, db_list)

			F, _, _ = self.model_forward(X)
			loss = self.cross_entropy_loss(F, Y)
			self.loss_list.append(loss)
			acc = self.test(X, Y)
			self.acc_list.append(acc)

			if (epoch + 1) % 10 == 0:
				print('Finish epoch ' + str(epoch + 1) + ' with loss ' + str(loss) + ' and prediction accuracy is ' + str(acc))

		if self.plot_loss:
			self.plot_loss_curve()

		#with open('loss.txt', 'w') as f:
		#	f.write(','.join(map(str, self.loss_list)))

		#with open('acc.txt', 'w') as f:
		#	f.write(','.join(map(str, self.acc_list)))

		return

	def plot_loss_curve(self):
		plt.plot(range(len(self.loss_list)), self.loss_list)
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.title('Train loss over epoches')
		plt.show()
		
	def test(self, X, Y):
		pred = self.predict(X)
		acc = np.sum(np.argmax(pred, axis=1) == np.argmax(Y, axis=1)) / float(X.shape[0])
		
		return acc

	def predict(self, X):
		F, _, _ = self.model_forward(X)
		pred = self.softmax(F)

		return pred


	    