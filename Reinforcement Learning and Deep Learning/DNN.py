import numpy as np

class DNN(object):
	def __init__(self, learning_rate=0.1, batch_size=32, epochs=30, shuffle=True):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.layers = 3
		self.num_units = [10, 10]
		self.epochs = epochs
		self.shuffle = shuffle
		self.W_list, self.b_list = [], []

	def build_weight(self, X, Y):
		num_features = len(X[0])
		num_output = len(set(Y))
		self.num_units = [num_features] + self.num_units + [num_output]
		W_list, b_list = [], []
		for i in range(self.layers):
			W_list.append([[0]*self.num_units[i+1] for _ in range(self.num_units[i])])
			b_list.append([[0]*self.num_units[i+1]])
		return W_list, b_list

	def relu(self, Z):
		return [max(0, z) for z in Z]

	def cross_entropy(self, F, y):
		L = 0.0
		for i in range(len(y)):
			k = y[i].index(1)
			cur = 0
			for j in range(len(F[i])):
				cur += np.exp(F[i][j])

			L += F[i][k] - np.log(cur)

		return -1.0 * L / n

	def cross_entropy_differentiation(self, F, y):
		dF = [[0.0]*len(F[0]) for _ in range(len(F))]
		n = len(F)
		for i in range(len(F)):
			idx = F[i].index(1)
			s = sum([np.exp(F[i][j]) for j in range(len(F[i]))])

			for j in range(len(F[0])):
				if j == idx:
					dF[i][j] = -(1.0/n) * (1 - np.exp(F[i][j])/s) 
				else:
					dF[i][j] = np.exp(F[i][j]) / s

		return dF

	def affine(self, X, W, b):
		n, l = len(b), len(W)
		Z = [[0]*n for _ in range(m)]
		for i in range(len(X)):
			for j in range(n):
				for k in range(l):
					Z[i][j] += X[i][k]*W[k][j] + b[j]
		return Z

	def forward_propagation(self, X):
		Z, A = None, X
		for i in range(self.layers):
			Z = affine(A, self.W_list[i], self.b_list[i])
			A = relu(Z)

		return A

	def label_to_one_hot(self, Y):
		y_set = set(Y)
		label = [[0]*len(y_set) for _ in range(len(Y))]
		for i in range(len(Y)):
			label[i][Y[i]] = 1

		return label

	def backward_propagation(self):
		pass

	def train(self, X, Y):
		label = self.label_to_one_hot(Y)
		self.W_list, self.b_list = self.build_weight(X, Y)

		for epoch in self.epochs:
			if shuffle:
				idx_list = range(len(X))
				np.random.shuffle(idx_list)
				X = [X[idx] for idx in idx_list]
				label = [label[idx] for idx in idx_list]

			for idx_batch in range(len(X) // self.batch_size + 1):
				start_idx = idx_batch*self.batch_size
				batch_X = X[start_idx:start_idx+self.batch_size]
				batch_label = label[start_idx:start_idx+self.batch_size]
				if not batch_X:
					break

				F = self.forward_propagation(batch_X)
				#loss = self.cross_entropy(F, batch_label)

				dF = self.cross_entropy_differentiation(F, batch_label)

				# back propagation










