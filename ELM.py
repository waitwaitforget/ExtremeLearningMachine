# Implementation of Extreme Learning Machine and Oneclass Classifier

import numpy as np 
from scipy.linalg import pinv2


from sklearn.preprocessing import LabelBinarizer

class ELM(object):
	def __init__(self, input_dim, hidden_dim, C):

		self.weight = np.zeros((input_dim, hidden_dim))
		self.bias = np.zeros((hidden_dim, 1))
		self.beta = np.zeros((hidden_dim,))

		self.binarizer=LabelBinarizer(-1, 1)

		self.C = C 
		self._init_weights()

	def _init_weights(self):
		self.weight = np.random.randn(self.weight.shape[0], self.weight.shape[1])
		self.bias = np.random.randn(self.bias.shape[0],)

	def sigmoid(self, x):
		return 1.0 /(1.0 + np.exp(-x))

	def gaussian(self, x):
		return np.exp(-pow(x, 2.0))

	def _compute_input_activations(self, x):
		acts = np.add(np.dot(x, self.weight), self.bias)
		return acts 

	def fit(self, input, target):
		H = self.sigmoid(self._compute_input_activations(input))
		self.classes_ = np.unique(target)

		y_bin = self.binarizer.fit_transform(target)

		self.beta = np.dot(H.T, pinv2(1.0/self.C + np.dot(H, H.T)))
		self.beta = np.dot(self.beta, y_bin)

	def predict(self, input):
		pred = self.sigmoid(self._compute_input_activations(input))

		pred = np.dot(pred, self.beta)
		dist = np.zeros((input.shape[0], len(self.classes_)))

		for i in range(len(self.classes_)):
			dist[:,i] = np.abs(pred - self.classes_[i])[:,0]
		pred = np.argmax(dist,1)
		pred = np.array(self.classes_)[pred]
		return pred


class OCELM(ELM):
	def __init__(self, input_dim, hidden_dim, C, mu):
		super(OCELM, self).__init__(input_dim, hidden_dim, C)
		self.mu = mu

	def fit(self, input, target):
		H = self.sigmoid(self._compute_input_activations(input))
		self.classes_ = np.unique(target)
		assert len(self.classes_)==1, 'target should only has one class'

		y_bin = self.binarizer.fit_transform(target)

		self.beta = np.dot(H.T, pinv2(1.0/self.C + np.dot(H, H.T)))
		self.beta = np.dot(self.beta, y_bin)

		distance  = np.abs(np.add(np.dot(H, self.beta), -target))
		distance  = np.sort(distance, axis=None)

		# set threshold
		N = input.shape[0]
		cutoff = int(np.floor(N * self.mu))
		self.threshold = distance[-cutoff]

	def predict(self, input):
		H = self.sigmoid(self._compute_input_activations(input))
		D = np.dot(H, self.beta) - self.classes_
		print D
		pos_idx = np.where(D < self.threshold)

		pred = np.zeros((input.shape[0],))
		pred[pos_idx[0]] = 1
		return pred


def test():
	x = np.random.randn(100, 6)
	y = np.array([-1]*50 + [1]*50)
	y = y.reshape(100, 1)

	elm = ELM(6, 10, 1)

	elm.fit(x,y)
	#print elm.beta

	testx = np.vstack((np.random.randn(3,6), 10*np.random.randn(2,6)))
	print "prediction: "
	print elm.predict(testx)

if __name__=='__main__':
	test()