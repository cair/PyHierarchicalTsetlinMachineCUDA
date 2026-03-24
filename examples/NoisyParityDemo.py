from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time


ensembles = 10
epochs = 4000

train_data = np.loadtxt("examples/NoisyXORTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(32, 1500, 40.1,  hierarchy_structure=((AND_GROUP, 3), (OR_ALTERNATIVES, 10), (AND_GROUP, 2), (OR_ALTERNATIVES, 2), (AND_GROUP, 2)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
