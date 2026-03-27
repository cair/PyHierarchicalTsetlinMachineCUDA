from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

number_of_values = 2
noise = 0.0
number_of_addends = 2
number_of_examples = 10000

X_train_integer = np.random.randint(number_of_values, size=(number_of_examples, number_of_addends), dtype=np.int32)
Y_train = X_train_integer.sum(axis=1)
X_train = np.zeros((number_of_examples, number_of_addends*number_of_values), dtype=np.int32)
for i in range(number_of_examples):
	for j in range(number_of_addends):
		X_train[i, j*number_of_values + X_train_integer[i, j]] = 1

Y_train = np.where(np.random.rand(number_of_examples) <= noise, Y_train + 1, Y_train) # Adds noise

X_test_integer = np.random.randint(number_of_values, size=(number_of_examples, number_of_addends), dtype=np.int32)
Y_test = X_test_integer.sum(axis=1)
X_test = np.zeros((number_of_examples, number_of_addends*number_of_values), dtype=np.int32)
for i in range(number_of_examples):
	for j in range(number_of_addends):
		X_test[i, j*number_of_values + X_test_integer[i, j]] = 1

tm = MultiClassTsetlinMachine(200, 300, 2.5, number_of_state_bits=8, boost_true_positive_feedback=1, hierarchy_structure=((tm.AND_GROUP, 10), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 20), (tm.AND_GROUP, 1)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
