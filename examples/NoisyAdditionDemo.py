from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

noise = 0.0
number_of_addends = 4
number_of_examples = 10000

X_train = np.random.randint(2, size=(number_of_examples, number_of_addends)).astype(np.int32)
Y_train = X_train.sum(axis=1)
Y_train = np.where(np.random.rand(number_of_examples) <= noise, Y_train + 1, Y_train) # Adds noise

X_test = np.random.randint(2, size=(number_of_examples, number_of_addends)).astype(np.int32)
Y_test = X_test.sum(axis=1)

#tm = MultiClassTsetlinMachine(100, 15*64*10, 20.0, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, number_of_values), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 8), (tm.AND_GROUP, 2)))
tm = MultiClassTsetlinMachine(10, 15, 4.1, number_of_state_bits=8, boost_true_positive_feedback=1, hierarchy_structure=((tm.AND_GROUP, 2), (tm.AND_GROUP, 2)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
