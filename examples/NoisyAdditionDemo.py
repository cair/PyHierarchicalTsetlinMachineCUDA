from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

noise = 0.0
number_of_addends = 2
examples = 10

X_train_integer = np.randomrandint(10, size=(examples, addends))
Y_train = X_train_integers.sum(axis=0)

print(X_train_integer)
print(Y_tain)

tm = TsetlinMachine(32, 3000, 30.1, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, 10), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 10), (tm.AND_GROUP, 1)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
