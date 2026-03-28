from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

clauses = 32
s = 15.1

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = TsetlinMachine(clauses, 3000, s, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, 4), (tm.OR_ALTERNATIVES, 10), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 2), (tm.AND_GROUP, 2)))

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	for i in range(clauses):
		print("CLAUSE %d" % (i))
		for j in range(tm.hierarchy_size[1]):
			print("\tComponent %d: " % (j), end= '')

			l = []
			for k in range(tm.number_of_literals_per_leaf):
				if tm.ta_action(i, j, k):
					if k < tm.number_of_literals_per_leaf // 2:
						l.append("x%d" % (k,))
					else:
						l.append("¬x%d" % (k - tm.number_of_literals_per_leaf // 2,))
			print(" ^ ".join(l))

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
