from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

board_dim = 4

clauses = 1000
s = 50.0
T = 1500

data = np.loadtxt("./examples/hex_data.txt").astype(np.uint32)
X_train = data[:int(len(data)*0.8),0:-1]
Y_train = data[:int(len(data)*0.8),-1]

X_test = data[int(len(data)*0.8):,0:-1]
Y_test = data[int(len(data)*0.8):,-1]

#tsetlin_machine = TsetlinMachine(clauses, T, s, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, 25), (tm.OR_ALTERNATIVES, 1), (tm.AND_GROUP, 4)))
tsetlin_machine = TsetlinMachine(clauses, T, s, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, board_dim * board_dim), (tm.AND_GROUP, 1)))

print("\nAccuracy over 1000 epochs:\n")
for e in range(1000):
	start_training = time()
	tsetlin_machine.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
	stop_testing = time()

	#tm.print_hierarchy()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
