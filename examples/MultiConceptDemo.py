from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=250, type=int)
	parser.add_argument("--number-of-clauses", default=32, type=int)
	parser.add_argument("--number-of-examples", default=5000, type=int)
	parser.add_argument("--T", default=64, type=int)
	parser.add_argument("--s", default=2.1, type=float)
	parser.add_argument("--copies", default=1, type=int)
	parser.add_argument("--number-of-elements", default=2, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

features = args.number_of_elements*2

X_train = np.zeros((args.number_of_examples, features*copies), dtype=np.uint32)
Y_train = np.zeros(args.number_of_examples, dtype=np.uint32)
for i in range(examples):
	x = np.random.randint(elements, size=(2))

	for j in range(copies):
		X_train[i, j*features + x[0]] = 1
		X_train[i, j*features + elements + x[1]] = 1

	Y_train[i] = np.logical_xor(x[0] % 2, x[1] % 2)

Y_train = np.where(np.random.rand(examples) <= noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.zeros((examples, features*copies), dtype=np.uint32)
Y_test = np.zeros(examples, dtype=np.uint32)
for i in range(examples):
	x = np.random.randint(elements, size=(2))

	for j in range(copies):
		X_test[i, j*features + x[0]] = 1
		X_test[i, j*features + elements + x[1]] = 1

	Y_test[i] = np.logical_xor(x[0] % 2, x[1] % 2)

tm = TsetlinMachine(clauses, T, s, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, features), (tm.OR_ALTERNATIVES, alternatives), (tm.AND_GROUP, 1)))

print("\nAccuracy over 1000 epochs:\n")
for e in range(args.epochs):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	tm.print_hierarchy()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
