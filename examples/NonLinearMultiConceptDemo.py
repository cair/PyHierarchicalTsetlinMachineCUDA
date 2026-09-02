from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=1000, type=int)
	parser.add_argument("--number-of-clauses", default=2, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int)
	parser.add_argument("--T", default=128, type=int)
	parser.add_argument("--s", default=21.1, type=float)
	parser.add_argument("--number-of-alternatives", default=64, type=int)
	parser.add_argument("--number-of-elements", default=16, type=int)
	parser.add_argument("--number-of-copies", default=2, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

features = args.number_of_elements*4

X_train = np.zeros((args.number_of_examples, features), dtype=np.uint32)
Y_train = np.zeros(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
	x = np.random.randint(args.number_of_elements, size=(2))

	for j in range(args.number_of_elements):
		if x[0] == j:
			if np.random.random() <= 0.5:
				X_train[i, x[0]] = 1
				X_train[i, args.number_of_elements + x[0]] = 0
			else:
				X_train[i, x[0]] = 0
				X_train[i, args.number_of_elements + x[0]] = 1
		else:
			if np.random.random() <= 0.5:
				X_train[i, x[0]] = 0
				X_train[i, args.number_of_elements + x[0]] = 0
			else:
				X_train[i, x[0]] = 1
				X_train[i, args.number_of_elements + x[0]] = 1

	for j in range(args.number_of_elements):
		if x[1] == j:
			if np.random.random() <= 0.5:
				X_train[i, 2*args.number_of_elements + x[1]] = 1
				X_train[i, 3*args.number_of_elements + x[1]] = 0
			else:
				X_train[i, 2*args.number_of_elements + x[1]] = 0
				X_train[i, 3*args.number_of_elements + x[1]] = 1
		else:
			if np.random.random() <= 0.5:
				X_train[i, 2*args.number_of_elements + x[1]] = 0
				X_train[i, 3*args.number_of_elements + x[1]] = 0
			else:
				X_train[i, 2*args.number_of_elements + x[1]] = 1
				X_train[i, 3*args.number_of_elements + x[1]] = 1
			

	Y_train[i] = np.logical_xor(x[0] % 2, x[1] % 2)

Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.zeros((args.number_of_examples, features), dtype=np.uint32)
Y_test = np.zeros(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
	x = np.random.randint(args.number_of_elements, size=(2))

	X_test[i, x[0]] = 1
	X_test[i, args.number_of_elements + x[1]] = 1

	Y_test[i] = np.logical_xor(x[0] % 2, x[1] % 2)

tm = MultiClassTsetlinMachine(
	args.number_of_clauses,
	args.T,
	args.s,
	number_of_state_bits=8,
	boost_true_positive_feedback=0,
	hierarchy_structure=(
		(tm.AND_GROUP, features),
		(tm.OR_ALTERNATIVES, args.number_of_alternatives),
		(tm.AND_ALTERNATIVES, args.number_of_copies)
	),
	append_negated=False
)

print("\nAccuracy over %d epochs:\n" % (args.epochs))
for e in range(args.epochs):
	start_training = time()
	tm.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	tm.print_hierarchy()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
