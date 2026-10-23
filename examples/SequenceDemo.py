from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist
import argparse
import itertools
import math

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=250, type=int)
	parser.add_argument("--runs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=4, type=int)
	parser.add_argument("--number-of-state-bits", default=10, type=int)
	parser.add_argument("--number-of-examples", default=10, type=int)
	parser.add_argument("--T", default=1000, type=int)
	parser.add_argument("--s", default=10.0, type=float)
	parser.add_argument("--sequence-length", default=3, type=int)
	parser.add_argument("--number-of-or-alternatives", default=20, type=int)
	parser.add_argument("--number-of-and-alternatives", default=4, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument('--vanilla', action='store_true')
	parser.add_argument('--and-alternatives', action='store_true')
	parser.add_argument('--and-group-normalization', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')
	parser.add_argument('--weighted-clauses', action='store_true')
	parser.add_argument('--generate-data', action='store_true')

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

or_alternatives = args.number_of_or_alternatives
clauses = args.number_of_clauses
T = args.T
s = args.s

if args.generate_data:
	X_unique = np.ones((math.factorial(args.sequence_length), args.sequence_length * args.sequence_length), dtype=np.uint32)
	Y_unique = np.empty(math.factorial(args.sequence_length), dtype=np.uint32)

	permutations = itertools.permutations(range(args.sequence_length))
	example = 0
	for p in permutations:
		for i in range(args.sequence_length):
			X_unique[example, i*args.sequence_length + p[i]] = 0

		print(p)
		Y_unique[example] = np.random.randint(2)
		example += 1

	print(X_unique)

	train_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	
	print(train_indices)
	X_train = X_unique[train_indices].copy()
	Y_train = Y_unique[train_indices].copy()
	Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

	print(X_train)
	print(Y_train)
	
	test_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	X_test = X_unique[test_indices].copy()
	Y_test = Y_unique[test_indices].copy()

	np.savetxt("examples/SequenceTrainingData.txt", np.append(X_train, Y_train.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')
	np.savetxt("examples/SequenceTestingData.txt", np.append(X_test, Y_test.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')
else:
	train_data = np.loadtxt("./examples/SequenceTrainingData.txt").astype(np.uint32)
	X_train = train_data[:,0:-1]
	Y_train = train_data[:,-1]

	test_data = np.loadtxt("./examples/SequenceTestingData.txt").astype(np.uint32)
	X_test = test_data[:,0:-1]
	Y_test = test_data[:,-1]

f = open("sequence_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%.2f_%d_%d.txt" % (args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_or_alternatives, args.weighted_clauses, args.noise, args.no_clipping, args.epochs), "w")

for r in range(args.runs):
	seed = np.random.randint(10000)

	if args.vanilla:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses * args.number_of_or_alternatives,
			args.T,
			args.s,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			hierarchy_structure=(
				(tm.AND_GROUP, 28*28*2),
				(tm.AND_GROUP, 1)
			)
		)
	elif args.and_alternatives:
		tsetlin_machine = TsetlinMachine(
			clauses,
			T,
			s,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			hierarchy_structure=(
				(tm.AND_GROUP, 28*28*2),
				(tm.OR_ALTERNATIVES, or_alternatives),
				(tm.AND_ALTERNATIVES, args.number_of_and_alternatives)
			)
		)
	else:
		tsetlin_machine = TsetlinMachine(
			clauses,
			T,
			s,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			hierarchy_structure=(
				(tm.AND_GROUP, 28*28),
				(tm.OR_ALTERNATIVES, or_alternatives),
				(tm.AND_GROUP, 2)
			)
		)

	print("\nAccuracy over %d epochs:\n" % (args.epochs))

	for e in range(args.epochs):
		start_training = time()
		tsetlin_machine.fit(X_train, Y_train)
		stop_training = time()

		start_testing = time()
		result = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
		stop_testing = time()

		print("#%d/%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (r+1, e+1, result, stop_training-start_training, stop_testing-start_testing))

		f.write("%d %d %.2f\n" % (r, e, result))
		f.flush()
f.close()