from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=250, type=int)
	parser.add_argument("--runs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=4, type=int)
	parser.add_argument("--number-of-state-bits", default=10, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int)
	parser.add_argument("--T", default=1000, type=int)
	parser.add_argument("--s", default=10.0, type=float)
	parser.add_argument("--number-of-parity-bits", default=4, type=int)
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
	(X_mnist_train, Y_mnist_train), (X_mnist_test, Y_mnist_test) = mnist.load_data()

	X_mnist_train = np.where(X_mnist_train.reshape((X_mnist_train.shape[0], 28*28)) > 75, 1, 0)
	X_mnist_test = np.where(X_mnist_test.reshape((X_mnist_test.shape[0], 28*28)) > 75, 1, 0)

	x_mnist_train_count = [X_mnist_train[Y_mnist_train == 0].shape[0], X_mnist_train[Y_mnist_train == 1].shape[0]]

	X_train = np.empty((args.number_of_examples, 28*28*args.number_of_parity_bits))
	Y_train = np.empty(args.number_of_examples)
	for i in range(args.number_of_examples):
		x = np.random.randint(2, size=(args.number_of_parity_bits))

		for j in range(args.number_of_parity_bits):
			X_train[i,28*28*j:28*28*(j+1)] = X_mnist_train[Y_mnist_train == x[j]][np.random.randint(x_mnist_train_count[x[j]])]

		Y_train[i] = x.sum() % 2

	np.savetxt("examples/MNISTParityTrainingData.txt", np.append(X_train, Y_train.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')

	X_test = np.empty((args.number_of_examples, 28*28*args.number_of_parity_bits))
	Y_test = np.empty(args.number_of_examples)
	for i in range(args.number_of_examples):
		x = np.random.randint(2, size=(args.number_of_parity_bits))

		for j in range(args.number_of_parity_bits):
			X_test[i,28*28*j:28*28*(j+1)] = X_mnist_train[Y_mnist_train == x[j]][np.random.randint(x_mnist_train_count[x[j]])]

		Y_test[i] = x.sum() % 2

	np.savetxt("examples/MNISTParityTestingData.txt", np.append(X_train, Y_train.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')


else:
	train_data = np.loadtxt("./examples/MNISTParityTrainingData.txt").astype(np.uint32)
	X_train = train_data[:,0:-1]
	Y_train = train_data[:,-1]

	test_data = np.loadtxt("./examples/MNISTParityTestingData.txt").astype(np.uint32)
	X_test = test_data[:,0:-1]
	Y_test = test_data[:,-1]

f = open("mnist_parity_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%.2f_%d_%d.txt" % (args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_or_alternatives, args.weighted_clauses, args.noise, args.no_clipping, args.epochs), "w")

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
				(tm.AND_GROUP, 28*28*args.number_of_parity_bits),
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
				(tm.AND_GROUP, 28*28*args.number_of_parity_bits),
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
				(tm.AND_GROUP, 2),
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