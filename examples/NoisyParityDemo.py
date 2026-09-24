from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=1000, type=int)
	parser.add_argument("--runs", default=10, type=int)
	parser.add_argument("--number-of-clauses", default=32, type=int)
	parser.add_argument("--number-of-state-bits", default=7, type=int)
	parser.add_argument("--T", default=250, type=int)
	parser.add_argument("--s", default=25.0, type=float)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument("--number-of-alternatives-1", default=3, type=int)
	parser.add_argument("--number-of-alternatives-2", default=3, type=int)
	parser.add_argument('--vanilla', action='store_true')
	parser.add_argument('--and-group-normalization', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

f = open("noisy_parity_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%d_%d.txt" % (args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_alternatives_1, args.number_of_alternatives_2, args.no_clipping, args.epochs), "w")

for r in range(args.runs):
	seed = np.random.randint(10000)
	if args.vanilla:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses * 3 * args.number_of_alternatives_1 * 2 * args.number_of_alternatives_2 * 2 / 12,
			args.T,
			args.s,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed, number_of_state_bits=args.number_of_state_bits,
			boost_true_positive_feedback=0,
			no_clipping=args.no_clipping,
			hierarchy_structure=(
				(tm.AND_GROUP, 12),
				(tm.AND_GROUP, 1)
			)
		)
	else:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			boost_true_positive_feedback=0,
			no_clipping=args.no_clipping,
			hierarchy_structure=(
				(tm.AND_GROUP, 3),
				(tm.OR_ALTERNATIVES, args.number_of_alternatives_1),
				(tm.AND_GROUP, 2),
				(tm.OR_ALTERNATIVES, args.number_of_alternatives_2),
				(tm.AND_GROUP, 2)
			)
		)

	print("\nAccuracy over %d epochs:\n" % (args.epochs,))

	for e in range(args.epochs):
		start_training = time()
		tsetlin_machine.fit(X_train, Y_train)
		stop_training = time()

		start_testing = time()
		result = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
		stop_testing = time()

		tsetlin_machine.print_hierarchy()

		print("\n#%d/%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (r+1, e+1, result, stop_training-start_training, stop_testing-start_testing))

		f.write("%d %d %.2f\n" % (r, e, result))
		f.flush()

f.close()