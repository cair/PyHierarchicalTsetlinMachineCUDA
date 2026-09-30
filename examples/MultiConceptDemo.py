from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=100, type=int)
	parser.add_argument("--runs", default=10, type=int)
	parser.add_argument("--number-of-clauses", default=2, type=int)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int)
	parser.add_argument("--T", default=128, type=int)
	parser.add_argument("--s", default=21.1, type=float)
	parser.add_argument("--number-of-alternatives", default=64, type=int)
	parser.add_argument("--number-of-elements", default=16, type=int)
	parser.add_argument("--number-of-concepts", default=2, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument('--vanilla', action='store_true')
	parser.add_argument('--and-group-normalization', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

features = args.number_of_elements*2

X_train = np.zeros((args.number_of_examples, features), dtype=np.uint32)
Y_train = np.zeros(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
	x = np.random.randint(args.number_of_elements, size=(2))

	X_train[i, x[0]] = 1
	X_train[i, args.number_of_elements + x[1]] = 1

	Y_train[i] = np.logical_xor(x[0] % 2, x[1] % 2)

Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.zeros((args.number_of_examples, features), dtype=np.uint32)
Y_test = np.zeros(args.number_of_examples, dtype=np.uint32)
for i in range(args.number_of_examples):
	x = np.random.randint(args.number_of_elements, size=(2))

	X_test[i, x[0]] = 1
	X_test[i, args.number_of_elements + x[1]] = 1

	Y_test[i] = np.logical_xor(x[0] % 2, x[1] % 2)

f = open("multi_concept_statistics_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%.2f_%d.txt" % (args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_alternatives, args.number_of_elements, args.noise, args.no_clipping), "w")

for r in range(args.runs):
	seed = np.random.randint(10000)
	if args.vanilla:
		tsetlin_machine = MultiClassTsetlinMachine(
			args.number_of_clauses * args.number_of_alternatives * args.number_of_concepts,
			args.T,
			args.s,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			boost_true_positive_feedback=0,
			no_clipping=args.no_clipping,
			append_negated=False,
			hierarchy_structure=(
				(tm.AND_GROUP, features),
				(tm.AND_GROUP, 1)
			)
		)
	else:
		tsetlin_machine = MultiClassTsetlinMachine(
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
			append_negated=False,

			hierarchy_structure=(
				(tm.AND_GROUP, features),
				(tm.OR_ALTERNATIVES, args.number_of_alternatives),
				(tm.AND_ALTERNATIVES, args.number_of_concepts)
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
