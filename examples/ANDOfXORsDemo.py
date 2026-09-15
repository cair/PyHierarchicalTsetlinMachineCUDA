from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=2, type=int)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--number-of-irrelevant-features", default=1, type=int)
	parser.add_argument("--number-of-training-examples", default=10000, type=int)
	parser.add_argument("--number-of-testing-examples", default=1000, type=int)
	parser.add_argument("--T", default=40, type=float)
	parser.add_argument("--s", default=2.5, type=float)
	parser.add_argument("--number-of-alternatives", default=10, type=int)
	parser.add_argument("--number-of-ands", default=4, type=int)
	parser.add_argument("--noise", default=0.01, type=float)
	parser.add_argument('--vanilla', action='store_true')

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

and_factors = np.empty(args.number_of_ands, dtype=np.uint32)

X_train = np.random.randint(2, size=(args.number_of_training_examples, args.number_of_ands*(2 + args.number_of_irrelevant_features)), dtype=np.uint32)
Y_train = np.zeros(args.number_of_training_examples, dtype=np.uint32)

for i in range(args.number_of_training_examples):
	Y_train[i] = np.random.randint(2)
	if Y_train[i] == 1:
		and_factors[:] = 1
		for j in range(args.number_of_ands):
			if np.random.random() <= 0.5:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,1]
			else:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,0]
	else:
		number_of_true_factors = np.random.randint(args.number_of_ands)
		and_factors[:] = 0
		and_factors[:number_of_true_factors] = 1
		np.random.shuffle(and_factors)

	for j in range(args.number_of_ands):
		if and_factors[j]:
			if np.random.random() <= 0.5:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,1]
			else:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,0]
		else:
			if np.random.random() <= 0.5:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,0]
			else:
				X_train[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,1]

Y_train = np.where(np.random.rand(args.number_of_training_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.random.randint(2, size=(args.number_of_testing_examples, args.number_of_ands*(2 + args.number_of_irrelevant_features)), dtype=np.uint32)
Y_test = np.zeros(args.number_of_testing_examples, dtype=np.uint32)

for i in range(args.number_of_testing_examples):
	Y_test[i] = np.random.randint(2)
	if Y_test[i] == 1:
		and_factors[:] = 1
		for j in range(args.number_of_ands):
			if np.random.random() <= 0.5:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,1]
			else:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,0]
	else:
		number_of_true_factors = np.random.randint(args.number_of_ands)
		and_factors[:] = 0
		and_factors[:number_of_true_factors] = 1
		np.random.shuffle(and_factors)

	for j in range(args.number_of_ands):
		if and_factors[j]:
			if np.random.random() <= 0.5:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,1]
			else:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,0]
		else:
			if np.random.random() <= 0.5:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2 ] = [0,0]
			else:
				X_test[i, j * (2 + args.number_of_irrelevant_features):j * (2 + args.number_of_irrelevant_features) + 2] = [1,1]


average_result = 0
for i in range(10):
	if not args.vanilla:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			number_of_state_bits=args.number_of_state_bits,
			hierarchy_structure=(
				(tm.AND_GROUP, 2 + args.number_of_irrelevant_features),
				(tm.OR_ALTERNATIVES, args.number_of_alternatives),
				(tm.AND_GROUP, args.number_of_ands)
			),
			seed=np.random.randint(np.iinfo(np.int32).max)
		)
	else:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			number_of_state_bits=args.number_of_state_bits,
			boost_true_positive_feedback=0,
			hierarchy_structure=(
				(tm.AND_GROUP, (2 + args.number_of_irrelevant_features) * args.number_of_ands),
				(tm.OR_ALTERNATIVES, args.number_of_alternatives)
			),
			seed=np.random.randint(np.iinfo(np.int32).max)
		)

	start_training = time()
	for e in range(args.epochs):
		tsetlin_machine.fit(X_train, Y_train)
	stop_training = time()

	start_testing = time()
	result = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
	stop_testing = time()

	average_result += result / 10.0

	tsetlin_machine.print_hierarchy(print_ta_state=True)

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))

print("\nAverage Accuracy: %.2f%%" % (average_result,))
