from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
from keras.datasets import mnist
import argparse
import itertools
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.stats import zipfian

np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})

def count_inversions(permutation):
    """Counts the number of inversions to determine parity."""
    inv = 0
    n = len(permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if permutation[i] > permutation[j]:
                inv += 1
    return inv

def default_args(**kwargs):
	parser = argparse.ArgumentParser()
	parser.add_argument("--epochs", default=250, type=int)
	parser.add_argument("--runs", default=100, type=int)
	parser.add_argument("--number-of-clauses", default=4, type=int)
	parser.add_argument("--number-of-state-bits", default=8, type=int)
	parser.add_argument("--number-of-examples", default=10000, type=int)
	parser.add_argument("--T", default=20, type=int)
	parser.add_argument("--s", default=10.0, type=float)
	parser.add_argument("--q", default=1.0, type=float)
	parser.add_argument("--clause-weight-update-ratio", default=1.0, type=float)
	parser.add_argument("--sequence-length", default=6, type=int)
	parser.add_argument("--number-of-synonyms", default=3, type=int)
	parser.add_argument("--number-of-or-alternatives", default=1, type=int)
	parser.add_argument("--number-of-and-alternatives", default=1, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument('--vanilla', action='store_true')
	parser.add_argument('--and-alternatives', action='store_true')
	parser.add_argument('--and-group-normalization', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')
	parser.add_argument('--weighted-clauses', action='store_true')
	parser.add_argument('--generate-data', action='store_true')
	parser.add_argument('--literal-budget', default=36, type=int)
	parser.add_argument('--append-negated', action='store_true')
	parser.add_argument('--negate-input', action='store_true')
	parser.add_argument("--synonym-skewedness", default=1.2, type=float)

	# Run with: python3 ./examples/SequenceDemo.py --generate-data --weighted-clauses --append-negated

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

if args.generate_data:
	rng = np.random.default_rng(seed=10)

	Y_unique = np.zeros(math.factorial(args.sequence_length), dtype=np.uint32)
	permutations = list(itertools.permutations(range(args.sequence_length)))
	for i in range(math.factorial(args.sequence_length)):
		Y_unique[i] = count_inversions(permutations[i]) % 2

	X_train = np.zeros((args.number_of_examples, args.sequence_length * args.sequence_length * args.number_of_synonyms), dtype=np.uint32)
	Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
	train_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	for i in range(args.number_of_examples):
		for j in range(args.sequence_length):
			X_train[i, j*args.sequence_length*args.number_of_synonyms + permutations[train_indices[i]][j]*args.number_of_synonyms + zipfian.rvs(args.synonym_skewedness, args.number_of_synonyms, size=1)[0] - 1] = 1
		Y_train[i] = Y_unique[train_indices[i]]
	Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

	X_test = np.zeros((args.number_of_examples, args.sequence_length * args.sequence_length * args.number_of_synonyms), dtype=np.uint32)
	Y_test = np.empty(args.number_of_examples, dtype=np.uint32)
	test_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	for i in range(args.number_of_examples):
		for j in range(args.sequence_length):
			X_test[i, j*args.sequence_length*args.number_of_synonyms + permutations[test_indices[i]][j]*args.number_of_synonyms + zipfian.rvs(args.synonym_skewedness, args.number_of_synonyms, size=1)[0] - 1] = 1
		Y_test[i] = Y_unique[test_indices[i]]

	if args.negate_input:
		X_train = 1 - X_train
		X_test = 1 - X_test

	np.savetxt("examples/SequenceTrainingData.txt", np.append(X_train, Y_train.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')
	np.savetxt("examples/SequenceTestingData.txt", np.append(X_test, Y_test.reshape((args.number_of_examples, 1)), axis=1), fmt='%d')
else:
	train_data = np.loadtxt("./examples/SequenceTrainingData.txt").astype(np.uint32)
	X_train = train_data[:,0:-1]
	Y_train = train_data[:,-1]

	test_data = np.loadtxt("./examples/SequenceTestingData.txt").astype(np.uint32)
	X_test = test_data[:,0:-1]
	Y_test = test_data[:,-1]

f = open("sequence_statistics_%d_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d_%.2f_%d_%d_%d_%d_%d_%d_%d.txt" % (args.literal_budget, args.number_of_clauses, args.T, args.s, args.number_of_state_bits, args.vanilla, args.and_group_normalization, args.constant_update_p, args.binary_inference, args.number_of_and_alternatives, args.weighted_clauses, args.noise, args.no_clipping, args.epochs, args.q, args.clause_weight_update_ratio, args.sequence_length, args.append_negated, args.number_of_and_alternatives), "w")

for r in range(args.runs):
	seed = np.random.randint(10000)

	if args.vanilla:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses * args.number_of_and_alternatives * args.number_of_or_alternatives,
			args.T,
			args.s,
			q=args.q,
			clause_weight_update_ratio=args.clause_weight_update_ratio,
			literal_budget=args.literal_budget,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			append_negated=args.append_negated,
			hierarchy_structure=(
				(tm.AND_GROUP, args.sequence_length*args.sequence_length*args.number_of_synonyms),
				(tm.AND_GROUP, 1)
			)
		)
	elif args.and_alternatives:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses* args.number_of_and_alternatives,
			args.T,
			args.s,
			q=args.q,
			clause_weight_update_ratio=args.clause_weight_update_ratio,
			literal_budget=args.literal_budget,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			append_negated=args.append_negated,
			hierarchy_structure=(
				(tm.AND_GROUP, args.number_of_synonyms * args.sequence_length),
				(tm.OR_ALTERNATIVES, args.number_of_or_alternatives),
				(tm.AND_GROUP, args.sequence_length)
			)
		)
	else:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			q=args.q,
			clause_weight_update_ratio=args.clause_weight_update_ratio,
			literal_budget=args.literal_budget,
			weighted_clauses=args.weighted_clauses,
			binary_inference=args.binary_inference,
			constant_update_p=args.constant_update_p,
			and_group_normalization=args.and_group_normalization,
			seed=seed,
			number_of_state_bits=args.number_of_state_bits,
			no_clipping=args.no_clipping,
			append_negated=args.append_negated,
			hierarchy_structure=(
				(tm.AND_GROUP, args.sequence_length),
				(tm.OR_GROUP, args.sequence_length),
				(tm.AND_ALTERNATIVES, args.number_of_and_alternatives)
			)
		)

	print("\nAccuracy over %d epochs:\n" % (args.epochs))

	for e in range(args.epochs):
		start_training = time()
		tsetlin_machine.fit(X_train, Y_train)
		stop_training = time()

		start_testing = time()
		Y_test_pred = tsetlin_machine.predict(X_test)
		stop_testing = time()

		result_test = 100*accuracy_score(Y_test, Y_test_pred)
		precision_test = 100*precision_score(Y_test, Y_test_pred)
		recall_test = 100*recall_score(Y_test, Y_test_pred)

		result_train = 100*(tsetlin_machine.predict(X_train) == Y_train).mean()

		tsetlin_machine.print_hierarchy(print_ta_state=True)

		clause_precision = tsetlin_machine.clause_precision(X_test, Y_test)
		clause_recall = tsetlin_machine.clause_recall(X_test, Y_test)

		for i in range(tsetlin_machine.number_of_clauses):
			print(i, clause_precision[i], clause_recall[i], tsetlin_machine.get_clause_weight(i))

		transformed_X = tsetlin_machine.transform(X_test)
		high_precision_matches = 0
		max_precision = np.zeros(transformed_X.shape[0])
		for i in range(transformed_X.shape[0]):
			high_precision_match = False
			for j in range(tsetlin_machine.number_of_clauses):
				if not high_precision_match and transformed_X[i, j] > 0 and clause_precision[j] >= 1.0:
					high_precision_matches += 1
					high_precision_match = True

				if transformed_X[i, j] > 0 and clause_precision[j] >= max_precision[i]:
					max_precision[i] = clause_precision[j]

		known = 1.0*transformed_X.max(axis=1).sum()/transformed_X.shape[0]

		print("STATISTICS RUN #%d EPOCH #%d\n" % (r+1, e+1))

		print(transformed_X.sum() / transformed_X.shape[0])		
		print(transformed_X.sum() / (transformed_X.shape[0] * tsetlin_machine.number_of_clauses))

		print("Number of unique patterns: %d" % (math.factorial(args.sequence_length) * args.number_of_synonyms**args.sequence_length,))
		print("Number of unique patterns in the training data: %d" % (len(np.unique(X_train, axis=0)),))
		print("Number of unique patterns in the testing data: %d" % (len(np.unique(X_test, axis=0)),))

		unique_train = np.unique(X_train, axis=0)
		unique_test = np.unique(X_test, axis=0)
		combined = np.vstack((unique_train, unique_test))
		_, counts = np.unique(combined, axis=0, return_counts=True)
		overlap_count = np.sum(counts > 1)
		print("Number of unique patterns occuring in both the training and testing data: %d" % (overlap_count,))

		print("#%d/%d Coverage: %.2f%% Test Accuracy: %.2f%% Test Precision: %.2f%% Test Recall: %.2f%% (%.2f%%/%.2f%%/%.2f%%/%.2f%%/%.2f%% %.2f%%) Training Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" %
			(r+1, e+1, 100.0 * known, result_test, precision_test, recall_test, 100.0 * np.percentile(max_precision, 1), 100.0 * np.percentile(max_precision, 10), 100.0 * np.median(max_precision), 100.0 * np.percentile(max_precision, 90), 100.0 * np.percentile(max_precision, 99), 100.0 * high_precision_matches / transformed_X.shape[0], result_train, stop_training-start_training, stop_testing-start_testing))

		print("Clause precision: %.2f%% (p1) %.2f%% (p10) %.2f%% (p50) %.2f%% (p90) %.2f%% (p99)" % (100.0 * np.percentile(clause_precision, 1), 100.0 * np.percentile(clause_precision, 10), 100.0 * np.median(clause_precision), 100.0 * np.percentile(clause_precision, 90), 100.0 * np.percentile(clause_precision, 99)))
		print("Clause recall: %.2f%% (p1) %.2f%% (p10) %.2f%% (p50) %.2f%% (p90) %.2f%% (p99)" % (100.0 * np.percentile(clause_recall, 1), 100.0 * np.percentile(clause_recall, 10), 100.0 * np.median(clause_recall), 100.0 * np.percentile(clause_recall, 90), 100.0 * np.percentile(clause_recall, 99)))

		coverage = transformed_X.sum(axis=1)
		print("Absolute votes per sample (votes, percentage):", end=' ')
		activations = np.unique(coverage, return_counts=True)
		for i in range(len(activations[0])):
			print("(%d,%.2f%%)" % (activations[0][i], 100.0*activations[1][i]/transformed_X.shape[0]), end=' ')
		print()

		f.write("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" % (r, e, result_test, 100.0 * np.percentile(max_precision, 1), 100.0 * np.percentile(max_precision, 10), 100.0 * np.median(max_precision), 100.0 * np.percentile(max_precision, 90), 100.0 * np.percentile(max_precision, 99), 100.0 * high_precision_matches / transformed_X.shape[0], result_train))
		f.flush()
f.close()