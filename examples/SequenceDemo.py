# Copyright (c) 2026 Ole-Christoffer Granmo and the University of Agder

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
	parser.add_argument('--synonym-distribution-drift', action='store_true')
	parser.add_argument('--sequence-distribution-drift', action='store_true')
	parser.add_argument("--synonym-skewedness", default=0.0, type=float)
	parser.add_argument("--sequence-skewedness", default=0.0, type=float)
	parser.add_argument("--print-hierarchy", action='store_true')

	# Run with: python3 ./examples/SequenceDemo.py --generate-data --weighted-clauses --append-negated

	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

config_dict = vars(args)

ext_parts = [f"{v}" for v in sorted(config_dict.values())]
raw_extension = "_".join(ext_parts)

f = open("sequence_statistics_" + raw_extension + ".txt", "w")

if args.generate_data:
	rng = np.random.default_rng(seed=10)

	Y_unique = np.zeros(math.factorial(args.sequence_length), dtype=np.uint32)
	permutations = list(itertools.permutations(range(args.sequence_length)))
	for i in range(math.factorial(args.sequence_length)):
		Y_unique[i] = count_inversions(permutations[i]) % 2

	X_train = np.zeros((args.number_of_examples, args.sequence_length * args.sequence_length * args.number_of_synonyms), dtype=np.uint32)
	Y_train = np.empty(args.number_of_examples, dtype=np.uint32)
	
	if args.sequence_skewedness == 0.0:
		train_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	else:
		train_indices = zipfian.rvs(a=args.sequence_skewedness, loc=-1, n=math.factorial(args.sequence_length), size=args.number_of_examples)

	for i in range(args.number_of_examples):
		for j in range(args.sequence_length):
			X_train[i, j*args.sequence_length*args.number_of_synonyms + permutations[train_indices[i]][j]*args.number_of_synonyms + zipfian.rvs(a=args.synonym_skewedness, n=args.number_of_synonyms, loc=-1, size=1)[0]] = 1
		Y_train[i] = Y_unique[train_indices[i]]
	Y_train = np.where(np.random.rand(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

	X_test = np.zeros((args.number_of_examples, args.sequence_length * args.sequence_length * args.number_of_synonyms), dtype=np.uint32)
	Y_test = np.empty(args.number_of_examples, dtype=np.uint32)

	if args.sequence_skewedness == 0.0:
		test_indices = np.random.choice(math.factorial(args.sequence_length), size=args.number_of_examples, replace=True)
	else:
		test_indices = zipfian.rvs(a=args.sequence_skewedness, loc=-1, n=math.factorial(args.sequence_length), size=args.number_of_examples)
	
	if args.sequence_distribution_drift:
		test_indices = math.factorial(args.sequence_length) - 1 - test_indices

	for i in range(args.number_of_examples):
		for j in range(args.sequence_length):
			if args.synonym_distribution_drift:
				X_test[i, j*args.sequence_length*args.number_of_synonyms + permutations[test_indices[i]][j]*args.number_of_synonyms + args.number_of_synonyms - 1 - zipfian.rvs(args.synonym_skewedness, args.number_of_synonyms, loc=-1, size=1)[0]] = 1
			else:
				X_test[i, j*args.sequence_length*args.number_of_synonyms + permutations[test_indices[i]][j]*args.number_of_synonyms + zipfian.rvs(args.synonym_skewedness, args.number_of_synonyms, loc=-1, size=1)[0]] = 1

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
		result_train = 100*(tsetlin_machine.predict(X_train) == Y_train).mean()

		clause_precision_test = tsetlin_machine.clause_precision(X_test, Y_test)
		clause_recall_test = tsetlin_machine.clause_recall(X_test, Y_test)

		transformed_X_test = tsetlin_machine.transform(X_test)

		print("CLAUSES RUN #%d EPOCH #%d\n" % (r+1, e+1))

		if args.print_hierarchy:
			tsetlin_machine.print_hierarchy(print_ta_state=True, precision_recall = (clause_precision_test, clause_recall_test))
			print()

		print("STATISTICS RUN #%d EPOCH #%d\n" % (r+1, e+1))

		print("Number of unique patterns: %d" % (math.factorial(args.sequence_length) * args.number_of_synonyms**args.sequence_length,))
		print("Number of unique patterns in the training data: %d" % (len(np.unique(X_train, axis=0)),))
		print("Number of unique patterns in the testing data: %d" % (len(np.unique(X_test, axis=0)),))

		coverage = transformed_X_test.sum(axis=1)
		activations = np.unique(coverage, return_counts=True)

		unique_train = np.unique(X_train, axis=0)
		unique_test = np.unique(X_test, axis=0)
		combined = np.vstack((unique_train, unique_test))
		_, counts = np.unique(combined, axis=0, return_counts=True)
		overlap_count = np.sum(counts > 1)
		print("Number of unique patterns occurring in both the training and testing data: %d" % (overlap_count,))

		print("Number of unseen patterns in the testing data: %d" % (len(np.unique(X_test, axis=0)) - overlap_count,))

		print()

		pattern_frequency_train = np.unique(X_train, axis=0, return_counts=True)[1] / X_train.shape[0]
		print("Training pattern frequency: %.3f%% (1p) %.3f%% (5p)  %.3f%% (50p) %.3f%% (95p) %.3f%% (99p)" % (100.0 * np.percentile(pattern_frequency_train, 1), 100.0 * np.percentile(pattern_frequency_train, 5), 100.0 * np.percentile(pattern_frequency_train, 50), 100.0 * np.percentile(pattern_frequency_train, 95), 100.0 * np.percentile(pattern_frequency_train, 99)))

		print()

		print("Percentage of test data covered by at least one clause: %.2f%%" % (100.0 * (1.0 - (activations[1][0]/transformed_X_test.shape[0])),))
		print()
		print("Testing accuracy: %.2f%%" % (result_test,))
		print("Training accuracy: %.2f%%" % (result_train,))
		
		print()

		print("Testing time: %.2fs" % (stop_testing-start_testing,))
		print("Training time: %.2fs" % (stop_training-start_training,))

		print()

		print("Testing clause precision: %.1f%% (p1) %.1f%% (p5) %.1f%% (p50) %.1f%% (p95) %.1f%% (p99)" % (100.0 * np.percentile(clause_precision_test, 1), 100.0 * np.percentile(clause_precision_test, 5), 100.0 * np.percentile(clause_precision_test, 50), 100.0 * np.percentile(clause_precision_test, 95), 100.0 * np.percentile(clause_precision_test, 99)))
		print("Testing clause recall: %.1f%% (p1) %.1f%% (p5) %.1f%% (p50) %.1f%% (p95) %.1f%% (p99)" % (100.0 * np.percentile(clause_recall_test, 1), 100.0 * np.percentile(clause_recall_test, 5), 100.0 * np.percentile(clause_recall_test, 50), 100.0 * np.percentile(clause_recall_test, 95), 100.0 * np.percentile(clause_recall_test, 99)))

		print()

		print("Mean testing clause precision: %.1f%%" % (100.0*clause_precision_test.mean()))
		print("Mean testing clause recall: %.1f%%" % (100.0*clause_recall_test.mean()))

		print()

		max_example_precision = np.zeros(transformed_X_test.shape[0])
		for i in range(transformed_X_test.shape[0]):
			for j in range(args.number_of_clauses):
				if transformed_X_test[i, j] > 0 and clause_precision_test[j] >= max_example_precision[i]:
					max_example_precision[i] = clause_precision_test[j]

		print("Max clause precision per test sample: %.1f%% (p1) %.1f%% (p5) %.1f%% (p50) %.1f%% (p95) %.1f%% (p99)" % (100.0 * np.percentile(max_example_precision, 1), 100.0 * np.percentile(max_example_precision, 5), 100.0 * np.percentile(max_example_precision, 50), 100.0 * np.percentile(max_example_precision, 95), 100.0 * np.percentile(max_example_precision, 99)))

		print()

		print("Absolute votes per test sample (votes, percentage):", end=' ')
		for i in range(len(activations[0])):
			print("(%d,%.2f%%)" % (activations[0][i], 100.0*activations[1][i]/transformed_X_test.shape[0]), end=' ')
		print()
		print()
		print("Mean absolute votes per sample: %.1f" % (transformed_X_test.sum() / transformed_X_test.shape[0]))

		f.write("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" %
			(
				r,
				e,
				100.0 * (1.0 - (activations[1][0]/transformed_X_test.shape[0])),
				result_test,
				result_train,
				100.0 * np.percentile(clause_precision_test, 1),
				100.0 * np.percentile(clause_precision_test, 10),
				100.0 * np.median(clause_precision_test),
				100.0 * np.percentile(clause_precision_test, 90),
				100.0 * np.percentile(clause_precision_test, 99),
				100.0 * np.percentile(clause_recall_test, 1),
				100.0 * np.percentile(clause_recall_test, 10),
				100.0 * np.median(clause_recall_test),
				100.0 * np.percentile(clause_recall_test, 90),
				100.0 * np.percentile(clause_recall_test, 99),
				100.0*clause_precision_test.mean(),
				100.0*clause_recall_test.mean(),
				100.0 * np.percentile(max_example_precision, 1),
				100.0 * np.percentile(max_example_precision, 10),
				100.0 * np.median(max_example_precision),
				100.0 * np.percentile(max_example_precision, 90),
				100.0 * np.percentile(max_example_precision, 99),
				transformed_X_test.sum() / transformed_X_test.shape[0]
			)
		)
		f.flush()
f.close()