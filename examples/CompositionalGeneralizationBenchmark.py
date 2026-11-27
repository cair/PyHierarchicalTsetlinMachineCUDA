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
import argparse
import itertools
import math
from sklearn.metrics import precision_score, recall_score, accuracy_score
from scipy.stats import zipfian
import math

np.set_printoptions(formatter={'float_kind': '{:.2f}'.format})

def count_inversions(permutation, relation_range):
    """Counts the number of inversions to determine parity."""
    inv = 0
    n = len(permutation)
    for i in range(n):
        for j in range(i + 1, n):
            if (permutation[i] > permutation[j]) and (abs(permutation[i] - permutation[j]) <= relation_range):
                inv += 1
    return inv

def get_missing_row_indices(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # Ensure both arrays are contiguous in memory and have the exact same dtype
    A = np.ascontiguousarray(A)
    B = np.ascontiguousarray(B, dtype=A.dtype)
    
    # Create a 1D void dtype representing an entire row's bytes
    row_dtype = np.dtype((np.void, A.dtype.itemsize * A.shape[1]))
    
    # View the 2D matrices as 1D arrays of rows
    A_view = A.view(row_dtype).ravel()
    B_view = B.view(row_dtype).ravel()
    
    # Find rows in A that are NOT in B and return their indices
    mask = ~np.isin(A_view, B_view)
    return np.flatnonzero(mask)

# Data statistics (constant across runs and epochs, so calculated once)
def unique_statistics(train, test):
	"""Returns (unique in train, unique in test, unique in both, unique in test only), counting identical rows once."""
	unique_train = np.unique(train, axis=0)
	unique_test = np.unique(test, axis=0)
	_, counts = np.unique(np.vstack((unique_train, unique_test)), axis=0, return_counts=True)
	overlap = int(np.sum(counts > 1))
	return len(unique_train), len(unique_test), overlap, len(unique_test) - overlap

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
	parser.add_argument("--type-ia-ii-ratio-clauses", default=1.0, type=float)
	parser.add_argument("--type-ia-ii-ratio-weights", default=1.0, type=float)
	parser.add_argument("--clause-weight-update-ratio", default=1.0, type=float)
	parser.add_argument("--number-of-informative-symbols", default=6, type=int)
	parser.add_argument("--number-of-non-informative-symbols", default=0, type=int)
	parser.add_argument("--number-of-synonyms", default=3, type=int)
	parser.add_argument("--non-informative-vocabulary-size", default=3, type=int)
	parser.add_argument("--number-of-or-alternatives", default=1, type=int)
	parser.add_argument("--number-of-or-alternatives-interior", default=1, type=int)
	parser.add_argument("--noise", default=0.0, type=float)
	parser.add_argument("--number-of-missing-training-permutations", default=0, type=int)
	parser.add_argument("--constant-update-p", action='store_true')
	parser.add_argument('--binary-inference', action='store_true')
	parser.add_argument('--vanilla', action='store_true')
	parser.add_argument('--hierarchical-rotations', action='store_true')
	parser.add_argument('--and-group-normalization', action='store_true')
	parser.add_argument('--no-clipping', action='store_true')
	parser.add_argument('--weighted-clauses', action='store_true')
	parser.add_argument('--literal-budget', default=None, type=int)
	parser.add_argument('--append-negated', action='store_true')
	parser.add_argument('--negate-input', action='store_true')
	parser.add_argument('--extended-statistics', action='store_true')
	parser.add_argument('--relation-range', default=None, type=int)
	parser.add_argument('--hierarchical', action='store_true')
	parser.add_argument('--synonym-distribution-drift', action='store_true')
	parser.add_argument('--sequence-distribution-drift', action='store_true')
	parser.add_argument("--non-informative-vocabulary-skewedness", default=0.0, type=float)
	parser.add_argument("--synonym-skewedness", default=0.0, type=float)
	parser.add_argument("--sequence-skewedness", default=0.0, type=float)
	parser.add_argument("--skewed-test-data-only", action='store_true')
	parser.add_argument("--print-hierarchy", action='store_true')
	parser.add_argument("--data-seed", default=36, type=int)
	parser.add_argument("--run-seed", default=37, type=int)
	parser.add_argument("--resample-seed", default=38, type=int)
	parser.add_argument("--print-clause-activation-statistics", action='store_true')
	sampling = parser.add_mutually_exclusive_group()
	sampling.add_argument("--resample", action='store_true')
	sampling.add_argument("--shuffle", action='store_true')
	
	args = parser.parse_args()
	for key, value in kwargs.items():
		if key in args.__dict__:
			setattr(args, key, value)
	return args

args = default_args()

config_dict = vars(args)

ext_parts = [f"{v}" for v in config_dict.values()]
raw_extension = "_".join(ext_parts)

rng_data = np.random.default_rng(seed=args.data_seed)

f = open(f"compositional_generalization_statistics_{raw_extension}.txt", "w")

sequence_length = args.number_of_informative_symbols + args.number_of_non_informative_symbols
number_of_features_per_sequence_element = args.non_informative_vocabulary_size + args.number_of_informative_symbols*args.number_of_synonyms
number_of_features_per_sequence_element_informative = args.number_of_informative_symbols*args.number_of_synonyms

if args.relation_range == None:
	args.relation_range = args.number_of_informative_symbols

assert args.relation_range >= 1

Y_unique = np.zeros(math.factorial(args.number_of_informative_symbols), dtype=np.uint32)
permutations = list(itertools.permutations(range(args.number_of_informative_symbols)))
rng_data.shuffle(permutations)
for i in range(math.factorial(args.number_of_informative_symbols)):
	Y_unique[i] = count_inversions(permutations[i], args.relation_range) % 2

X_train = np.zeros((args.number_of_examples, sequence_length * number_of_features_per_sequence_element), dtype=np.uint32)
X_train_informative = np.zeros((args.number_of_examples, sequence_length * number_of_features_per_sequence_element), dtype=np.uint32)
Y_train = np.empty(args.number_of_examples, dtype=np.uint32)

if args.sequence_skewedness == 0.0 or args.skewed_test_data_only:
	train_indices = rng_data.choice(math.factorial(args.number_of_informative_symbols) - args.number_of_missing_training_permutations, size=args.number_of_examples, replace=True)
else:
	train_indices = zipfian.rvs(a=args.sequence_skewedness, loc=-1, n=math.factorial(args.number_of_informative_symbols) - args.number_of_missing_training_permutations, size=args.number_of_examples, random_state=rng_data)

non_informative_vocabulary_skewedness = 0.0 if args.skewed_test_data_only else args.non_informative_vocabulary_skewedness
synonym_skewedness = 0.0 if args.skewed_test_data_only else args.synonym_skewedness

for i in range(args.number_of_examples):
	position_of_informative_symbols = np.sort(rng_data.choice(np.arange(sequence_length), size=args.number_of_informative_symbols, replace=False))
	position_of_non_informative_symbol = np.arange(sequence_length)[~np.isin(np.arange(sequence_length), position_of_informative_symbols)]
	
	for j in range(args.number_of_informative_symbols):
		# Finds the slot for the current informative symbol, and then the synonym inside that slot
		position_inside_sequence_element = args.non_informative_vocabulary_size + permutations[train_indices[i]][j]*args.number_of_synonyms + zipfian.rvs(a=synonym_skewedness, n=args.number_of_synonyms, loc=-1, size=1, random_state=rng_data)[0]
		X_train_informative[i, position_of_informative_symbols[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1
		X_train[i, position_of_informative_symbols[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1

	for j in range(args.number_of_non_informative_symbols):
		position_inside_sequence_element = zipfian.rvs(a=non_informative_vocabulary_skewedness, n=args.non_informative_vocabulary_size, loc=-1, size=1, random_state=rng_data)[0]
		X_train[i, position_of_non_informative_symbol[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1

	Y_train[i] = Y_unique[train_indices[i]]

Y_train = np.where(rng_data.random(args.number_of_examples) <= args.noise, 1 - Y_train, Y_train)  # Adds noise

X_test = np.zeros((args.number_of_examples, sequence_length*number_of_features_per_sequence_element), dtype=np.uint32)
X_test_informative = np.zeros((args.number_of_examples, sequence_length*number_of_features_per_sequence_element), dtype=np.uint32)
Y_test = np.empty(args.number_of_examples, dtype=np.uint32)

if args.sequence_skewedness == 0.0:
	test_indices = rng_data.choice(math.factorial(args.number_of_informative_symbols), size=args.number_of_examples, replace=True)
else:
	test_indices = zipfian.rvs(a=args.sequence_skewedness, loc=-1, n=math.factorial(args.number_of_informative_symbols), size=args.number_of_examples, random_state=rng_data)

if args.sequence_distribution_drift:
	test_indices = math.factorial(args.number_of_informative_symbols) - 1 - test_indices

for i in range(args.number_of_examples):
	position_of_informative_symbols = np.sort(rng_data.choice(np.arange(sequence_length), size=args.number_of_informative_symbols, replace=False))
	position_of_non_informative_symbols = np.arange(sequence_length)[~np.isin(np.arange(sequence_length), position_of_informative_symbols)]
	
	for j in range(args.number_of_informative_symbols):
		# Finds the slot for the current informative symbol
		position_inside_sequence_element = permutations[test_indices[i]][j] * args.number_of_synonyms

		# Finds the synonym inside the slot
		if args.synonym_distribution_drift:
			position_inside_sequence_element += args.number_of_synonyms - 1 - zipfian.rvs(args.synonym_skewedness, args.number_of_synonyms, loc=-1, size=1, random_state=rng_data)[0]
		else:
			position_inside_sequence_element += zipfian.rvs(a=args.synonym_skewedness, n=args.number_of_synonyms, loc=-1, size=1, random_state=rng_data)[0]

		position_inside_sequence_element += args.non_informative_vocabulary_size

		# Inserts the synonym inside the correct sequence slot
		X_test_informative[i, position_of_informative_symbols[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1

		X_test[i, position_of_informative_symbols[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1

	for j in range(args.number_of_non_informative_symbols):
		position_inside_sequence_element = zipfian.rvs(a=args.non_informative_vocabulary_skewedness, n=args.non_informative_vocabulary_size, loc=-1, size=1, random_state=rng_data)[0]
		X_test[i, position_of_non_informative_symbols[j]*number_of_features_per_sequence_element + position_inside_sequence_element] = 1

	Y_test[i] = Y_unique[test_indices[i]]

permutations_held_out_from_training = test_indices >= math.factorial(args.number_of_informative_symbols) - args.number_of_missing_training_permutations
synonym_sequences_held_out_from_training = get_missing_row_indices(X_test_informative, X_train_informative)
synonym_sequences_held_out_from_training = synonym_sequences_held_out_from_training[~permutations_held_out_from_training[synonym_sequences_held_out_from_training]]

if args.negate_input:
	X_train = 1 - X_train
	X_test = 1 - X_test

number_of_unique_examples = math.factorial(sequence_length) // math.factorial(args.number_of_non_informative_symbols) * args.number_of_synonyms**args.number_of_informative_symbols * args.non_informative_vocabulary_size**args.number_of_non_informative_symbols
(
	number_of_unique_training_examples,
	number_of_unique_testing_examples,
	number_of_overlapping_training_and_testing_examples,
	number_of_unseen_testing_examples
) = unique_statistics(X_train, X_test)

number_of_unique_patterns = math.perm(sequence_length, args.number_of_informative_symbols) * args.number_of_synonyms**args.number_of_informative_symbols
(
	number_of_unique_training_patterns,
	number_of_unique_testing_patterns,
	number_of_overlapping_training_and_testing_patterns,
	number_of_unseen_testing_patterns
) = unique_statistics(X_train_informative, X_test_informative)

pattern_frequency_train = np.unique(X_train_informative, axis=0, return_counts=True)[1] / X_train_informative.shape[0]
pattern_frequency_train_percentiles = 100.0 * np.percentile(pattern_frequency_train, [1, 5, 50, 95, 99])

rng_run = np.random.default_rng(seed=args.run_seed)
rng_resample = np.random.default_rng(seed=args.resample_seed)

if args.hierarchical_rotations:
	X_train = np.concatenate(
		(
			X_train,
			np.roll(X_train, shift=2*number_of_features_per_sequence_element, axis=1),
			np.roll(X_train, shift=4*number_of_features_per_sequence_element, axis=1)
		),
		axis=1
	)

	X_test = np.concatenate(
		(
			X_test,
			np.roll(X_test, shift=2*number_of_features_per_sequence_element, axis=1),
			np.roll(X_test, shift=4*number_of_features_per_sequence_element, axis=1),
		),
		axis=1
	)

for r in range(args.runs):
	seed = rng_run.integers(10000)

	if args.vanilla:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses * args.number_of_or_alternatives * args.number_of_or_alternatives_interior,
			args.T,
			args.s,
			q=args.q,
			type_ia_ii_ratio_clauses = args.type_ia_ii_ratio_clauses,
			type_ia_ii_ratio_weights = args.type_ia_ii_ratio_weights,
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
				(tm.AND_GROUP, sequence_length * number_of_features_per_sequence_element),
				(tm.AND_GROUP, 1)
			)
		)
	elif args.hierarchical:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			q=args.q,
			type_ia_ii_ratio_clauses = args.type_ia_ii_ratio_clauses,
			type_ia_ii_ratio_weights = args.type_ia_ii_ratio_weights,
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
				(tm.AND_GROUP, number_of_features_per_sequence_element),
				(tm.OR_ALTERNATIVES, args.number_of_or_alternatives),
				(tm.AND_GROUP, sequence_length // 2),
				(tm.OR_ALTERNATIVES, args.number_of_or_alternatives_interior),
				(tm.AND_GROUP, 2)
			)
		)
	elif args.hierarchical_rotations:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			q=args.q,
			type_ia_ii_ratio_clauses = args.type_ia_ii_ratio_clauses,
			type_ia_ii_ratio_weights = args.type_ia_ii_ratio_weights,
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
				(tm.AND_GROUP, number_of_features_per_sequence_element),
				(tm.AND_GROUP, sequence_length // 2),
				(tm.OR_ALTERNATIVES, args.number_of_or_alternatives_interior),
				(tm.AND_GROUP, 2),
				(tm.OR_GROUP_CONVOLUTION, 3)
			)
		)
	else:
		tsetlin_machine = TsetlinMachine(
			args.number_of_clauses,
			args.T,
			args.s,
			q=args.q,
			type_ia_ii_ratio_clauses = args.type_ia_ii_ratio_clauses,
			type_ia_ii_ratio_weights = args.type_ia_ii_ratio_weights,
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
				(tm.AND_GROUP, number_of_features_per_sequence_element),
				(tm.OR_ALTERNATIVES, args.number_of_or_alternatives),
				(tm.AND_GROUP, sequence_length)
			)
		)

	print("\nAccuracy over %d epochs:\n" % (args.epochs))

	for e in range(args.epochs):
		if args.resample:
			sample_indices = rng_resample.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
		elif args.shuffle:
			sample_indices = rng_resample.permutation(X_train.shape[0])
		else:
			sample_indices = np.arange(X_train.shape[0])

		start_training = time()
		tsetlin_machine.fit(X_train[sample_indices], Y_train[sample_indices])
		stop_training = time()

		start_testing = time()
		Y_test_pred = tsetlin_machine.predict(X_test)
		stop_testing = time()

		result_test = 100*accuracy_score(Y_test, Y_test_pred)
		result_test_held_out_permutations = 100 * (Y_test_pred[permutations_held_out_from_training] == Y_test[permutations_held_out_from_training]).mean()
		result_test_held_out_synonym_sequences = 100 * (Y_test_pred[synonym_sequences_held_out_from_training] == Y_test[synonym_sequences_held_out_from_training]).mean()
		result_train = 100*(tsetlin_machine.predict(X_train) == Y_train).mean()

		print("CLAUSES RUN #%d EPOCH #%d\n" % (r+1, e+1))

		print("STATISTICS RUN #%d EPOCH #%d\n" % (r+1, e+1))

		print("Number of unique examples: %d" % (number_of_unique_examples,))
		print("Number of unique examples in the training data: %d" % (number_of_unique_training_examples,))
		print("Number of unique examples in the testing data: %d" % (number_of_unique_testing_examples,))
		print("Number of unique examples occurring in both the training and testing data: %d" % (number_of_overlapping_training_and_testing_examples,))
		print("Number of unseen examples in the testing data: %d" % (number_of_unseen_testing_examples,))

		print()

		print("Number of unique patterns: %d" % (number_of_unique_patterns,))
		print("Number of unique patterns in the training data: %d" % (number_of_unique_training_patterns,))
		print("Number of unique patterns in the testing data: %d" % (number_of_unique_testing_patterns,))
		print("Number of unique patterns occurring in both the training and testing data: %d" % (number_of_overlapping_training_and_testing_patterns,))
		print("Number of unseen patterns in the testing data: %d" % (number_of_unseen_testing_patterns,))

		print()

		print("Training pattern frequency: %.3f%% (1p) %.3f%% (5p) %.3f%% (50p) %.3f%% (95p) %.3f%% (99p)" % tuple(pattern_frequency_train_percentiles))

		print()

		print("Testing accuracy: %.2f%%" % (result_test,))
		if args.number_of_missing_training_permutations > 0:
			print("Testing accuracy on held out permutations: %.2f%%" % (result_test_held_out_permutations,))
		if len(synonym_sequences_held_out_from_training) > 0:
			print("Testing accuracy on held out synonym sequences: %.2f%%" % (result_test_held_out_synonym_sequences,))
		print("Training accuracy: %.2f%%" % (result_train,))
		
		print()

		print("Testing time: %.2fs" % (stop_testing-start_testing,))
		print("Training time: %.2fs" % (stop_training-start_training,))

		print()

		if args.extended_statistics:
			clause_precision_test = tsetlin_machine.clause_precision(X_test, Y_test)
			clause_recall_test = tsetlin_machine.clause_recall(X_test, Y_test)

			clause_precision_train = tsetlin_machine.clause_precision(X_train, Y_train)
			clause_recall_train = tsetlin_machine.clause_recall(X_train, Y_train)

			transformed_X_test = tsetlin_machine.transform(X_test)

			coverage = transformed_X_test.sum(axis=1)
			activations = np.unique(np.ceil(coverage), return_counts=True)

			if activations[0][0] == 0:
				print("Percentage of test data covered by at least one clause: %.2f%%" % (100.0 * (1.0 - (activations[1][0]/transformed_X_test.shape[0])),))
			else:
				print("Percentage of test data covered by at least one clause: 100%")
			
			print()

			print("Testing clause precision: %.1f%% (p1) %.1f%% (p5) %.1f%% (p10) %.1f%% (p50) %.1f%% (p90) %.1f%% (p95) %.1f%% (p99)" % (
				100.0 * np.percentile(clause_precision_test, 1),
				100.0 * np.percentile(clause_precision_test, 5),
				100.0 * np.percentile(clause_precision_test, 10),
				100.0 * np.percentile(clause_precision_test, 50),
				100.0 * np.percentile(clause_precision_test, 90),
				100.0 * np.percentile(clause_precision_test, 95),
				100.0 * np.percentile(clause_precision_test, 99))
			)
			print("Testing clause recall: %.1f%% (p1) %.1f%% (p5) %.1f%% (p10) %.1f%% (p50) %.1f%% (p90) %.1f%% (p95) %.1f%% (p99)" % (
				100.0 * np.percentile(clause_recall_test, 1),
				100.0 * np.percentile(clause_recall_test, 5),
				100.0 * np.percentile(clause_recall_test, 10),
				100.0 * np.percentile(clause_recall_test, 50),
				100.0 * np.percentile(clause_recall_test, 90),
				100.0 * np.percentile(clause_recall_test, 95),
				100.0 * np.percentile(clause_recall_test, 99))
			)

			print()

			print("Mean testing clause precision: %.1f%%" % (100.0*clause_precision_test.mean()))
			print("Mean testing clause recall: %.1f%%" % (100.0*clause_recall_test.mean()))

			print()

			print("Training clause precision: %.1f%% (p1) %.1f%% (p5) %.1f%% (p10) %.1f%% (p50) %.1f%% (p90) %.1f%% (p95) %.1f%% (p99)" % (
				100.0 * np.percentile(clause_precision_train, 1),
				100.0 * np.percentile(clause_precision_train, 5),
				100.0 * np.percentile(clause_precision_train, 10),
				100.0 * np.percentile(clause_precision_train, 50),
				100.0 * np.percentile(clause_precision_train, 90),
				100.0 * np.percentile(clause_precision_train, 95),
				100.0 * np.percentile(clause_precision_train, 99))
			)

			print("Training clause recall: %.1f%% (p1) %.1f%% (p5) %.1f%% (p10) %.1f%% (p50) %.1f%% (p90) %.1f%% (p95) %.1f%% (p99)" % (
				100.0 * np.percentile(clause_recall_train, 1),
				100.0 * np.percentile(clause_recall_train, 5),
				100.0 * np.percentile(clause_recall_train, 10),
				100.0 * np.percentile(clause_recall_train, 50),
				100.0 * np.percentile(clause_recall_train, 90),
				100.0 * np.percentile(clause_recall_train, 95),
				100.0 * np.percentile(clause_recall_train, 99))
			)

			print()

			print("Mean training clause precision: %.1f%%" % (100.0*clause_precision_train.mean()))
			print("Mean training clause recall: %.1f%%" % (100.0*clause_recall_train.mean()))

			print()

			max_example_precision = np.zeros(transformed_X_test.shape[0])
			for i in range(transformed_X_test.shape[0]):
				for j in range(tsetlin_machine.number_of_clauses):
					if transformed_X_test[i, j] > 0 and clause_precision_test[j] >= max_example_precision[i]:
						max_example_precision[i] = clause_precision_test[j]

			print("Max clause precision per test sample: %.1f%% (p1) %.1f%% (p5) %.1f%% (p10) %.1f%% (p50) %.1f%% (p90) %.1f%% (p95) %.1f%% (p99)" % (
				100.0 * np.percentile(max_example_precision, 1),
				100.0 * np.percentile(max_example_precision, 5),
				100.0 * np.percentile(max_example_precision, 10),
				100.0 * np.percentile(max_example_precision, 50),
				100.0 * np.percentile(max_example_precision, 90),
				100.0 * np.percentile(max_example_precision, 95),
				100.0 * np.percentile(max_example_precision, 99))
			)

			print()
			print("Mean absolute votes per sample: %.1f" % (transformed_X_test.sum() / transformed_X_test.shape[0]))
			print()

		if args.print_clause_activation_statistics and args.extended_statistics:
			print()
			print("Absolute votes per test sample (votes, percentage):", end=' ')
			for i in range(len(activations[0])):
				print("(%d,%.2f%%)" % (activations[0][i], 100.0*activations[1][i]/transformed_X_test.shape[0]), end=' ')
			print()

		if args.print_hierarchy and args.extended_statistics:
			tsetlin_machine.print_hierarchy(print_ta_state=True, precision_recall = (clause_precision_test, clause_recall_test))
			print()

		if args.extended_statistics:
			f.write("%d %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n" %
				(
					r,
					e,
					result_test,
					result_test_held_out_permutations,
					result_test_held_out_synonym_sequences,
					result_train,
					100.0 * np.percentile(clause_precision_test, 1),
					100.0 * np.percentile(clause_precision_test, 5),
					100.0 * np.percentile(clause_precision_test, 10),
					100.0 * np.median(clause_precision_test),
					100.0 * np.percentile(clause_precision_test, 90),
					100.0 * np.percentile(clause_precision_test, 95),
					100.0 * np.percentile(clause_precision_test, 99),
					100.0 * np.percentile(clause_recall_test, 1),
					100.0 * np.percentile(clause_recall_test, 5),
					100.0 * np.percentile(clause_recall_test, 10),
					100.0 * np.median(clause_recall_test),
					100.0 * np.percentile(clause_recall_test, 90),
					100.0 * np.percentile(clause_recall_test, 95),
					100.0 * np.percentile(clause_recall_test, 99),
					100.0*clause_precision_test.mean(),
					100.0*clause_recall_test.mean(),
					100.0 * np.percentile(max_example_precision, 1),
					100.0 * np.percentile(max_example_precision, 5),
					100.0 * np.percentile(max_example_precision, 10),
					100.0 * np.median(max_example_precision),
					100.0 * np.percentile(max_example_precision, 90),
					100.0 * np.percentile(max_example_precision, 95),
					100.0 * np.percentile(max_example_precision, 99),
					transformed_X_test.sum() / transformed_X_test.shape[0],
					100.0 * (1.0 - (activations[1][0]/transformed_X_test.shape[0])) if activations[0][0] == 0 else 100.0
				)
			)
		else:
			f.write("%d %d %.2f %.2f %.2f %.2f\n" %
				(
					r,
					e,
					result_test,
					result_test_held_out_permutations,
					result_test_held_out_synonym_sequences,
					result_train
				)
			)
		f.flush()
f.close()
