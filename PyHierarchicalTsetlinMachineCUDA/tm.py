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

import numpy as np

import PyHierarchicalTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

from time import time

OR_GROUP = "OR*"
OR_ALTERNATIVES = "OR"
AND_GROUP = "AND"

VANILLA_TM = 0
WEIGHTED_TM = 1
COALESCED_TM = 2

g = curandom.XORWOWRandomNumberGenerator() 

class CommonTsetlinMachine():

	def __init__(self, number_of_clauses, T, s, q=1.0, hierarchy_structure=None, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.number_of_clauses = number_of_clauses
		self.number_of_state_bits = number_of_state_bits
		self.T = int(T)
		self.s = s
		self.q = q
		self.hierarchy_structure = hierarchy_structure
		self.depth = len(hierarchy_structure)

		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

		# Calculates the number of nodes at each level of the hierarchy
		self.hierarchy_size = [0] * (self.depth + 1)
		self.hierarchy_size[self.depth] = 1
		for d in range(self.depth - 1):
			self.hierarchy_size[self.depth - d - 1] = self.hierarchy_structure[self.depth - d - 1][1] * self.hierarchy_size[self.depth - d]

		# Represents hierarchy structure for transfer to GPU
		self.hierarchy_structure_factors = [0] * (self.depth - 1)
		self.hierarchy_structure_alternatives = [0] * (self.depth - 1)
		for d in range(1, self.depth):
			self.hierarchy_structure_factors[d-1] = self.hierarchy_structure[d][1]
			if self.hierarchy_structure[d][0] == OR_ALTERNATIVES:
				self.hierarchy_structure_alternatives[d-1] = 1

		# Calculates total number of features spanned by the hierarchy
		self.number_of_features_hierarchy = 1
		for d in range(self.depth - 1, -1, -1):
			if (self.hierarchy_structure[d][0] == OR_GROUP or self.hierarchy_structure[d][0] == AND_GROUP):
				self.number_of_features_hierarchy *= self.hierarchy_structure[d][1]

		# Calculates literal chunks per leaf
		self.number_of_features_per_leaf = self.hierarchy_structure[0][1]
		if self.append_negated:
			self.number_of_literals = self.number_of_features_hierarchy * 2
			self.number_of_literals_per_leaf = self.number_of_features_per_leaf * 2
			self.number_of_literal_chunks_per_leaf = int((self.number_of_literals_per_leaf - 1) / 32 + 1)
		else:
			self.number_of_literals = self.number_of_features_hierarchy
			self.number_of_literals_per_leaf = self.number_of_features_per_leaf
			self.number_of_literal_chunks_per_leaf = int((self.number_of_literals_per_leaf - 1) / 32 + 1)

		# Calculates the number of literal chunks for the full hierarchy
		self.hierarchy_size[0] = self.number_of_literal_chunks_per_leaf * self.hierarchy_size[1]

		# Calculates number of literal chunks overall for the feature vector (ignores OR alternatives)
		self.number_of_literal_chunks = self.number_of_literal_chunks_per_leaf
		for d in range(self.depth - 1, 0, -1):
			if (self.hierarchy_structure[d][0] == OR_GROUP or self.hierarchy_structure[d][0] == AND_GROUP):
				self.number_of_literal_chunks *= self.hierarchy_structure[d][1]

		self.cuda_modules()

		self.first = True

	def cuda_modules(self):
		parameters = """
	#define CLAUSES %d
	#define DEPTH %d
	#define COMPONENTS %d
	#define LITERALS_PER_LEAF %d
	#define TA_CHUNKS_PER_LEAF %d
	#define LITERAL_CHUNKS %d
	#define STATE_BITS %d
	#define BOOST_TRUE_POSITIVE_FEEDBACK %d
	#define S %f
	#define THRESHOLD %d
	#define Q %f

	#define NEGATIVE_CLAUSES %d
	#define FLIP_POLARITY %d
		""" % (self.number_of_clauses, self.depth, self.hierarchy_size[1], self.number_of_literals_per_leaf, self.number_of_literal_chunks_per_leaf, self.number_of_literal_chunks, self.number_of_state_bits, self.boost_true_positive_feedback, self.s, self.T, self.q, self.negative_clauses, self.flip_polarity)
		
		mod_prepare = SourceModule(parameters + kernels.code_header + kernels.code_prepare, no_extern_c=True)
		self.prepare_weights = mod_prepare.get_function("prepare_weights")
		self.prepare_hierarchy = mod_prepare.get_function("prepare_hierarchy")

		mod_update = SourceModule(parameters + kernels.code_header + kernels.code_update, no_extern_c=True)		
		self.update_hierarchy = mod_update.get_function("update_hierarchy")
		self.update_hierarchy.prepare("PiPPPiPPPPPi")

		self.update_weights = mod_update.get_function("update_weights")
		self.update_weights.prepare("PiiPPPPi")

		self.evaluate_leaves = mod_update.get_function("evaluate_leaves")
		self.evaluate_leaves.prepare("PPPiPPPi")

		self.evaluate_final = mod_update.get_function("evaluate_final")
		self.evaluate_final.prepare("iPPP")

		self.evaluate_and_groups = mod_update.get_function("evaluate_and_groups")
		self.evaluate_and_groups.prepare("PPii")

		self.propagate_and_group_false_truth_values = mod_update.get_function("propagate_and_group_false_truth_values")
		self.propagate_and_group_false_truth_values.prepare("PPii")

		self.evaluate_or_groups = mod_update.get_function("evaluate_or_groups")
		self.evaluate_or_groups.prepare("PPii")

		self.evaluate_or_alternatives = mod_update.get_function("evaluate_or_alternatives")
		self.evaluate_or_alternatives.prepare("PPii")

		# CUDA modules for encoding input data
		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.prepare_encode_hierarchy = mod_encode.get_function("prepare_encode_hierarchy")
		self.encode_hierarchy = mod_encode.get_function("encode_hierarchy")

	def encode_X(self, X, encoded_X_hierarchy_gpu):
		number_of_examples = X.shape[0]

		# Allocates GPU memory for input data
		Xm = np.ascontiguousarray(X.flatten()).astype(np.uint32)
		X_gpu = cuda.mem_alloc(Xm.nbytes)
		cuda.memcpy_htod(X_gpu, Xm)

		# Prepares for leaf encoding of the input data
		self.prepare_encode_hierarchy(X_gpu, encoded_X_hierarchy_gpu, np.int32(self.number_of_literal_chunks), np.int32(number_of_examples), grid=self.grid, block=self.block)
		cuda.Context.synchronize()	
		
		# Encodes the input data split across the leaves
		self.encode_hierarchy(X_gpu, encoded_X_hierarchy_gpu, np.int32(self.number_of_features_hierarchy), np.int32(self.number_of_literal_chunks), np.int32(self.hierarchy_size[1]), np.int32(self.number_of_features_per_leaf), np.int32(self.number_of_literal_chunks_per_leaf), np.int32(self.append_negated), np.int32(number_of_examples), grid=self.grid, block=self.block)
		cuda.Context.synchronize()

	def allocate_gpu_memory(self):
		# GPU memory for accumulating votes, level by level
		self.hierarchy_votes = []
		for d in range(1, self.depth):
			self.hierarchy_votes.append(cuda.mem_alloc(self.number_of_clauses*int(self.hierarchy_size[d])*4))
		self.hierarchy_votes.append(cuda.mem_alloc(self.number_of_clauses*4))

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_factors_gpu = cuda.mem_alloc((self.depth-1)*4)
		cuda.memcpy_htod(self.hierarchy_structure_factors_gpu, np.array(self.hierarchy_structure_factors, dtype=np.int32))

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_alternatives_gpu = cuda.mem_alloc((self.depth-1)*4)
		cuda.memcpy_htod(self.hierarchy_structure_alternatives_gpu, np.array(self.hierarchy_structure_alternatives, dtype=np.int32))

		# GPU memory for storing Tsetlin Automata states
		self.ta_state_hierarchy_gpu = cuda.mem_alloc(self.number_of_clauses*self.hierarchy_size[0]*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		self.component_weights_gpu = cuda.mem_alloc(self.number_of_clauses*self.hierarchy_size[1]*4) # Only positive weights...
		self.class_sum_gpu = cuda.mem_alloc(self.number_of_outputs*4)

	def ta_action(self, clause, leaf, ta):
		ta_state_hierarchy = np.empty(self.number_of_clauses*self.hierarchy_size[1]*self.number_of_literal_chunks_per_leaf*self.number_of_state_bits, dtype=np.uint32)
		cuda.memcpy_dtoh(ta_state_hierarchy, self.ta_state_hierarchy_gpu)
		ta_state_hierarchy = ta_state_hierarchy.reshape((self.number_of_clauses, self.hierarchy_size[1], self.number_of_literal_chunks_per_leaf, self.number_of_state_bits))

		return (ta_state_hierarchy[clause, leaf, ta // 32, self.number_of_state_bits-1] & (1 << (ta % 32))) > 0

	def ta_state(self, clause, leaf, ta):
		ta_state_hierarchy = np.empty(self.number_of_clauses*self.hierarchy_size[1]*self.number_of_literal_chunks_per_leaf*self.number_of_state_bits, dtype=np.uint32)
		cuda.memcpy_dtoh(ta_state_hierarchy, self.ta_state_hierarchy_gpu)
		ta_state_hierarchy = ta_state_hierarchy.reshape((self.number_of_clauses, self.hierarchy_size[1], self.number_of_literal_chunks_per_leaf, self.number_of_state_bits))
		
		state = 0
		for b in range(self.number_of_state_bits):
			if (ta_state_hierarchy[clause, leaf, ta // 32, b] & (1 << (ta % 32))) > 0:
				state |= (1 << b)
		
		return state

	def get_state(self):
		# To be updated
		ta_state_hierarchy = np.empty(self.number_of_clauses*self.hierarchy_size[1]*self.number_of_literal_chunks_per_leaf*self.number_of_state_bits, dtype=np.uint32)
		cuda.memcpy_dtoh(self.ta_state_hierarchy, self.ta_state_hierarchy_gpu)
		clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses).astype(np.int32)
		cuda.memcpy_dtoh(self.clause_weights, self.clause_weights_gpu)
		return((self.ta_state_hierarchy, self.clause_weights, self.number_of_outputs, self.number_of_clauses, self.hierarchy_structure, self.boost_true_positive_feedback, self.number_of_state_bits, self.append_negated, self.min_y, self.max_y))

	def set_state(self, state):
		# To be updated
		self.number_of_outputs = state[2]
		self.number_of_clauses = state[3]
		self.hierarchy_structure = state[4]
		self.boost_true_positive_feedback = state[5]
		self.number_of_state_bits = state[6]
		self.append_negated = state[7]
		self.min_y = state[8]
		self.max_y = state[9]
		
		self.ta_state_hierarchy_gpu = cuda.mem_alloc(self.number_of_clauses*self.hierarchy_size[0]*self.number_of_state_bits*4)
		self.clause_weights_gpu = cuda.mem_alloc(self.number_of_outputs*self.number_of_clauses*4)
		cuda.memcpy_htod(self.ta_state_hierarchy_gpu, state[0])
		cuda.memcpy_htod(self.clause_weights_gpu, state[1])

	# Transform input data for processing at next layer
	def transform(self, X):
		None # To be updated

	def initialize_weights_and_ta_states(self):
		self.prepare_weights(g.state, np.int32(self.tm_type), np.int32(self.number_of_outputs), self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()

		self.prepare_hierarchy(g.state, np.int32(self.number_of_outputs), self.ta_state_hierarchy_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()

	def evaluate_hierarchy(self, encoded_X_hierarchy, e):
		# Initializes class sums to zero
		class_sum = np.ascontiguousarray(np.zeros(self.number_of_outputs)).astype(np.int32)
		cuda.memcpy_htod(self.class_sum_gpu, class_sum)

		# Evaluates all the hierarchy leaves in parallel
		self.evaluate_leaves.prepared_call(
			self.grid,
			self.block,
			self.ta_state_hierarchy_gpu,
			self.component_weights_gpu,
			self.hierarchy_votes[0],
			self.depth,
			self.hierarchy_structure_factors_gpu,
			self.hierarchy_structure_alternatives_gpu,
			encoded_X_hierarchy,
			np.int32(e)
		)
		cuda.Context.synchronize()

		# Propagates votes bottom-up in the hierarchy, starting from the clause components (leaves)
		for d in range(1, self.depth):
			if (self.hierarchy_structure[d][0] == AND_GROUP):
				self.evaluate_and_groups.prepared_call(
					self.grid,
					self.block,
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					self.hierarchy_size[d + 1],
					self.hierarchy_structure[d][1]
				)
				cuda.Context.synchronize()
			elif self.hierarchy_structure[d][0] == OR_GROUP:
				self.evaluate_or_groups.prepared_call(
					self.grid,
					self.block,
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					self.hierarchy_size[d + 1],
					self.hierarchy_structure[d][1]
				)
				cuda.Context.synchronize()
			elif self.hierarchy_structure[d][0] == OR_ALTERNATIVES:
				self.evaluate_or_alternatives.prepared_call(
					self.grid,
					self.block,
					self.hierarchy_votes[d-1],
					self.hierarchy_votes[d],
					self.hierarchy_size[d + 1],
					self.hierarchy_structure[d][1]
				)
				cuda.Context.synchronize()
			else:
				printf("Unknown node type!")
				sys.exit()

		# Adds up the votes from each clause (hierarchy root)
		self.evaluate_final.prepared_call(
			self.grid,
			self.block,
			np.int32(self.number_of_outputs),
			self.hierarchy_votes[self.depth-1],
			self.clause_weights_gpu,
			self.class_sum_gpu
		)
		cuda.Context.synchronize()

	def _fit(self, X, encoded_Y, epochs=100, incremental=False):
		number_of_examples = X.shape[0]

		if self.first:
			# Allocates memory and prepares weights and Tsetlin automata states on first run 
			self.allocate_gpu_memory()

			self.initialize_weights_and_ta_states()

			self.first = False
		elif not incremental:
			# Re-initializes weights and Tsetlin automata states if training is not incremental
			self.initialize_weights_and_ta_states()

		# Allocates GPU memory for training data
		encoded_X_hierarchy_training_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_literal_chunks * 4))
		Y_gpu = cuda.mem_alloc(encoded_Y.nbytes)

		self.encode_X(X, encoded_X_hierarchy_training_gpu)
		cuda.memcpy_htod(Y_gpu, encoded_Y)

		for epoch in range(epochs):
			for e in range(number_of_examples):
				self.evaluate_hierarchy(encoded_X_hierarchy_training_gpu, e)

				# Propagates the root value and any intermittent node values back to the leaves.
				# The purpose is to determine which leaves only has True nodes on the path from leaf to root.
				for d in range(self.depth-1, 0, -1):
					self.propagate_and_group_false_truth_values.prepared_call(
						self.grid,
						self.block,
						self.hierarchy_votes[d-1],
						self.hierarchy_votes[d],
						self.hierarchy_size[d + 1],
						self.hierarchy_structure[d][1]
					)
					cuda.Context.synchronize()

				# Updates the clause components (leaves) based on the propagated truth values
				self.update_hierarchy.prepared_call(
					self.grid,
					self.block,
					g.state,
					np.int32(self.number_of_outputs),
					self.ta_state_hierarchy_gpu,
					self.clause_weights_gpu,
					self.hierarchy_votes[0],
					self.depth,
					self.hierarchy_structure_factors_gpu,
					self.hierarchy_structure_alternatives_gpu,
					self.class_sum_gpu,
					encoded_X_hierarchy_training_gpu,
					Y_gpu,
					np.int32(e)
				)
				cuda.Context.synchronize()

				# Updates the clause weights
				if (self.tm_type in [WEIGHTED_TM, COALESCED_TM]):
					self.update_weights.prepared_call(
						self.grid,
						self.block,
						g.state,
						np.int32(self.tm_type),
						np.int32(self.number_of_outputs),
						self.clause_weights_gpu,
						self.hierarchy_votes[self.depth-1],
						self.class_sum_gpu,
						Y_gpu,
						np.int32(e)
					)
					cuda.Context.synchronize()
		return
       
	def _score(self, X):
		number_of_examples = X.shape[0]
		
		encoded_X_hierarchy_test_gpu = cuda.mem_alloc(int(number_of_examples * self.number_of_literal_chunks * 4))
		self.encode_X(X, encoded_X_hierarchy_test_gpu)

		class_sum = np.ascontiguousarray(np.zeros((self.number_of_outputs, number_of_examples))).astype(np.int32)
		class_sum_example = np.ascontiguousarray(np.zeros(self.number_of_outputs)).astype(np.int32)

		for e in range(number_of_examples):
			self.evaluate_hierarchy(encoded_X_hierarchy_test_gpu, e)

			cuda.memcpy_dtoh(class_sum_example, self.class_sum_gpu)
			class_sum[:, e] = class_sum_example
		
		class_sum = np.clip(class_sum.reshape((self.number_of_outputs, number_of_examples)), -self.T, self.T)

		return class_sum

	def print_hierarchy(self):
		for i in range(self.number_of_clauses):
			print("CLAUSE %d" % (i))
			for j in range(self.hierarchy_size[1]):
				component_remainder = j
				size = 1

				previous_index = np.ones((self.depth-1), dtype=np.int32)*-1
				headings = []
				for d in range(1, self.depth):
					depth_d_node_index = component_remainder % self.hierarchy_structure[d][1]
					component_remainder = component_remainder / self.hierarchy_structure[d][1]

					if previous_index[d-1] != depth_d_node_index:
						headings.append("\t" * (self.depth - d) + "%s" % (self.hierarchy_structure[d][0]))
						previous_index[d-1] = depth_d_node_index
					else:
						headings.append('')

				for d in range(self.depth-2, -1, -1):
					print(headings[d])

				l = []
				for k in range(self.number_of_literals_per_leaf):
					if self.ta_action(i, j, k):
						if k < self.number_of_literals_per_leaf // 2:
							l.append("x%d(%d)" % (k, self.ta_state(i, j, k)))
						else:
							l.append("¬x%d(%d)" % (k - self.number_of_literals_per_leaf // 2, self.ta_state(i, j, k)))
				
				print("\t" * self.depth + " ^ ".join(l))
	
class MultiOutputTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.negative_clauses = 1
		super().__init__(number_of_clauses, T, s, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = Y.shape[1]
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.uint32).transpose()

class MultiClassCoalescedTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.negative_clauses = 1
		self.tm_type = COALESCED_TM
		self.flip_polarity = 1

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = int(np.max(Y) + 1)
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.empty((Y.shape[0], self.number_of_outputs), dtype = np.int32)
		for i in range(self.number_of_outputs):
			encoded_Y[:,i] = np.where(Y == i, self.T, -self.T)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class MultiClassTsetlinMachine:
	def __init__(self, number_of_clauses, T, s, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.number_of_clauses = number_of_clauses
		self.T = T
		self.s = s
		self.weighted_clauses = weighted_clauses
		self.hierarchy_structure = hierarchy_structure
		self.q = q
		self.boost_true_positive_feedback = boost_true_positive_feedback
		self.number_of_state_bits = number_of_state_bits
		self.append_negated = append_negated
		self.grid = grid
		self.block = block

		self.configured = False

	def fit(self, X, Y, epochs=100, incremental=False):
		self.number_of_outputs = int(np.max(Y) + 1)
		
		if not self.configured:
			self.tms = []
			for i in range(self.number_of_outputs):
				self.tms.append(TsetlinMachine(self.number_of_clauses, self.T, self.s, weighted_clauses=self.weighted_clauses, hierarchy_structure=self.hierarchy_structure, q=self.q, boost_true_positive_feedback=self.boost_true_positive_feedback, number_of_state_bits=self.number_of_state_bits, append_negated=self.append_negated, grid=self.grid, block=self.block))

			self.configured = True
		
		encoded_Y = np.empty(Y.shape[0], dtype = np.int32)

		for epoch in range(epochs):
			for i in range(self.number_of_outputs):
				target_X = X[Y==i]

				not_target_X = X[Y!=i]
				not_target_index = np.random.rand(not_target_X.shape[0]) <= 1.0/(self.number_of_outputs - 1)

				balanced_X = np.vstack((target_X, not_target_X[not_target_index,:]))
				balanced_Y = np.hstack((np.ones(target_X.shape[0]), np.zeros(not_target_X.shape[0])))
				index = np.arange(balanced_X.shape[0])
				np.random.shuffle(index)

				self.tms[i].fit(balanced_X[index], balanced_Y[index], epochs=1, incremental=incremental)
		return

	def score(self, X):
		class_sums = np.empty((self.number_of_outputs, X.shape[0]), dtype=np.int32)
		for i in range(self.number_of_outputs):
			class_sums[i,:] = self.tms[i].score(X)

		return class_sums

	def predict(self, X):
		return np.argmax(self.score(X), axis=0)

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.negative_clauses = 1
		self.flip_polarity = 0

		if weighted_clauses:
			self.tm_type = WEIGHTED_TM
		else:
			self.tm_type = VANILLA_TM

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)

		self.number_of_outputs = 1

		self.max_y = None
		self.min_y = None
		
		encoded_Y = np.where(Y == 1, self.T, -self.T).astype(np.int32)

		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def score(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		return self._score(X)[0,:]

	def predict(self, X):
		return (self.score(X) >= 0).astype(np.int32)

class RegressionTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1)):
		self.negative_clauses = 0
		self.flip_polarity = 0

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block)

	def fit(self, X, Y, epochs=100, incremental=False):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		self.number_of_outputs = 1
		self.patch_dim = (X.shape[1], 1, 1)

		self.max_y = np.max(Y)
		self.min_y = np.min(Y)
	
		encoded_Y = ((Y - self.min_y)/(self.max_y - self.min_y)*self.T).astype(np.int32)
			
		self._fit(X, encoded_Y, epochs = epochs, incremental = incremental)

		return

	def predict(self, X):
		X = X.reshape(X.shape[0], X.shape[1], 1)
		
		return 1.0*(self._score(X)[0,:])*(self.max_y - self.min_y)/(self.T) + self.min_y
