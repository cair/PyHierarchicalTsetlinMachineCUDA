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

from collections import deque
import numpy as np

import PyHierarchicalTsetlinMachineCUDA.kernels as kernels

import pycuda.curandom as curandom
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda import gpuarray
import sys

from time import time

OR_GROUP = " ∨* "
OR_ALTERNATIVES = " ∨ "
AND_GROUP = " ∧ "

VANILLA_TM = 0
WEIGHTED_TM = 1
COALESCED_TM = 2


class CommonTsetlinMachine():

	def __init__(self, number_of_clauses, T, s, q=1.0, hierarchy_structure=None, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
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
		self.hierarchy_structure_type = [0] * (self.depth - 1)
		for d in range(1, self.depth):
			self.hierarchy_structure_factors[d-1] = self.hierarchy_structure[d][1]
			if self.hierarchy_structure[d][0] == OR_ALTERNATIVES:
				self.hierarchy_structure_type[d-1] = 1
			elif self.hierarchy_structure[d][0] == OR_GROUP:
				self.hierarchy_structure_type[d-1] = 2

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

		self.np_rng = np.random.default_rng(seed)
		if seed is not None:
			self.cuda_rng = curandom.XORWOWRandomNumberGenerator(seed_getter=lambda N: gpuarray.to_gpu(self.np_rng.integers(1, 2**30, size=N).astype(np.int32)))
		else:
			self.cuda_rng = curandom.XORWOWRandomNumberGenerator() 

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
		self.evaluate_and_groups.prepare("PPiii")

		self.propagate_and_group_false_truth_values = mod_update.get_function("propagate_and_group_false_truth_values")
		self.propagate_and_group_false_truth_values.prepare("PPii")

		self.propagate_or_group_false_truth_values = mod_update.get_function("propagate_or_group_false_truth_values")
		self.propagate_or_group_false_truth_values.prepare("PPPii")

		self.evaluate_or_groups = mod_update.get_function("evaluate_or_groups")
		self.evaluate_or_groups.prepare("PPii")

		self.evaluate_or_alternatives = mod_update.get_function("evaluate_or_alternatives")
		self.evaluate_or_alternatives.prepare("PPii")

		# CUDA modules for encoding input data
		mod_encode = SourceModule(kernels.code_encode, no_extern_c=True)
		self.prepare_encode_hierarchy = mod_encode.get_function("prepare_encode_hierarchy")
		self.encode_hierarchy = mod_encode.get_function("encode_hierarchy")

		mod_clauses = SourceModule(parameters + kernels.code_clauses, no_extern_c=True)
		self.kernel_get_ta_states = mod_clauses.get_function("get_ta_states")

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
			self.hierarchy_votes.append(cuda.mem_alloc(self.number_of_clauses*int(self.hierarchy_size[d])*8))
		self.hierarchy_votes.append(cuda.mem_alloc(self.number_of_clauses*8))

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_factors_gpu = cuda.mem_alloc((self.depth-1)*4)
		cuda.memcpy_htod(self.hierarchy_structure_factors_gpu, np.array(self.hierarchy_structure_factors, dtype=np.int32))

		# GPU memory for storing hierarchy structure
		self.hierarchy_structure_type_gpu = cuda.mem_alloc((self.depth-1)*4)
		cuda.memcpy_htod(self.hierarchy_structure_type_gpu, np.array(self.hierarchy_structure_type, dtype=np.int32))

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
		self.prepare_weights(self.cuda_rng.state, np.int32(self.tm_type), np.int32(self.number_of_outputs), self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
		cuda.Context.synchronize()

		self.prepare_hierarchy(self.cuda_rng.state, np.int32(self.number_of_outputs), self.ta_state_hierarchy_gpu, self.clause_weights_gpu, self.class_sum_gpu, grid=self.grid, block=self.block)
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
			self.hierarchy_structure_type_gpu,
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
					self.hierarchy_structure[d][1],
					d == self.depth-1
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
		if self.number_of_features_hierarchy != X.shape[1]:
			print("The number of features spanned by hierarchy does not align with the input data.")
			sys.exit(-1)

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
					if self.hierarchy_structure[d][0] != OR_GROUP:
						self.propagate_and_group_false_truth_values.prepared_call(
							self.grid,
							self.block,
							self.hierarchy_votes[d-1],
							self.hierarchy_votes[d],
							self.hierarchy_size[d + 1],
							self.hierarchy_structure[d][1]
						)
						cuda.Context.synchronize()
					else:
						self.propagate_or_group_false_truth_values.prepared_call(
							self.grid,
							self.block,
							self.cuda_rng.state,
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
					self.cuda_rng.state,
					np.int32(self.number_of_outputs),
					self.ta_state_hierarchy_gpu,
					self.clause_weights_gpu,
					self.hierarchy_votes[0],
					self.depth,
					self.hierarchy_structure_factors_gpu,
					self.hierarchy_structure_type_gpu,
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
						self.cuda_rng.state,
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

	def get_ta_states(self) -> np.ndarray:
		"""
		Get state value for each TA.
		Returns: Numpy array of shape (number_of_clauses, number_of_clause_components, number_of_literals_per_leaf)
		"""
		# Mem Allocation
		ta_states_gpu = gpuarray.to_gpu(
			np.zeros((self.number_of_clauses, self.hierarchy_size[1], self.number_of_literals_per_leaf), dtype=np.uint32)
		)

		# Calculate grid size based on the kernel
		total = self.number_of_clauses * self.hierarchy_size[1] * self.number_of_literals_per_leaf
		grid = (((total + self.block[0] - 1) // self.block[0]), 1, 1)
		self.kernel_get_ta_states(self.ta_state_hierarchy_gpu, ta_states_gpu, block=self.block, grid=grid)

		# Copy back to CPU
		return ta_states_gpu.get()

	def get_literals(self):
		"""
		Get included literals for each clause.
		Returns: Numpy array of shape (number_of_clauses, number_of_clause_components, number_of_literals_per_leaf)
		"""
		return (self.get_ta_states() >= (1 << (self.number_of_state_bits - 1))).astype(np.uint8)

	def map_ta_id_to_feature_id(self):
		"""
		Return an array of shape(number_of_clause_components, number_of_literals_per_leaf). That is the total number of TAs in a clause. Maps each TA id to a feature_id in the input. In each component, the first half of the TAs correspond to the positive features, and the second half correspond to the negated features.
		"""
		# BFS top-down traversal
		q = deque()
		q.append((self.depth - 1, 0, 0)) # (level, node_id, group_id)

		comp_grps = -1 * np.ones(self.hierarchy_size[1], dtype=np.int32)
		while q:
			level, node_id, group_id = q.popleft()

			if level == 0:
				# This is the leaf component
				comp_grps[node_id] = group_id
				continue

			n_children = self.hierarchy_structure[level][1]
			is_alt = (self.hierarchy_structure[level][0] == OR_ALTERNATIVES)
			for child_pos in range(n_children):
				child_id = node_id * n_children + child_pos
				if is_alt:
					# All children share the same features
					child_group_id = group_id
				else:
					# Features are partitioned among the children
					child_group_id = group_id * n_children + child_pos

				q.append((level - 1, child_id, child_group_id))

		# map each TA in a component to a feature
		half = self.number_of_literals_per_leaf // 2
		lit_ids = np.arange(self.number_of_literals_per_leaf)
		local_feats = lit_ids % half if self.append_negated else lit_ids
		fmap = comp_grps[:, None] * (half if self.append_negated else self.number_of_literals_per_leaf) + local_feats[None, :]

		return fmap

	def calc_hierarchy_votes(self, X):
		"""
		Get the clause activation information for each sample in X.
		"""
		assert not self.first, "Model must be trained before getting activations."

		number_of_examples = X.shape[0]
		encoded_X_hierarchy_test_gpu = gpuarray.empty((number_of_examples, self.number_of_literal_chunks), dtype=np.uint32)
		self.encode_X(X, encoded_X_hierarchy_test_gpu.gpudata)

		class_sum = np.ascontiguousarray(np.zeros((self.number_of_outputs, number_of_examples))).astype(np.int32)
		class_sum_example = np.ascontiguousarray(np.zeros(self.number_of_outputs)).astype(np.int32)
		hierarchy_votes = []
		for e in range(number_of_examples):
			self.evaluate_hierarchy(encoded_X_hierarchy_test_gpu.gpudata, e)

			cuda.memcpy_dtoh(class_sum_example, self.class_sum_gpu)
			class_sum[:, e] = class_sum_example

			hierarchy_votes_example = []
			for d in range(self.depth):
				temp_arr = np.empty(self.number_of_clauses*int(self.hierarchy_size[d+1]), dtype=np.int32)
				cuda.memcpy_dtoh(temp_arr, self.hierarchy_votes[d])
				hierarchy_votes_example.append(temp_arr.reshape((self.number_of_clauses, int(self.hierarchy_size[d+1]))))

			hierarchy_votes.append(hierarchy_votes_example)
		
		class_sum = np.clip(class_sum.reshape((self.number_of_outputs, number_of_examples)), -self.T, self.T)
		return hierarchy_votes, class_sum

	def print_hierarchy(self, print_ta_state=False):
		for i in range(self.number_of_clauses):
			print("\nCLAUSE #%d: " % (i), end='')

			previous_index = np.ones((self.depth-1), dtype=np.int32)*-1
			for j in range(self.hierarchy_size[1]):
				component_remainder = j
				size = 1

				left = []
				right = []
				inside = []
				feature_base = 0
				size = self.hierarchy_structure[0][1]
				for d in range(1, self.depth):
					depth_d_node_index = component_remainder % self.hierarchy_structure[d][1]
					component_remainder = component_remainder // self.hierarchy_structure[d][1]

					if self.hierarchy_structure[d][0] != OR_ALTERNATIVES:
						feature_base += size * depth_d_node_index 
						size *= self.hierarchy_structure[d][1];

					if previous_index[d-1] == -1:
						left.append("(")
					elif depth_d_node_index == 0 and previous_index[d-1] != depth_d_node_index:
						right.append(")")
						left.insert(0, "(")
					elif previous_index[d-1] != depth_d_node_index:
						inside.append(self.hierarchy_structure[d][0])
					
					previous_index[d-1] = depth_d_node_index

				for s in right:
					print(s, end='')

				for s in inside:
					print(s, end='')

				for s in left:
					print(s, end='')

				l = []
				for k in range(self.number_of_literals_per_leaf):
					if self.ta_action(i, j, k):
						if k < self.number_of_literals_per_leaf // 2:
							if print_ta_state:
								l.append("x%d(%d)" % (feature_base + k, self.ta_state(i, j, k)))
							else:
								l.append("x%d" % (feature_base + k,))
						else:
							if print_ta_state:
								l.append("¬x%d(%d)" % (feature_base + k - self.number_of_literals_per_leaf // 2, self.ta_state(i, j, k)))
							else:
								l.append("¬x%d" % (feature_base + k - self.number_of_literals_per_leaf // 2,))
				
				if len(l) > 1:
					print("(" + " ∧ ".join(l) + ")", end = '')
				elif len(l) == 1:
					print(l[0], end = '')

			print(")" * (self.depth - 1))

	def save(self) -> dict:
		ta_state_hierarchy = np.empty(self.number_of_clauses*self.hierarchy_size[0]*self.number_of_state_bits, dtype=np.uint32)
		cuda.memcpy_dtoh(ta_state_hierarchy, self.ta_state_hierarchy_gpu)

		clause_weights = np.empty(self.number_of_outputs*self.number_of_clauses, dtype=np.int32)
		cuda.memcpy_dtoh(clause_weights, self.clause_weights_gpu)

		return {
			'ta_state_hierarchy': ta_state_hierarchy,
			'clause_weights': clause_weights,
			'number_of_outputs': self.number_of_outputs,
			'min_y': self.min_y,
			'max_y': self.max_y,
			'params': {
				'number_of_clauses': self.number_of_clauses,
				'T': self.T,
				's': self.s,
				'q': self.q,
				'hierarchy_structure': self.hierarchy_structure,
				'boost_true_positive_feedback': self.boost_true_positive_feedback,
				'number_of_state_bits': self.number_of_state_bits,
				'append_negated': self.append_negated,
			},
			'negative_clauses': self.negative_clauses,
			'tm_type': self.tm_type,
			'flip_polarity': self.flip_polarity,
			'weighted_clauses': getattr(self, 'weighted_clauses', None),
		}

	def load(self, state_dict: dict):
		self.number_of_outputs = state_dict['number_of_outputs']
		self.min_y = state_dict['min_y']
		self.max_y = state_dict['max_y']

		self.allocate_gpu_memory()

		cuda.memcpy_htod(self.ta_state_hierarchy_gpu, state_dict['ta_state_hierarchy'])
		cuda.memcpy_htod(self.clause_weights_gpu, state_dict['clause_weights'])

		self.first = False

	
class MultiOutputTsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		super().__init__(number_of_clauses, T, s, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

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
	def __init__(self, number_of_clauses, T, s, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		self.tm_type = COALESCED_TM
		self.flip_polarity = 1

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

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
	def __init__(self, number_of_clauses, T, s, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
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
		self.seed = np.random.randint(1, 2**30) if seed is None else seed

		self.configured = False

	def fit(self, X, Y, epochs=100, incremental=False):
		self.number_of_outputs = int(np.max(Y) + 1)

		if not self.configured:
			self.tms = []
			for i in range(self.number_of_outputs):
				self.tms.append(TsetlinMachine(self.number_of_clauses, self.T, self.s, weighted_clauses=self.weighted_clauses, hierarchy_structure=self.hierarchy_structure, q=self.q, boost_true_positive_feedback=self.boost_true_positive_feedback, number_of_state_bits=self.number_of_state_bits, append_negated=self.append_negated, grid=self.grid, block=self.block, seed=self.seed+i))

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

	def save(self) -> dict:
		return {
			'number_of_outputs': self.number_of_outputs,
			'tms': [t.save() for t in self.tms],
			'params': {
				'number_of_clauses': self.number_of_clauses,
				'T': self.T,
				's': self.s,
				'q': self.q,
				'weighted_clauses': self.weighted_clauses,
				'hierarchy_structure': self.hierarchy_structure,
				'boost_true_positive_feedback': self.boost_true_positive_feedback,
				'number_of_state_bits': self.number_of_state_bits,
				'append_negated': self.append_negated,
				'grid': self.grid,
				'block': self.block,
			},
		}

	def load(self, state_dict: dict):
		self.number_of_outputs = state_dict['number_of_outputs']

		self.tms = []
		for tm_state in state_dict['tms']:
			t = TsetlinMachine(self.number_of_clauses, self.T, self.s, weighted_clauses=self.weighted_clauses, hierarchy_structure=self.hierarchy_structure, q=self.q, boost_true_positive_feedback=self.boost_true_positive_feedback, number_of_state_bits=self.number_of_state_bits, append_negated=self.append_negated, grid=self.grid, block=self.block)
			t.load(tm_state)
			self.tms.append(t)

		self.configured = True

class TsetlinMachine(CommonTsetlinMachine):
	def __init__(self, number_of_clauses, T, s, weighted_clauses=False, hierarchy_structure=((AND_GROUP, 1)), q=1.0, boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 1
		self.flip_polarity = 0

		if weighted_clauses:
			self.tm_type = WEIGHTED_TM
		else:
			self.tm_type = VANILLA_TM

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, q=q, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

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
	def __init__(self, number_of_clauses, T, s, hierarchy_structure=((AND_GROUP, 1)), boost_true_positive_feedback=1, number_of_state_bits=8, append_negated=True, grid=(16*13,1,1), block=(128,1,1), seed=None):
		self.negative_clauses = 0
		self.flip_polarity = 0

		super().__init__(number_of_clauses, T, s, hierarchy_structure=hierarchy_structure, boost_true_positive_feedback=boost_true_positive_feedback, number_of_state_bits=number_of_state_bits, append_negated=append_negated, grid=grid, block=block, seed=seed)

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
