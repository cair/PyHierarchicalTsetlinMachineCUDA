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


code_header = """
	#include <curand_kernel.h>
	
	#define INT_SIZE 32ULL

	#define NEG_INFINITY (-1 * INFINITY)

	//#define NEG_INFINITY 0x80000000

	#if (LITERALS_PER_LEAF % 32 != 0)
	#define FILTER_HIERARCHICAL (~(0xffffffff << (LITERALS_PER_LEAF % INT_SIZE)))
	#else
	#define FILTER_HIERARCHICAL 0xffffffff
	#endif

	#define VANILLA_TM 0

	#define WEIGHTED_TM 1

	#define COALESCED_TM 2
"""

code_update = """
	extern "C"
    {
    	__device__ inline void update_clause_weight(curandState *localState, int tm_type, int number_of_outputs, float *clause_weight, int clause_output, int y, float class_sum);
    	__device__ inline void update_component_hierarchy(curandState *localState, int number_of_outputs, float *clause_weight, unsigned int *ta_state, int component_output, int *X, int y, float class_sum);

    	// Increment the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void inc(unsigned int *ta_state, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;

				carry_next = ta_state[id + b] & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] |= carry;
				}
			}   
		}

		// Decrement the states of each of those 32 Tsetlin Automata flagged in the active bit vector.
		__device__ inline void dec(unsigned int *ta_state, int chunk, unsigned int active)
		{
			unsigned int carry, carry_next;
			int id = chunk*STATE_BITS;
			carry = active;
			for (int b = 0; b < STATE_BITS; ++b) {
				if (carry == 0)
					break;
				carry_next = (~ta_state[id + b]) & carry; // Sets carry bits (overflow) passing on to next bit
				ta_state[id + b] = ta_state[id + b] ^ carry; // Performs increments with XOR
				carry = carry_next;
			}

			if (carry > 0) {
				for (int b = 0; b < STATE_BITS; ++b) {
					ta_state[id + b] &= ~carry;
				}
			} 
		}

		__device__ inline void update_clause_weight_log(curandState *localState, int tm_type, int number_of_outputs, float *clause_weight, int clause_output, int y, float class_sum)
		{
			int target = 1 - 2*(class_sum > y);
			
			if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, number_of_outputs-1)) {
				return;
			}

			int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		
			float absolute_prediction_error = fabsf(y - class_sum);
			if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
				if (target*sign > 0) {
					if (clause_output && abs(*clause_weight) < INT_MAX) {
						(*clause_weight) += sign;
					}
				} else if (target*sign < 0 && clause_output) {
					// Type II Feedback

					(*clause_weight) -= sign;
					
					#if FLIP_POLARITY == 0
						if (*clause_weight == 0) {
							*clause_weight += sign;
						}
					#endif
				}
			}
		}

		__device__ inline void update_component_hierarchy_log(curandState *localState, int number_of_outputs, float *clause_weight, unsigned int *ta_state, int component_output, int *X, int y, float class_sum)
		{
			int target = 1 - 2*(class_sum > y);

			if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, number_of_outputs-1)) {
				return;
			}

			int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		
			float absolute_prediction_error = fabsf(y - class_sum);
			if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
				if (target*sign > 0) {
					// Type I Feedback
					for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF; ++ta_chunk) {
						// Generate random bit values
						unsigned int la_feedback = 0;
						for (int b = 0; b < INT_SIZE; ++b) {
							if (curand_uniform(localState) <= 1.0/S) {
								la_feedback |= (1 << b);
							}
						}

						if (component_output) {
							#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
								inc(ta_state, ta_chunk, X[ta_chunk]);
							#else
								inc(ta_state, ta_chunk, X[ta_chunk] & (~la_feedback));
							#endif

							dec(ta_state, ta_chunk, (~X[ta_chunk]) & la_feedback);
						} else {
							dec(ta_state, ta_chunk, la_feedback);
						}
					}
				} else if (target*sign < 0 && component_output) {
					// Type II Feedback

					for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF; ++ta_chunk) {
						inc(ta_state, ta_chunk, (~X[ta_chunk]) & (~ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1]));
					}
				}
			}
		}

		// Evaluate example
		__global__ void evaluate_leaves_log(unsigned int *global_ta_state, float *component_weights, float *global_component_output, int depth, int *hierarchy_structure_factors, int *hierarchy_structure_type, int *X, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int *Xi = &X[(unsigned long long)example*LITERAL_CHUNKS];

			// Evaluate each clause component (leaf) in separate threads
			for (int clause_component = index; clause_component < CLAUSES*COMPONENTS; clause_component += stride) {
				int clause = clause_component / COMPONENTS;
				int component = clause_component % COMPONENTS;

				int component_remainder = component;
				int feature_chunk_base = 0;
				int ta_chunk_base = 0;
				int size_feature_chunk_base = 1;
				int size_ta_chunk_base = 1;
				for (int d = 0; d < depth-1; ++d) {
					int depth_d_node_index = component_remainder % hierarchy_structure_factors[d];
					component_remainder = component_remainder / hierarchy_structure_factors[d];

					if (hierarchy_structure_type[d] != 1) {
						feature_chunk_base += size_feature_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_feature_chunk_base *= hierarchy_structure_factors[d];
					}

					if (hierarchy_structure_type[d] != 2) {
						ta_chunk_base += size_ta_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_ta_chunk_base *= hierarchy_structure_factors[d];
					}
				}

				// Get state of current ta team component
				unsigned int *ta_state = &global_ta_state[clause*COMPONENTS*TA_CHUNKS_PER_LEAF*STATE_BITS + ta_chunk_base*STATE_BITS];

				// Evaluate clause component

				#if LOG_SCALE == 1
					float component_output = 0;
				#else
					float component_output = 1;
				#endif

				for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF-1; ++ta_chunk) {
					// Compare the TA state of the component (leaf) against the corresponding part of the feature vector
					if ((ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1] & Xi[feature_chunk_base + ta_chunk]) != ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1]) {
						#if LOG_SCALE == 1
							component_output = NEG_INFINITY;
						#else
							component_output = 0;
						#endif
							
						break;
					}
				}

				if ((ta_state[(TA_CHUNKS_PER_LEAF-1)*STATE_BITS + STATE_BITS - 1] & Xi[feature_chunk_base + TA_CHUNKS_PER_LEAF-1] & FILTER_HIERARCHICAL) != (ta_state[(TA_CHUNKS_PER_LEAF-1)*STATE_BITS + STATE_BITS - 1] & FILTER_HIERARCHICAL)) {
					#if LOG_SCALE == 1
						component_output = NEG_INFINITY;
					#else
						component_output = 0;
					#endif
				}

				global_component_output[clause_component] = component_output;
			}
		}

		__global__ void evaluate_and_groups_log(float *child_input, float *and_group_node_output, int number_of_and_group_nodes, int number_of_and_group_factors)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Multiply the votes from the children of each AND node
			for (int and_group_node = index; and_group_node < CLAUSES*number_of_and_group_nodes; and_group_node += stride) {
				// Multiply and factors

				#if LOG_SCALE == 1
					float and_group_vote_product = 0;
				#else
					float and_group_vote_product = 1;
				#endif

				for (int and_factor = 0; and_factor < number_of_and_group_factors; ++and_factor) {
					// Aggregate votes from each child node through multiplication
					
					#if LOG_SCALE == 1
				 		and_group_vote_product += child_input[and_group_node*number_of_and_group_factors + and_factor];
				 	#else
				 		and_group_vote_product *= child_input[and_group_node*number_of_and_group_factors + and_factor];
				 	#endif	
				}
				
				and_group_node_output[and_group_node] = and_group_vote_product;
			}
		}

		__global__ void propagate_and_group_false_truth_values_log(float *child_input, float *group_node_output, int number_of_group_nodes, int number_of_group_node_children)
		//__global__ void propagate_and_group_false_truth_values_log(int *child_input, int *group_node_output, int number_of_group_nodes, int number_of_group_node_children)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// If a group node is false, all children are made false.
			for (int group_node = index; group_node < CLAUSES*number_of_group_nodes; group_node += stride) {
				#if LOG_SCALE == 1
					if (group_node_output[group_node] == NEG_INFINITY) {
						for (int and_factor = 0; and_factor < number_of_group_node_children; ++and_factor) {
							if (child_input[group_node*number_of_group_node_children + and_factor] >= 0) {
								child_input[group_node*number_of_group_node_children + and_factor] = NEG_INFINITY;	
							}
						}
					}		
				#else
					if (group_node_output[group_node] == 0) {
						for (int and_factor = 0; and_factor < number_of_group_node_children; ++and_factor) {
							if (child_input[group_node*number_of_group_node_children + and_factor] > 0) {
								child_input[group_node*number_of_group_node_children + and_factor] = 0;	
							}
						}
					}
				#endif

				if (group_node_output[group_node] == -1) {
					for (int and_factor = 0; and_factor < number_of_group_node_children; ++and_factor) {
						child_input[group_node*number_of_group_node_children + and_factor] = -1;	
					}
				}
			}
		}

		__global__ void propagate_or_group_false_truth_values_log(curandState *state, float *child_input, float *group_node_output, int number_of_group_nodes, int number_of_group_node_children)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int non_zero_children[361];

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			// If a group node is false, all children are made false.
			for (int group_node = index; group_node < CLAUSES*number_of_group_nodes; group_node += stride) {
				int number_of_non_zero_children = 0;

				#if LOG_SCALE == 1
					if (group_node_output[group_node] == -1) {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							child_input[group_node*number_of_group_node_children + or_addend] = -1;	
						}
					}  else if (group_node_output[group_node] == NEG_INFINITY) {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							if (child_input[group_node*number_of_group_node_children + or_addend] >= 0) {
								child_input[group_node*number_of_group_node_children + or_addend] = NEG_INFINITY;	
							}
						}
					} else {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							if (child_input[group_node*number_of_group_node_children + or_addend] >= 0) {
								non_zero_children[number_of_non_zero_children] = or_addend;
								number_of_non_zero_children++;
							}
						}
					}
				#else
					if (group_node_output[group_node] == -1) {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							child_input[group_node*number_of_group_node_children + or_addend] = -1;	
						}
					}  else if (group_node_output[group_node] == 0) {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							if (child_input[group_node*number_of_group_node_children + or_addend] > 0) {
								child_input[group_node*number_of_group_node_children + or_addend] = 0;	
							}
						}
					} else {
						for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
							if (child_input[group_node*number_of_group_node_children + or_addend] > 0) {
								non_zero_children[number_of_non_zero_children] = or_addend;
								number_of_non_zero_children++;
							}
						}
					}
				#endif

				if (group_node_output[group_node] != -1) {
					int selected_child;
					if (number_of_non_zero_children > 0) {
						selected_child = non_zero_children[curand(&localState) % number_of_non_zero_children];
					} else {
						selected_child = curand(&localState) % number_of_group_node_children;
					}
					
					for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
						if (selected_child != or_addend) {
							child_input[group_node*number_of_group_node_children + or_addend] = -1;
						}
					}
				}
			}

			state[index] = localState;
		}

		__global__ void evaluate_or_alternatives_log(float *child_input, float *or_alternatives_node_output, int number_of_or_alternatives_nodes, int number_of_or_alternatives)
		//__global__ void evaluate_or_alternatives_log(int *child_input, int *or_alternatives_node_output, int number_of_or_alternatives_nodes, int number_of_or_alternatives)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Add up the votes from the children of each OR node
			for (int or_alternatives_node = index; or_alternatives_node < CLAUSES*number_of_or_alternatives_nodes; or_alternatives_node += stride) {
				#if LOG_SCALE == 1
					float or_alternatives_vote_max = NEG_INFINITY;
					for (int or_alternative = 0; or_alternative < number_of_or_alternatives; ++or_alternative) {
						if (child_input[or_alternatives_node * number_of_or_alternatives + or_alternative] > or_alternatives_vote_max) {
							or_alternatives_vote_max = child_input[or_alternatives_node * number_of_or_alternatives + or_alternative];
						}
					}

					if (or_alternatives_vote_max != NEG_INFINITY) {
						// Sum up votes from each or alternative
						float or_alternatives_vote_sum = 0;
						for (int or_alternative = 0; or_alternative < number_of_or_alternatives; ++or_alternative) {
							// Aggregates or alternatives through summation
							
							or_alternatives_vote_sum += exp2f(child_input[or_alternatives_node * number_of_or_alternatives + or_alternative] - or_alternatives_vote_max);
						}

						// Store vote sum as node output
						or_alternatives_node_output[or_alternatives_node] = or_alternatives_vote_max + log2f(or_alternatives_vote_sum);
					} else {
						or_alternatives_node_output[or_alternatives_node] = NEG_INFINITY;
					}
				#else
					// Sum up votes from each or alternative
					int or_alternatives_vote_sum = 0;
					for (int or_alternative = 0; or_alternative < number_of_or_alternatives; ++or_alternative) {
						// Aggregate same input or alternatives through summation						
						or_alternatives_vote_sum += child_input[or_alternatives_node * number_of_or_alternatives + or_alternative];
					}

					// Store vote sum as node output
					or_alternatives_node_output[or_alternatives_node] = or_alternatives_vote_sum;
				#endif
			}
		}

		__global__ void evaluate_final_log(int number_of_outputs, float *clause_output, float *clause_weights, float *class_sum)
		//__global__ void evaluate_final_log(int number_of_outputs, int *clause_output, int *clause_weights, int *class_sum)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Add up the votes from each clause
			for (int class_id = index; class_id < number_of_outputs; class_id += stride) {
				#if LOG_SCALE == 1
					float clause_output_max = NEG_INFINITY;
					for (int clause = 0; clause < CLAUSES; ++clause) {
						if (clause_output[clause] > clause_output_max) {
							clause_output_max = clause_output[clause];
						}
					}

					if (clause_output_max != NEG_INFINITY) {
						float weighted_clause_output_sum = 0;
						for (int clause = 0; clause < CLAUSES; ++clause) {
							weighted_clause_output_sum += clause_weights[class_id*CLAUSES + clause] * exp2f(clause_output[clause] - clause_output_max);
						}

						if (log2f(fabs(weighted_clause_output_sum)) + clause_output_max >= log2f(THRESHOLD)) {
							float sign = (1 - 2 * (weighted_clause_output_sum < 0));
							class_sum[class_id] = sign*THRESHOLD;
						} else {
							class_sum[class_id] = weighted_clause_output_sum * exp2f(clause_output_max);
						}
					} else {
						class_sum[class_id] = 0;
					}
				#else
					for (int clause = 0; clause < CLAUSES; ++clause) {
						if (clause_output[clause]) {
							float clause_weight = clause_weights[class_id*CLAUSES + clause];
							class_sum[class_id] += clause_weight * clause_output[clause];				
						}
					}
				#endif
			}
		}

		// Update state of Tsetlin Automata team
		__global__ void update_hierarchy_log(curandState *state, int number_of_outputs, unsigned int *global_ta_state, float *clause_weights, float *component_output, int depth, int *hierarchy_structure_factors, int *hierarchy_structure_type, float *class_sum, int *X, int *y, int example)
		//__global__ void update_hierarchy_log(curandState *state, int number_of_outputs, unsigned int *global_ta_state, int *clause_weights, int *component_output, int depth, int *hierarchy_structure_factors, int *hierarchy_structure_type, int *class_sum, int *X, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];
			
			int *Xi = &X[(unsigned long long)example*LITERAL_CHUNKS];

			// Calculate clause output first
			for (int clause_component = index; clause_component < CLAUSES*COMPONENTS; clause_component += stride) {
				if (component_output[clause_component] == -1) {
					continue;
				}

				int clause = clause_component / COMPONENTS;
				int component = clause_component % COMPONENTS;

				// Get state of current clause component
				// This one must be updated for reuse of TAs within OR group

				int component_remainder = component;
				int feature_chunk_base = 0;
				int ta_chunk_base = 0;
				int size_feature_chunk_base = 1;
				int size_ta_chunk_base = 1;
				for (int d = 0; d < depth-1; ++d) {
					int depth_d_node_index = component_remainder % hierarchy_structure_factors[d];
					component_remainder = component_remainder / hierarchy_structure_factors[d];

					if (hierarchy_structure_type[d] != 1) {
						feature_chunk_base += size_feature_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_feature_chunk_base *= hierarchy_structure_factors[d];
					}

					if (hierarchy_structure_type[d] != 2) {
						ta_chunk_base += size_ta_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_ta_chunk_base *= hierarchy_structure_factors[d];
					}
				}

				// Get state of current ta team component
				unsigned int *ta_state = &global_ta_state[clause*COMPONENTS*TA_CHUNKS_PER_LEAF*STATE_BITS + ta_chunk_base*STATE_BITS];

				for (unsigned long long class_id = 0; class_id < number_of_outputs; ++class_id) {
					float local_class_sum = class_sum[class_id];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}

					if (index == 0 && example == 0) {
						printf("%f\\n", local_class_sum);
					}

					#if LOG_SCALE == 1
						update_component_hierarchy_log(&localState, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], ta_state, component_output[clause_component] != NEG_INFINITY, &Xi[feature_chunk_base], y[example*number_of_outputs + class_id], local_class_sum);
					#else
						update_component_hierarchy(&localState, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], ta_state, component_output[clause_component] > 0, &Xi[feature_chunk_base], y[example*number_of_outputs + class_id], local_class_sum);
					#endif
				}
			}
		
			state[index] = localState;
		}

		// Update state of Tsetlin Automata team
		//__global__ void update_weights_log(curandState *state, int tm_type, int number_of_outputs, int *clause_weights, int *clause_output, float *class_sum, int *y, int example)
		__global__ void update_weights_log(curandState *state, int tm_type, int number_of_outputs, float *clause_weights, float *clause_output, float *class_sum, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				for (unsigned long long class_id = 0; class_id < number_of_outputs; ++class_id) {
					float local_class_sum = class_sum[class_id];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}

					#if LOG_SCALE == 1
						update_clause_weight_log(&localState, tm_type, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], clause_output[clause] != NEG_INFINITY, y[example*number_of_outputs + class_id], local_class_sum);
					#else
						update_clause_weight(&localState, tm_type, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], clause_output[clause] > 0, y[example*number_of_outputs + class_id], local_class_sum);
					#endif
				}
			}
		
			state[index] = localState;
		}

		////////////////////

		__device__ inline void update_clause_weight(curandState *localState, int tm_type, int number_of_outputs, int *clause_weight, int clause_output, int y, float class_sum)
		{
			int target = 1 - 2*(class_sum > y);
			
			if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, number_of_outputs-1)) {
				return;
			}

			int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		
			int absolute_prediction_error = abs(y - class_sum);
			if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
				if (target*sign > 0) {
					if (clause_output && abs(*clause_weight) < INT_MAX) {
						(*clause_weight) += sign;
					}
				} else if (target*sign < 0 && clause_output) {
					// Type II Feedback

					(*clause_weight) -= sign;
					
					#if FLIP_POLARITY == 0
						if (*clause_weight == 0) {
							*clause_weight += sign;
						}
					#endif
				}
			}
		}

		__device__ inline void update_component_hierarchy(curandState *localState, int number_of_outputs, int *clause_weight, unsigned int *ta_state, int component_output, int *X, int y, float class_sum)
		{
			int target = 1 - 2*(class_sum > y);
			
			if (target == -1 && curand_uniform(localState) > 1.0*Q/max(1, number_of_outputs-1)) {
				return;
			}
			
			int sign = (*clause_weight >= 0) - (*clause_weight < 0);
		
			int absolute_prediction_error = abs(y - class_sum);
			if (curand_uniform(localState) <= 1.0*absolute_prediction_error/(2*THRESHOLD)) {
				if (target*sign > 0) {
					// Type I Feedback
					for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF; ++ta_chunk) {
						// Generate random bit values
						unsigned int la_feedback = 0;
						for (int b = 0; b < INT_SIZE; ++b) {
							if (curand_uniform(localState) <= 1.0/S) {
								la_feedback |= (1 << b);
							}
						}

						if (component_output) {
							#if BOOST_TRUE_POSITIVE_FEEDBACK == 1
								inc(ta_state, ta_chunk, X[ta_chunk]);
							#else
								inc(ta_state, ta_chunk, X[ta_chunk] & (~la_feedback));
							#endif

							dec(ta_state, ta_chunk, (~X[ta_chunk]) & la_feedback);
						} else {
							dec(ta_state, ta_chunk, la_feedback);
						}
					}
				} else if (target*sign < 0 && component_output) {
					// Type II Feedback

					for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF; ++ta_chunk) {
						inc(ta_state, ta_chunk, (~X[ta_chunk]) & (~ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1]));
					}
				}
			}
		}

		// Evaluate example
		__global__ void evaluate_leaves(unsigned int *global_ta_state, int *component_weights, int *global_component_output, int depth, int *hierarchy_structure_factors, int *hierarchy_structure_type, int *X, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int *Xi = &X[(unsigned long long)example*LITERAL_CHUNKS];

			// Evaluate each clause component (leaf) in separate threads
			for (int clause_component = index; clause_component < CLAUSES*COMPONENTS; clause_component += stride) {
				int clause = clause_component / COMPONENTS;
				int component = clause_component % COMPONENTS;

				int component_remainder = component;
				int feature_chunk_base = 0;
				int ta_chunk_base = 0;
				int size_feature_chunk_base = 1;
				int size_ta_chunk_base = 1;
				for (int d = 0; d < depth-1; ++d) {
					int depth_d_node_index = component_remainder % hierarchy_structure_factors[d];
					component_remainder = component_remainder / hierarchy_structure_factors[d];

					if (hierarchy_structure_type[d] != 1) {
						feature_chunk_base += size_feature_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_feature_chunk_base *= hierarchy_structure_factors[d];
					}

					if (hierarchy_structure_type[d] != 2) {
						ta_chunk_base += size_ta_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_ta_chunk_base *= hierarchy_structure_factors[d];
					}
				}

				// Get state of current ta team component
				unsigned int *ta_state = &global_ta_state[clause*COMPONENTS*TA_CHUNKS_PER_LEAF*STATE_BITS + ta_chunk_base*STATE_BITS];

				// Evaluate clause component
				int component_output = 1;
				for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF-1; ++ta_chunk) {
					// Compare the TA state of the component (leaf) against the corresponding part of the feature vector
					if ((ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1] & Xi[feature_chunk_base + ta_chunk]) != ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1]) {
						component_output = 0;
						break;
					}
				}

				if ((ta_state[(TA_CHUNKS_PER_LEAF-1)*STATE_BITS + STATE_BITS - 1] & Xi[feature_chunk_base + TA_CHUNKS_PER_LEAF-1] & FILTER_HIERARCHICAL) != (ta_state[(TA_CHUNKS_PER_LEAF-1)*STATE_BITS + STATE_BITS - 1] & FILTER_HIERARCHICAL)) {
					component_output = 0;
				}

				global_component_output[clause_component] = component_output;
			}
		}

		__global__ void evaluate_or_groups(int *child_input, int *or_group_node_output, int number_of_or_group_nodes, int number_of_or_group_addends)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Add up the votes of each OR node child
			for (int or_group_node = index; or_group_node < CLAUSES*number_of_or_group_nodes; or_group_node += stride) {
				// Add OR addends
				int or_group_vote_sum = 0;
				int max_vote_sum = 0;
        
				for (int or_addend = 0; or_addend < number_of_or_group_addends; ++or_addend) {
					// Aggregate votes from each child node through addition

					if (child_input[or_group_node*number_of_or_group_addends + or_addend] > max_vote_sum) {
						max_vote_sum = child_input[or_group_node*number_of_or_group_addends + or_addend];
					}

					int previous_or_group_vote_sum = or_group_vote_sum; 
					or_group_vote_sum += child_input[or_group_node*number_of_or_group_addends + or_addend];
					if (or_group_vote_sum < 0) {
						printf("OR* OVERFLOW %d -> %d\\n", previous_or_group_vote_sum, or_group_vote_sum);						
						or_group_vote_sum = previous_or_group_vote_sum;
					}
				}

				//or_group_node_output[or_group_node] = or_group_vote_sum;
				or_group_node_output[or_group_node] = max_vote_sum;
			}
		}

		__global__ void evaluate_and_groups(int *child_input, int *and_group_node_output, int number_of_and_group_nodes, int number_of_and_group_factors)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Multiply the votes from the children of each AND node
			for (int and_group_node = index; and_group_node < CLAUSES*number_of_and_group_nodes; and_group_node += stride) {
				// Multiply and factors
				int and_group_vote_product = 1;
				int max_vote_sum = 0;
				for (int and_factor = 0; and_factor < number_of_and_group_factors; ++and_factor) {
					// Aggregate votes from each child node through multiplication
					
					int previous_and_group_vote_product = and_group_vote_product;
					and_group_vote_product *= child_input[and_group_node*number_of_and_group_factors + and_factor];

					if (child_input[and_group_node*number_of_and_group_factors + and_factor] > max_vote_sum) {
						max_vote_sum = child_input[and_group_node*number_of_and_group_factors + and_factor];
					}

					if (and_group_vote_product < 0) {
						printf("AND OVERFLOW %d -> %d\\n", previous_and_group_vote_product, and_group_vote_product);						
						and_group_vote_product = previous_and_group_vote_product;
					}
				}

				// Store and group product as node output
				and_group_node_output[and_group_node] = and_group_vote_product;
				//and_group_node_output[and_group_node] = max_vote_sum;
			}
		}

		__global__ void propagate_and_group_false_truth_values(int *child_input, int *group_node_output, int number_of_group_nodes, int number_of_group_node_children)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// If a group node is false, all children are made false.
			for (int group_node = index; group_node < CLAUSES*number_of_group_nodes; group_node += stride) {
				if (group_node_output[group_node] == 0) {
					for (int and_factor = 0; and_factor < number_of_group_node_children; ++and_factor) {
						if (child_input[group_node*number_of_group_node_children + and_factor] > 0) {
							child_input[group_node*number_of_group_node_children + and_factor] = 0;	
						}
					}
				}
			}
		}

		__global__ void propagate_or_group_false_truth_values(curandState *state, int *child_input, int *group_node_output, int number_of_group_nodes, int number_of_group_node_children)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			int non_zero_children[361];

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			// If a group node is false, all children are made false.
			for (int group_node = index; group_node < CLAUSES*number_of_group_nodes; group_node += stride) {
				int number_of_non_zero_children = 0;

				if (group_node_output[group_node] == -1) {
					for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
						child_input[group_node*number_of_group_node_children + or_addend] = -1;	
					}
				}  else if (group_node_output[group_node] == 0) {
					for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
						if (child_input[group_node*number_of_group_node_children + or_addend] > 0) {
							child_input[group_node*number_of_group_node_children + or_addend] = 0;	
						}
					}
				} else {
					for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
						if (child_input[group_node*number_of_group_node_children + or_addend] > 0) {
							non_zero_children[number_of_non_zero_children] = or_addend;
							number_of_non_zero_children++;
						}
					}
				}

				if (group_node_output[group_node] != -1) {
					int selected_child;
					if (number_of_non_zero_children > 0) {
						selected_child = non_zero_children[curand(&localState) % number_of_non_zero_children];
					} else {
						selected_child = curand(&localState) % number_of_group_node_children;
					}
					
					for (int or_addend = 0; or_addend < number_of_group_node_children; ++or_addend) {
						if (selected_child != or_addend) {
							child_input[group_node*number_of_group_node_children + or_addend] = -1;
						}
					}
				}
			}

			state[index] = localState;
		}

		__global__ void evaluate_or_alternatives(int *child_input, int *or_alternatives_node_output, int number_of_or_alternatives_nodes, int number_of_or_alternatives)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Add up the votes from the children of each OR node
			for (int or_alternatives_node = index; or_alternatives_node < CLAUSES*number_of_or_alternatives_nodes; or_alternatives_node += stride) {
				// Sum up votes from each or alternative
				int or_alternatives_vote_sum = 0;
				for (int or_alternative = 0; or_alternative < number_of_or_alternatives; ++or_alternative) {
					// Aggregate same input or alternatives through summation
					

					int previous_or_alternatives_vote_sum = or_alternatives_vote_sum;
					or_alternatives_vote_sum += child_input[or_alternatives_node * number_of_or_alternatives + or_alternative];

					if (or_alternatives_vote_sum < 0) {
						printf("OR OVERFLOW %d -> %d\\n", previous_or_alternatives_vote_sum, or_alternatives_vote_sum);
						or_alternatives_vote_sum = previous_or_alternatives_vote_sum;
					}
				}

				// Store vote sum as node output
				or_alternatives_node_output[or_alternatives_node] = or_alternatives_vote_sum;
			}
		}

		__global__ void evaluate_final(int number_of_outputs, int *child_input, int *clause_weights, int *class_sum)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// Add up the votes from each clause
			for (int clause = index; clause < CLAUSES; clause += stride) {
				if (child_input[clause]) {
					for (int class_id = 0; class_id < number_of_outputs; ++class_id) {
						int clause_weight = clause_weights[class_id*CLAUSES + clause];
						atomicAdd(&class_sum[class_id], clause_weight * child_input[clause]);					
					}
				}
			}
		}

		// Update state of Tsetlin Automata team
		__global__ void update_hierarchy(curandState *state, int number_of_outputs, unsigned int *global_ta_state, int *clause_weights, int *component_output, int depth, int *hierarchy_structure_factors, int *hierarchy_structure_type, int *class_sum, int *X, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];
			
			int *Xi = &X[(unsigned long long)example*LITERAL_CHUNKS];

			// Calculate clause output first
			for (int clause_component = index; clause_component < CLAUSES*COMPONENTS; clause_component += stride) {
				if (component_output[clause_component] == -1) {
					continue;
				}

				int clause = clause_component / COMPONENTS;
				int component = clause_component % COMPONENTS;

				// Get state of current clause component
				// This one must be updated for reuse of TAs within OR group

				int component_remainder = component;
				int feature_chunk_base = 0;
				int ta_chunk_base = 0;
				int size_feature_chunk_base = 1;
				int size_ta_chunk_base = 1;
				for (int d = 0; d < depth-1; ++d) {
					int depth_d_node_index = component_remainder % hierarchy_structure_factors[d];
					component_remainder = component_remainder / hierarchy_structure_factors[d];

					if (hierarchy_structure_type[d] != 1) {
						feature_chunk_base += size_feature_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_feature_chunk_base *= hierarchy_structure_factors[d];
					}

					if (hierarchy_structure_type[d] != 2) {
						ta_chunk_base += size_ta_chunk_base * depth_d_node_index * TA_CHUNKS_PER_LEAF;
						size_ta_chunk_base *= hierarchy_structure_factors[d];
					}
				}

				// Get state of current ta team component
				unsigned int *ta_state = &global_ta_state[clause*COMPONENTS*TA_CHUNKS_PER_LEAF*STATE_BITS + ta_chunk_base*STATE_BITS];

				for (unsigned long long class_id = 0; class_id < number_of_outputs; ++class_id) {
					int local_class_sum = class_sum[class_id];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}

					update_component_hierarchy(&localState, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], ta_state, component_output[clause_component], &Xi[feature_chunk_base], y[example*number_of_outputs + class_id], local_class_sum);
				}
			}
		
			state[index] = localState;
		}

		// Update state of Tsetlin Automata team
		__global__ void update_weights(curandState *state, int tm_type, int number_of_outputs, int *clause_weights, float *clause_output, float *class_sum, int *y, int example)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			/* Copy state to local memory for efficiency */  
			curandState localState = state[index];

			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				for (unsigned long long class_id = 0; class_id < number_of_outputs; ++class_id) {
					int local_class_sum = class_sum[class_id];
					if (local_class_sum > THRESHOLD) {
						local_class_sum = THRESHOLD;
					} else if (local_class_sum < -THRESHOLD) {
						local_class_sum = -THRESHOLD;
					}
					update_clause_weight(&localState, tm_type, number_of_outputs, &clause_weights[class_id*CLAUSES + clause], clause_output[clause], y[example*number_of_outputs + class_id], local_class_sum);
				}
			}
		
			state[index] = localState;
		}
    }
"""

code_prepare = """
	extern "C"
    {
		__global__ void prepare_weights(curandState *state, int tm_type, int number_of_outputs, int *clause_weights)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			curandState localState = state[index];

			for (unsigned long long clause = index; clause < CLAUSES; clause += stride) {
				for (unsigned long long class_id = 0; class_id < number_of_outputs; ++class_id) {
					#if NEGATIVE_CLAUSES == 1
						if (tm_type == COALESCED_TM) {
							clause_weights[class_id*CLAUSES + clause] = 1 - 2 * (curand(&localState) % 2);
						} else {
							clause_weights[class_id*CLAUSES + clause] = 1 - 2 * (clause % 2);
						}
					#else
						clause_weights[class_id*CLAUSES + clause] = 1;
					#endif
				}
			}
				
			state[index] = localState;
		}

		__global__ void prepare_hierarchy(curandState *state, int number_of_outputs, unsigned int *global_ta_state, int *clause_weights)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			curandState localState = state[index];

			// Evaluate each clause component (leaf) in separate threads
			for (int clause_component = index; clause_component < CLAUSES*COMPONENTS; clause_component += stride) {
				// Get state of current clause component
				unsigned int *ta_state = &global_ta_state[clause_component*TA_CHUNKS_PER_LEAF*STATE_BITS];
				for (int ta_chunk = 0; ta_chunk < TA_CHUNKS_PER_LEAF; ++ta_chunk) {
					for (int b = 0; b < STATE_BITS-1; ++b) {
						ta_state[ta_chunk*STATE_BITS + b] = ~0;
					}
					ta_state[ta_chunk*STATE_BITS + STATE_BITS - 1] = 0;
				}
			}

			state[index] = localState;
		}
	}
"""

code_encode = """
	#include <curand_kernel.h>

	extern "C"
    {
		__global__ void prepare_encode_hierarchy(unsigned int *X, unsigned int *encoded_X, int number_of_literal_chunks, int number_of_examples)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			for (unsigned long long i = index; i < number_of_examples * number_of_literal_chunks; i += stride) {
				encoded_X[i] = 0;
			}
		}

		__global__ void encode_hierarchy(unsigned int *X, unsigned int *encoded_X, int number_of_features, int number_of_literal_chunks, int number_of_leaves, int number_of_features_per_leaf, int number_of_literal_chunks_per_leaf, int append_negated, int number_of_examples)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			unsigned int *Xi;
			unsigned int *encoded_Xi;

			for (unsigned long long i = index; i < number_of_examples; i += stride) {
				Xi = &X[i*number_of_features];
				encoded_Xi = &encoded_X[i*number_of_literal_chunks];

				for (int j = 0; j < number_of_features / number_of_features_per_leaf; ++j) {
					for (int k = 0; k < number_of_features_per_leaf; ++k) {
						if (Xi[j*number_of_features_per_leaf + k] == 1) {
							int leaf_chunk_nr = k / 32;
							int leaf_chunk_pos = k % 32;
							encoded_Xi[j*number_of_literal_chunks_per_leaf + leaf_chunk_nr] |= (1 << leaf_chunk_pos);
						} else if (append_negated) {
							int leaf_chunk_nr = (k + number_of_features_per_leaf) / 32;
							int leaf_chunk_pos = (k + number_of_features_per_leaf) % 32;
							encoded_Xi[j*number_of_literal_chunks_per_leaf + leaf_chunk_nr] |= (1 << leaf_chunk_pos);
						}
					}
				}
			} 
		}
	}
"""

code_transform = """
	extern "C"
    {
		// Transform examples
		__global__ void transform(unsigned int *global_ta_state, int *X, int *transformed_X)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			int stride = blockDim.x * gridDim.x;

			// To be updated
		}
	}
"""

code_clauses = """
	extern "C" __global__ void get_ta_states(const unsigned int* global_ta_state, unsigned int* unpacked_states) {
		int index = blockIdx.x * blockDim.x + threadIdx.x;
		int stride = blockDim.x * gridDim.x;

		for (unsigned long long i = index; i < 1LLU * CLAUSES * COMPONENTS * LITERALS_PER_LEAF; i += stride) {
			unsigned long long clause = i / (COMPONENTS * LITERALS_PER_LEAF);
			unsigned long long comp   = (i / LITERALS_PER_LEAF) % COMPONENTS;
			unsigned long long ta_idx = i % LITERALS_PER_LEAF;

			int chunk   = ta_idx / 32;
			int bit_pos = ta_idx % 32;

			unsigned int state = 0;
			for (int b = 0; b < STATE_BITS; ++b) {
				unsigned int plane = global_ta_state[
					(clause * COMPONENTS * TA_CHUNKS_PER_LEAF * STATE_BITS) +
					(comp   * TA_CHUNKS_PER_LEAF * STATE_BITS) +
					(chunk  * STATE_BITS) + b
				];
				if (plane & (1U << bit_pos)) state |= (1U << b);
			}

			unpacked_states[(clause * COMPONENTS * LITERALS_PER_LEAF) + (comp * LITERALS_PER_LEAF) + ta_idx] = state;
		}
	}
"""
