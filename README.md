# Hierarchical Tsetlin Machine in CUDA
Implements the [Hierarchical Tsetlin Machine](https://github.com/cair/HierarchicalTsetlinMachine) in CUDA.

<p align="center">
  <img width="60%" src="https://github.com/cair/PyHierarchicalTsetlinMachineCUDA/blob/main/figures/Visualization_voting.png">
</p>

## Installation

```bash
pip install PyHierarchicalTsetlinMachineCUDA
```
or
```bash
git clone git@github.com:cair/PyHierarchicalTsetlinMachineCUDA.git
cd PyHierarchicalTsetlinMachineCUDA
python ./setup.py sdist
pip install dist/PyHierarchicalTsetlinMachineCUDA-0.2.2.tar.gz 
```

## Examples

### Noisy Parity Demo

#### Code: NoisyParityDemo.py

```python
from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

clauses = 32
s = 25.0
T = 250

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = TsetlinMachine(clauses, T, s, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, 3), (tm.OR_ALTERNATIVES, 3), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 3), (tm.AND_GROUP, 2)))

print("\nAccuracy over 1000 epochs:\n")
for e in range(1000):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	tm.print_hierarchy()

	print("\n#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (e+1, result, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
python ./examples/NoisyParityData.py
python ./examples/NoisyParityDemo.py

Accuracy over 1000 epochs:

#1 Accuracy: 51.19% Training: 21.88s Testing: 1.37s
#2 Accuracy: 51.94% Training: 21.91s Testing: 1.37s
#3 Accuracy: 53.46% Training: 21.91s Testing: 1.37s
...
CLAUSE 1: (((((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)) ∧ ((¬x0 ∧ ¬x1) ∨ (x0 ∧ x1) ∨ (x0 ∧ x1))) ∨ (((¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1) ∨ (x0 ∧ x1)) ∧ ((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1))) ∨ (((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)) ∧ (¬x0 ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)))) ∧ ((((x0 ∧ ¬x1) ∨ (x0 ∧ ¬x1) ∨ (x1 ∧ ¬x0)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1))) ∨ (((x1 ∧ ¬x0) ∨ (x0 ∧ ¬x1) ∨ (x0 ∧ ¬x1)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (x0 ∧ x1))) ∨ (((x1 ∧ ¬x0) ∨ (x1 ∧ ¬x0) ∨ (x0 ∧ ¬x1)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1)))))
...
#778 Accuracy: 99.71% Training: 32.57s Testing: 2.18s
```

### Hex Demo

<p align="center">
  <img width="60%" src="https://github.com/cair/PyHierarchicalTsetlinMachineCUDA/blob/hex_description/figures/Hex-board-11x11.png">
	Source: https://en.wikipedia.org/wiki/Hex_(board_game)
</p>

#### Code: HexDemo.py

```python
from PyHierarchicalTsetlinMachineCUDA.tm import TsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm
import argparse

def default_args(**kwargs):
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--clauses", default=2000, type=int)
    parser.add_argument("--T", default=9000, type=int)
    parser.add_argument("--s", default=44.0, type=float)
    parser.add_argument("--q", default=1.0, type=float)
    parser.add_argument("--board_dim", default=10, type=int)
    parser.add_argument("--boost", default=1, type=int)
    parser.add_argument("--number_of_state_bits", default=10, type=int)
    parser.add_argument("--or_alternatives", default=60, type=int)
  
    args = parser.parse_args()
    for key, value in kwargs.items():
        if key in args.__dict__:
            setattr(args, key, value)
    return args

args = default_args()

data = np.loadtxt("./examples/hex_data.txt").astype(np.uint32)
X_train = data[:int(len(data)*0.8),0:-1]
Y_train = data[:int(len(data)*0.8),-1]

X_test = data[int(len(data)*0.8):,0:-1]
Y_test = data[int(len(data)*0.8):,-1]

tsetlin_machine = TsetlinMachine(args.clauses, args.T, args.s, weighted_clauses=False, number_of_state_bits=args.number_of_state_bits, boost_true_positive_feedback=args.boost, hierarchy_structure=((tm.AND_GROUP, 72), (tm.OR_ALTERNATIVES, args.or_alternatives), (tm.AND_GROUP, 4)))

print("\nAccuracy over %d epochs:\n" % (args.epochs))
for e in range(args.epochs):
	start_training = time()
	for b in range(10):
		tsetlin_machine.fit(X_train[b*len(Y_train)//10:(b+1)*len(Y_train)//10], Y_train[b*len(Y_train)//10:(b+1)*len(Y_train)//10], epochs=1, incremental=True)
	stop_training = time()

	start_testing = time()
	result_testing = 100*(tsetlin_machine.predict(X_test) == Y_test).mean()
	stop_testing = time()

	result_training = 100*(tsetlin_machine.predict(X_train) == Y_train).mean()

	print("#%d/%d Training Accuracy: %.2f%% Testing Accuracy: %.2f%% Training Time: %.2fs Testing Time: %.2fs" % (e+1, b+1, result_training, result_testing, stop_training-start_training, stop_testing-start_testing))
```

#### Output

```bash
#304 Training Accuracy: 100.00% Testing Accuracy: 99.24% Training Time: 32.15s Testing Time: 6.07s
#305 Training Accuracy: 100.00% Testing Accuracy: 99.26% Training Time: 32.13s Testing Time: 6.07s
#306 Training Accuracy: 100.00% Testing Accuracy: 99.25% Training Time: 32.11s Testing Time: 6.07s
```
## Paper

_A Tsetlin Machine for Logical Learning and Reasoning With AND-OR Hierarchies_. Ole-Christoffer Granmo, et al., 2026. (Forthcoming)

## Licence

MIT License

Copyright (c) 2026 Ole-Christoffer Granmo and the University of Agder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
