# Hierarchical Tsetlin Machine in CUDA
Implements the [Hierarchical Tsetlin Machine](https://github.com/cair/HierarchicalTsetlinMachine) in CUDA.

<p align="center">
  <img width="70%" src="https://github.com/cair/PyHierarchicalTsetlinMachineCUDA/blob/main/figures/Clause_Plot_Mayur_Shende.png">
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

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = TsetlinMachine(32, 3000, 30.1,
	number_of_state_bits=8,
	boost_true_positive_feedback=0,
	hierarchy_structure=(
		(tm.AND_GROUP, 3),
		(tm.OR_ALTERNATIVES, 10),
		(tm.AND_GROUP, 2),
		(tm.OR_ALTERNATIVES, 2),
		(tm.AND_GROUP, 2)
	)
)

print("\nAccuracy over 500 epochs:\n")
for i in range(500):
	start_training = time()
	tm.fit(X_train, Y_train, epochs=10, incremental=True)
	stop_training = time()

	start_testing = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	stop_testing = time()

	print("#%d Accuracy: %.2f%% Training: %.2fs Testing: %.2fs" % (i+1, result, stop_training-start_training, stop_testing-start_testing))
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
 CLAUSE #0: (((((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)) ∧ ((¬x0 ∧ ¬x1) ∨ (x0 ∧ x1) ∨ (x0 ∧ x1))) ∨ (((¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1) ∨ (x0 ∧ x1)) ∧ ((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1))) ∨ (((x0 ∧ x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)) ∧ ((¬x0 ∧ ¬x1) ∨ (x0 ∧ x1) ∨ (¬x0 ∧ ¬x1)))) ∧ ((((x0 ∧ ¬x1) ∨ (x0 ∧ ¬x1) ∨ (x1 ∧ ¬x0)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1))) ∨ (((x1 ∧ ¬x0) ∨ (x0 ∧ ¬x1) ∨ (x0 ∧ ¬x1)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (x0 ∧ x1))) ∨ (((x1 ∧ ¬x0) ∨ (x1 ∧ ¬x0) ∨ (x0 ∧ ¬x1)) ∧ ((x0 ∧ x1) ∨ (¬x0 ∧ ¬x1) ∨ (¬x0 ∧ ¬x1)))))
...
#395 Accuracy: 99.64% Training: 22.86s Testing: 1.42s
```

### Paper

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
