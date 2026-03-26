# PyHierarchicalTsetlinMachineCUDA
Implements the Hierarchical Tsetlin Machine in CUDA

## Installation

```bash
pip install pyhierarchicaltsetlinmachinecuda
```
or
```bash
git clone git@github.com:cair/PyHierarchicalTsetlinMachineCUDA.git
cd PyHierarchicalTsetlinMachineCUDA
python ./setup.py sdist
pip install dist/pyhierarchicaltsetlinmachinecuda-0.2.0.tar.gz 
```

## Examples

### Noisy Parity Demo

#### Code: NoisyParityDemo.py

```python
from PyHierarchicalTsetlinMachineCUDA.tm import MultiClassTsetlinMachine
import numpy as np
from time import time
import PyHierarchicalTsetlinMachineCUDA.tm as tm

train_data = np.loadtxt("./examples/NoisyParityTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1]
Y_train = train_data[:,-1]

test_data = np.loadtxt("./examples/NoisyParityTestingData.txt").astype(np.uint32)
X_test = test_data[:,0:-1]
Y_test = test_data[:,-1]

tm = MultiClassTsetlinMachine(32, 1500, 30.1, tm_type=tm.VANILLA_TM, number_of_state_bits=8, boost_true_positive_feedback=0, hierarchy_structure=((tm.AND_GROUP, 3), (tm.OR_ALTERNATIVES, 10), (tm.AND_GROUP, 2), (tm.OR_ALTERNATIVES, 2), (tm.AND_GROUP, 2)))

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

...
#86 Accuracy: 99.80% Training: 24.39s Testing: 1.52s
#87 Accuracy: 99.92% Training: 24.41s Testing: 1.52s
#88 Accuracy: 99.87% Training: 24.39s Testing: 1.51s
#89 Accuracy: 99.38% Training: 24.40s Testing: 1.52s
#90 Accuracy: 99.94% Training: 24.39s Testing: 1.52s
```

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
