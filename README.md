# Quanbit
<!-- [![codecov](https://codecov.io/gh/theochem/grid/branch/master/graph/badge.svg)](https://codecov.io/gh/theochem/grid)-->
[![Build Status](https://travis-ci.com/tczorro/quanbit.svg?branch=master)](https://travis-ci.org/tczorro/quanbit)
[![Python Version](https://img.shields.io/badge/python-3.6%2B-blue.svg)](https://docs.python.org/3/whatsnew/3.6.html)
[![GitHub](https://img.shields.io/github/license/tczorro/quanbit)](https://github.com/tczorro/quanbit/blob/master/LICENSE)
[![Code Style](https://img.shields.io/badge/code%20style-black-black.svg)](https://black.readthedocs.io/en/stable/)

## About
**Quanbit** is a pythonic package for simulating quantum computer. It faciliate basic quantum algorithm exploration on a classic computer.

## Platform
**Quanbit** is a pure python package supporting `Windows`, `Linux` and `MacOS`.

## Example: [Quantum Teleportation](https://en.wikipedia.org/wiki/Quantum_teleportation#Formal_presentation)
```python
from quanbit import X, Y, Z, BellBasis
from quanbit import qubit, bell_state, Measure
import numpy as np

# Alice has an unknown qubit C
theta, phi, alpha = np.random.rand(3)
qubit_C = qubit(theta, phi, alpha)

# To teleport the qubit, Alice and Bob need to share a maximally entangled state
# Anyone of the four states is sufficient, we choose 1/sqrt(2) (|00> + |11>) here
qubit_AB = bell_state(0)

# Now the total state is, where @ represent tensor product:
total_state = qubit_C @ qubit_AB

# Project the state of Alice's two qubits as a superpositions of the Bell basis
total_state = BellBasis(total_state, indices=[0, 1])

# Measuring her two cubits in Bell basis
CA, B_state = Measure(total_state, indices=[0, 1])

# Rotate Bob's state based on the measurement result
# if CA == (0, 0), no change need to be made
# when CA is in state \Psi+
if CA == (0, 1):
    B_state = X(B_state)
# when CA is in state \Psi-
elif CA == (1, 0):
    B_state = Z(X(B_state))
# when CA is in state \Phi-
elif CA == (1, 1):
    B_state = Z(B_state)
# Now Bob's state is the same as Alice initial qubit_C.
```

## License
**Quanbit** is distributed under [BSD 3-Clause](https://github.com/tczorro/quanbit/blob/master/LICENSE).

## Dependence
* Installation requirements: `numpy`
* Testing requirement: `pytest`
* QA requirement: `tox`

## Installation
### Install with PIP
To install Quanbit with pip:
```bash
pip install quanbit
```

### Install from source
To run tests:
```bash
pytest quanbit
```
To install Quanbit to system:
```bash
pip install .
```
### Local build and Testing
To install editable Quanbit locally:
```bash
pip install -e .
```
To run tests:
```bash
pytest tests
```

## Quality Assurance
To run QA locally:
```bash
tox
```
