# Copyright (c) 2019, Derrick Yang
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Operators and Measure function.

Attributes
----------
BellBasis : Operator, callable
    Description
CNOT : Operator, callable
    Description
Hadamard, H : Operator, callable
    Description
Pauli_X, X : Operator, callable
    Description
Pauli_Y, Y : Operator, callble
    Description
Pauli_Z, Z : Operator, callble
    Description
"""

import numpy as np
from abc import ABCMeta, abstractstaticmethod, abstractmethod


class Operators:
    def __init__(self, op_mtr):
        """Operator class for manipulate a Circuit instance.

        Parameters
        ----------
        op_mtr : np.ndarray(2,...)
            Unitary matrix to operator on circuit
        """
        self._op = op_mtr
        self._n = int(np.log(len(op_mtr.shape)) / np.log(2))

    @property
    def n_target(self):
        """int: number of qubit needed for this operator."""
        return self._n

    @property
    def matrix(self):
        """np.ndarray(2,...): matrix representation of the operation."""
        return self._op

    def __call__(self, target, indices=[0]):
        """Implement

        Parameters
        ----------
        target : Circuit
            target qubit(s) Circuit instance
        indices : list, optional
            Targeted indices for qubit(s) in the system

        Returns
        -------
        Circuit
            New Circuit instance after the operation is performed
        """
        return target.apply_operator(self, indices)


CNOT = Operators(
    np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
    ).reshape(2, 2, 2, 2)
)

BellBasis = Operators(
    1
    / np.sqrt(2)
    * np.array(
        [[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1, -1, 0]], dtype=complex
    ).reshape(2, 2, 2, 2)
)

Hadamard = Operators(1 / np.sqrt(2.0) * np.array([[1, 1], [1, -1]], dtype=complex))

H = Hadamard

# /simga 1
Pauli_X = Operators(np.array([[0, 1], [1, 0]], dtype=complex))
X = Pauli_X

# /simga 2
Pauli_Y = Operators(np.array([[0, -1j], [1j, 0]], dtype=complex))
Y = Pauli_Y

# /simga 3
Pauli_Z = Operators(np.array([[1, 0], [0, -1]], dtype=complex))
Z = Pauli_Z


def Measure(Circuit, indices):
    """Measure a Circuit at given indices for its value(s)

    Parameters
    ----------
    Circuit : Circuit
        a Circuit instance
    indices : list or list-like
        indices for measuring the circuit wafefunction

    Returns
    -------
    tuple(tuple, Circuit)
        the first element is the tuple of measured cubit
        the second element is the leftover cirtuit wavefunction
    """
    return Circuit.measure(indices)
