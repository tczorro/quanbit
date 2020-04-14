import numpy as np
from abc import ABCMeta, abstractstaticmethod, abstractmethod


class Operators:
    def __init__(self, op_mtr):
        self._op = op_mtr
        self._n = int(np.log(len(op_mtr.shape)) / np.log(2))

    @property
    def n_target(self):
        return self._n

    @property
    def matrix(self):
        return self._op

    def __call__(self, target, indices=[0]):
        # indices = []
        # for i in index:
        #     indices.extend([2 ** i - 1, 2 ** i])
        # indices = np.array(indices)
        # print('ind', indices)
        # print('target', target.state[indices])
        # target._state[indices] = self._op @ target.state[indices]
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

# /simga 2
Pauli_Y = Operators(np.array([[0, -1j],[1j, 0]], dtype=complex))

# /simga 3
Pauli_Z = Operators(np.array([[1, 0], [0, -1]], dtype=complex))


def Measure(Circuit):
    ...
