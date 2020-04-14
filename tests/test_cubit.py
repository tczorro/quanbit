from unittest import TestCase
from quanbit.circuit import Circuit, cubit
from quanbit.operators import CNOT, H, BellBasis, Pauli_X, Pauli_Y, Pauli_Z

import numpy as np
from numpy.testing import assert_allclose

class TestCubit(TestCase):

    def test_cubit(self):
        """Test generate a normal cubit."""
        cubit1 = Circuit(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
        cubit2 = Circuit(np.array([1, 0]))
        circuit1 = cubit1 @ cubit2
        assert_allclose(circuit1.state, [[1 / np.sqrt(2), 0],[1 / np.sqrt(2), 0]])

    def test_h_gate(self):
        cubit1 = Circuit(np.array([1, 0])) # state |0>
        cubit1 = H(cubit1)
        assert_allclose(cubit1.state, [1 / np.sqrt(2), 1 / np.sqrt(2)])
        cubit2 = Circuit(np.array([0, 1])) # state |1>
        cubit2 = H(cubit2)
        assert_allclose(cubit2.state, [1 / np.sqrt(2), -1 / np.sqrt(2)])

    def test_quantum_teleportation(self):
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.], [0, 1/ np.sqrt(2)]]))
        # generater a random state
        a = np.random.rand(3)
        alice = cubit(*a)
        phi = alice @ bell_state
        phi = CNOT(phi, [0, 1])
        phi = H(phi, [0])
        bits, result_state = phi.measure([0, 1])
        bit1, bit2 = bits
        if bit2:
            result_state = Pauli_X(result_state)
        if bit1:
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_2(self):
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.], [0, 1/ np.sqrt(2)]]))
        alice = cubit(1, 1)
        phi = alice @ bell_state
        phi = BellBasis(phi, [0, 1])
        bits, result_state = phi.measure([0, 1])
        bit1, bit2 = bits
        if bit1:
            result_state = Pauli_X(result_state)
        if bit2:
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)


