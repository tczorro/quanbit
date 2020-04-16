# Copyright (c) 2019, Derrick Yang
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from quanbit.circuit import Circuit, qubit
from quanbit.operators import CNOT, H, BellBasis, Pauli_X, Pauli_Y, Pauli_Z
from quanbit.measure import Measure

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal


class Testqubit(TestCase):
    def test_qubit(self):
        """Test generate a normal qubit."""
        qubit1 = Circuit(np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))
        qubit2 = Circuit(np.array([1, 0]))
        circuit1 = qubit1 @ qubit2
        assert_allclose(circuit1.state, [[1 / np.sqrt(2), 0], [1 / np.sqrt(2), 0]])

    def test_pauli_gate(self):
        """Test pauli x, y, z gate."""
        rand = np.random.rand(3)
        qubit1 = qubit(*rand)
        # test x gate
        new_bit = Pauli_X(qubit1)
        assert_allclose(new_bit.state[::-1], qubit1.state)
        # test y gate
        new_bit = Pauli_Y(qubit1)
        assert_allclose(new_bit.state[::-1] * np.array([-1j, 1j]), qubit1.state)
        # test z gate
        new_bit = Pauli_Z(qubit1)
        assert_allclose(new_bit.state * np.array([1, -1]), qubit1.state)

    def test_h_gate(self):
        """Test Hadamard works properly."""
        qubit1 = Circuit(np.array([1, 0]))  # state |0>
        qubit1 = H(qubit1)
        assert_allclose(qubit1.state, [1 / np.sqrt(2), 1 / np.sqrt(2)])
        qubit2 = Circuit(np.array([0, 1]))  # state |1>
        qubit2 = H(qubit2)
        assert_allclose(qubit2.state, [1 / np.sqrt(2), -1 / np.sqrt(2)])

    def test_cnot(self):
        """Test CNOT works properly no matter what sequence."""
        with self.assertWarns(RuntimeWarning):
            two_qubit = Circuit(np.random.rand(4).reshape(2, 2))
        # test the prob renormalized to 1
        assert_almost_equal(np.sum(two_qubit.state ** 2), 1)
        # apply CNOT on 0, 1 index
        ref_state = two_qubit.state.copy()
        ref_state[1] = ref_state[1][::-1]
        new_state = CNOT(two_qubit, [0, 1])
        assert_allclose(new_state.state, ref_state)

        # apply CNOT on 1, 0 index
        new_state = CNOT(two_qubit, [1, 0])
        ref_state = two_qubit.state.copy()
        ref_state[:, 1] = ref_state[:, 1][::-1]
        assert_allclose(new_state.state, ref_state)

    def test_quantum_teleportation(self):
        """Test quantum teleportation algorithm."""
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.0], [0, 1 / np.sqrt(2)]]))
        # generater a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = alice @ bell_state
        phi = CNOT(phi, [0, 1])
        phi = H(phi, [0])
        bits, result_state = Measure(phi, [0, 1])
        bit1, bit2 = bits
        if bit2:
            result_state = Pauli_X(result_state)
        if bit1:
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_permute(self):
        """Test quantum teleportation algorithm with permutation."""
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.0], [0, 1 / np.sqrt(2)]]))
        # generater a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = bell_state @ alice
        phi = CNOT(phi, [2, 1])
        phi = H(phi, [2])
        # assert False
        bits, result_state = Measure(phi, [2, 1])
        bit1, bit2 = bits
        if bit2:
            result_state = Pauli_X(result_state)
        if bit1:
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_bell_basis(self):
        """Test quantum teleportation algorithm with bell basis projection."""
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.0], [0, 1 / np.sqrt(2)]]))
        # generate a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = alice @ bell_state
        phi = BellBasis(phi, [0, 1])
        bits, result_state = Measure(phi, [0, 1])
        bit1, bit2 = bits
        if bit1:
            result_state = Pauli_X(result_state)
        if bit2:
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_bell_basis_permute(self):
        """Test quantum teleportation algorithm with bell basis and permutation."""
        bell_state = Circuit(np.array([[1 / np.sqrt(2), 0.0], [0, 1 / np.sqrt(2)]]))
        # generate a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = bell_state @ alice
        phi = BellBasis(phi, [1, 2])
        bits, result_state = Measure(phi, [1, 2])
        bit1, bit2 = bits
        if bit2:
            result_state = Pauli_Z(result_state)
        if bit1:
            result_state = Pauli_X(result_state)
        assert_allclose(result_state.state, alice.state)
