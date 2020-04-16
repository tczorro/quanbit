# Copyright (c) 2019, Derrick Yang
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase
from quanbit.circuit import Circuit, qubit, bell_state
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
        bell_st = bell_state(0)
        # generater a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = alice @ bell_st
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
        bell_st = bell_state(0)
        # generater a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = bell_st @ alice
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
        bell_st = bell_state(0)
        # generate a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = alice @ bell_st
        phi = BellBasis(phi, [0, 1])
        bits, result_state = Measure(phi, [0, 1])
        if bits == (0, 1):
            result_state = Pauli_X(result_state)
        elif bits == (1, 0):
            result_state = Pauli_Z(Pauli_X((result_state)))
        elif bits == (1, 1):
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_bell_revert(self):
        """Test quantum teleportation algorithm with bell basis projection."""
        bell_st = bell_state(0)
        # generate a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = alice @ bell_st
        phi = BellBasis(phi, [1, 0])
        bits, result_state = Measure(phi, [1, 0])
        if bits == (0, 1):
            result_state = Pauli_X(result_state)
        elif bits == (1, 0):
            result_state = Pauli_X(Pauli_Z((result_state)))
        elif bits == (1, 1):
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_quantum_teleportation_bell_basis_permute(self):
        """Test quantum teleportation algorithm with bell basis and permutation."""
        bell_st = bell_state(0)
        # generate a random state
        a = np.random.rand(3)
        alice = qubit(*a)
        phi = bell_st @ alice
        phi = BellBasis(phi, [1, 2])
        bits, result_state = Measure(phi, [1, 2])
        if bits == (0, 1):
            result_state = Pauli_X(result_state)
        elif bits == (1, 0):
            result_state = Pauli_X(Pauli_Z((result_state)))
        elif bits == (1, 1):
            result_state = Pauli_Z(result_state)
        assert_allclose(result_state.state, alice.state)

    def test_bell_basis(self):
        for i in range(3):
            coeff = np.zeros(4)
            coeff[i] = 1
            coeff = coeff.reshape(2,2)
        basic_state = Circuit(coeff)
        new_state = BellBasis(basic_state, [0, 1])

    def test_bell_state(self):
        for i in range(4):
            bell_st = bell_state(i)
            proj_st = BellBasis(bell_st, [0, 1])
            j = i // 2
            k = i % 2
            assert proj_st.state[j][k] == 1
            back_st = BellBasis(proj_st, [0, 1])
            np.allclose(back_st.state, bell_st.state)
