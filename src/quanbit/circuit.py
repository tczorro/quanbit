import numpy as np
import string
import warnings


class Circuit:
    def __init__(self, coeff):
        # check input type and coeff
        n_cubit = len(coeff.shape)
        for i in range(n_cubit):
            if coeff.shape[i] != 2:
                raise ValueError("Input coeff is not of proper shape.")
        # check input coeff is well normalized
        tot = np.sum(np.abs(coeff) ** 2)
        if abs(tot - 1) > 1e-5:
            warnings.warn("Norm excess 1, normalization will be applied.")
        self._state = np.array(coeff, dtype=complex) / np.sqrt(tot)
        self._nqubit = n_cubit

    @property
    def prob(self):
        return np.abs(self.state) ** 2

    def measure(self, indices=[0]):
        # get the indices to sum over prob exclude the selected indices
        n_bit = len(indices)
        sum_indices = tuple(set(np.arange(self.n_cubit)).difference(set(indices)))
        # flatten probability into 1d array
        prob = np.sum(self.prob, axis=sum_indices).flatten()
        # get the measure result in the index form and converted to binary
        random_result = np.random.choice(2 ** n_bit, p=prob)
        # convert index into binary result, the same as cubit value
        bin_result = "{:0{}b}".format(random_result, n_bit)
        result = tuple(map(int, bin_result))
        # obtain leftover state for unmeasured cubit(s)
        left_state = np.moveaxis(self._state, indices, list(range(n_bit)))[result]
        # renormalize the probablity distribution
        renorm_state = left_state / np.linalg.norm(left_state)
        return result, Circuit(renorm_state)

    def apply_operator(self, op, indices=[0]):
        n_bit = op.n_target
        if n_bit != len(indices):
            raise ValueError(
                f"# of cubit does not match with the operator.\n"
                f"n_qubit need: {self._nqubit}, got: {len(index)}"
            )
        result = np.tensordot(
            op.matrix, self.state, axes=[np.arange(-n_bit, 0), indices]
        )
        return Circuit(result)


    @property
    def n_cubit(self):
        return self._nqubit

    @property
    def state(self):
        return self._state

    def __matmul__(self, other):
        return Circuit(coeff=np.tensordot(self.state, other.state, axes=0))


def cubit(phi, theta, alpha=0.0):
    return Circuit(
        np.exp(alpha * 1j)
        * np.array(
            [np.cos(theta / 2), np.exp(phi * 1j) * np.sin(theta / 2)], dtype=complex,
        )
    )


def bell_state(self, style=0):
    ...
