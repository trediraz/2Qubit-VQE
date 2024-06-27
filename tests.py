import unittest
from qiskit import *
import numpy as np
from math import pi
from main import get_target_hamiltonian, get_expectation, find_lower_bound
from decomposition import paulis, decompose


class TestHamiltonian(unittest.TestCase):
    #checks whether Hamiltonian is hermitian
    def test_hermitian(self):
        H = get_target_hamiltonian()
        self.assertTrue(np.array_equal(H, H.conj().T))


class TestDecomposition(unittest.TestCase):
    # checks whether sigmas are hermitian
    def test_paulis_H(self):
        keys = ["I", "X", "Y", "Z", "IX", "YZ", "ZZ", "XYZ"]
        result = 1
        for i in keys:
            if not np.array_equal(paulis(i), paulis(i).conj().T):
                result = 0
        self.assertTrue(result)

    # checks whether sigmas are involuntary
    def test_paulis_sqr(self):
        result = 1
        keys = ["I", "X", "Y", "Z", "IX", "YZ", "ZZ, XYZ"]
        for i in keys:
            s2 = np.matmul(paulis(i), paulis(i))
            if not np.array_equal(s2, paulis('I'*len(i))):
                result = 0
        self.assertTrue(result)

    def test_decomposition_single(self):
        result = 1
        keys = ["I", "X", "Y", "Z", "IX", "YZ", "ZZ", "XYZ"]
        for key in keys:
            operator = paulis(key)
            d = decompose(operator)
            if not d[key] == 1:
                result = 0
        self.assertTrue(result)

    #test for given linear combination
    def test_decomposition_superposition(self):
        d1 = {"IX": np.random.random(), "XY": np.random.random(), "YZ": np.random.random()}
        d2 = {"XYZ": np.random.random(), "YZZ": np.random.random(), "ZII": np.random.random()}
        vd = [d1, d2]
        result = True
        for d in vd:
            n = 2**len(list(d.keys())[0])
            m = np.zeros((n, n))
            for key in d.keys():
                m = np.add(m, d[key]*paulis(key))
            new_d = decompose(m)
            if not new_d == d:
                result = False
        self.assertTrue(result)


class TestSimulation(unittest.TestCase):
    tolerance = 0.1

    def test_expectation_trivial(self):
        n = 100
        count = 0
        circuit = QuantumCircuit(1)
        circuit.measure_all()
        for i in range(n):
            count += get_expectation(circuit, 100)
        self.assertTrue(abs(1-count/n) < self.tolerance)

    def test_expectation_half(self):
        n = 100
        count = 0
        circuit = QuantumCircuit(1)
        circuit.rx(pi/2, 0)
        circuit.measure_all()
        for i in range(n):
            count += get_expectation(circuit, 100)
        self.assertTrue(abs(count/n) < self.tolerance)

    def test_lower_bound_paulis(self):
        correct = True
        keys = ["IX", "YI", "YZ", "ZZ", "II", "XX"]
        for key in keys:
            pauli = paulis(key)
            lower_bound = find_lower_bound(pauli, 100)
            if key == "II":
                if not lower_bound == 1:
                    correct = False
            elif abs(1+lower_bound) > self.tolerance:
                correct = False
        self.assertTrue(correct)


if __name__ == '__main__':
    unittest.main()