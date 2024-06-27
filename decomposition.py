import numpy as np


def paulis(pauli_string):
    pauli_operator = np.array([1])
    for pauli in pauli_string:
        if pauli == "I":
            sigma = np.array([[1, 0], [0, 1]])
        elif pauli == "X":
            sigma = np.array([[0, 1], [1, 0]])
        elif pauli == "Y":
            sigma = np.array([[0, -1j], [1j, 0]])
        elif pauli == "Z":
            sigma = np.array([[1, 0], [0, -1]])
        else:
            sigma = np.array([[1, 0], [0, 1]])
        pauli_operator = np.kron(pauli_operator, sigma)
    return pauli_operator


def sting_kron(v,w):
    result = []
    for x in v:
        for y in w:
            result.append(x+y)
    return result


def decompose(matrix):
    n = matrix.shape[0]
    d1_keys = ["I", "X", "Y", "Z"]
    keys = d1_keys
    for i in range(int(np.log2(n))-1):
        keys = sting_kron(keys, d1_keys)
    decomposition = {}
    for key in keys:
        operator = paulis(key)
        temp_m = np.matmul(matrix, operator)
        val = np.real(np.trace(temp_m)/n)
        if not val == 0:
            decomposition[key] = val

    return decomposition
