import numpy as np
import qiskit.circuit
import decomposition
from math import pi
from matplotlib import pyplot as plt
from qiskit import *
from qiskit_aer import AerSimulator
from qiskit.circuit.library import *
from scipy import optimize


def get_target_hamiltonian():
    return np.array([[2.18, 1.36-1.41j, 0.82-0.46j, 0.1-0.49j],
                    [1.36+1.41j, -0.7, 1.74-0.07j, -0.6+0.36j],
                    [0.82+0.46j, 1.74+0.07j, 0.54, 0.56-0.17j],
                    [0.1+0.49j, -0.6-0.36j, 0.56+0.17j, 0.58]])


def get_reference_hamiltonian():
    return np.array([[0.4, 0, 0, 2],[0, 0, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0.4]])


def naive_ansatz():
    angle = 1854 - 2 * pi * int(1854.0 / 2 / pi)  # Student ID 185400 / 100 and modulo 2pi
    #angle = pi/2
    angles = qiskit.circuit.ParameterVector("angle", 2)
    qc = QuantumCircuit(2, 1)
    qc.ry(angle, 0)
    qc.rx(-angle, 1)
    qc.cx(0, 1)
    qc.rz(angles[0], 1)
    qc.rx(angles[1], 1)
    qc.cx(0, 1)
    qc.ry(-angle, 0)
    qc.rx(angle, 1)
    return qc


def ansatz():
    angles = qiskit.circuit.ParameterVector("angle", 7)
    qc = QuantumCircuit(2, 1)
    qc.ry(angles[0], 0)
    qc.cx(0, 1)
    qc.rz(angles[2], 0)
    qc.rx(angles[3], 0)
    qc.rz(angles[5], 1)
    qc.ry(angles[4], 1)
    qc.cx(1, 0)
    qc.rx(angles[6], 1)
    qc.rz(angles[1], 0)
    qc.barrier()
    return qc


def append_pauli_measurements(circuit, basis):
    if "I" in basis:
        if basis[0] == "I" and not basis[1] == "I":
            circuit.append(SwapGate(), [0, 1])
        if 'X' in basis:
            circuit.append(HGate(), [0])
        if 'Y' in basis:
            circuit.append(SGate().inverse(), [0])
            circuit.append(HGate(), [0])
    else:
        if basis[0] == 'X':
            circuit.append(HGate(), [0])
        if basis[0] == "Y":
            circuit.append(SGate().inverse(), [0])
            circuit.append(HGate(), [0])
        if basis[1] == 'X':
            circuit.append(HGate(), [1])
        if basis[1] == "Y":
            circuit.append(SGate().inverse(), [1])
            circuit.append(HGate(), [1])
        circuit.append(CXGate(), [1, 0])
    circuit.measure([0], [0])
    return circuit


def get_expectation(circuit, shots):
    simulator = AerSimulator()
    counts = simulator.run(circuit, shots=shots).result().get_counts()
    result = 0
    if '0' in counts.keys():
        result += counts['0']
    if '1' in counts.keys():
        result -= counts['1']
    return result/shots


def vqe_ground(hamiltonian, angles, n, is_naive_ansatz):
    result = 0
    for key in hamiltonian.keys():
        if key == "I"*len(key):
            result += hamiltonian[key]
        else:
            if is_naive_ansatz:
                circuit = append_pauli_measurements(naive_ansatz().assign_parameters(angles), key)
            else:
                circuit = append_pauli_measurements(ansatz().assign_parameters(angles), key)
            result += hamiltonian[key]*get_expectation(circuit, n)
    return result


def find_lower_bound(H, shots, is_naive_ansatz):
    d = decomposition.decompose(H)
    result = []
    for i in range(10):
        n = 2 if is_naive_ansatz else 7
        x0 = np.random.random(n)
        result.append(optimize.minimize(lambda angles: vqe_ground(d, angles, shots, is_naive_ansatz), x0=x0, method='cobyla').fun)
    return np.min(result)


def print_as_paulis(M):
    d = decomposition.decompose(M)
    keys = ["I", 'X', 'Y', 'Z']
    print("   I:     X:     Y:     Z:")
    for k1 in keys:
        string = k1 + ": "
        for k2 in keys:
            key = "" + k1+k2
            if key in d.keys():
                string += f"{d[key]:.4f} "
            else:
                string += f"{0:.4f} "
        print(string)


def compere_eigen_values(H, is_naive_ansatz):
    EQ = find_lower_bound(H, 1000, is_naive_ansatz)
    EA = np.real(np.sort(np.linalg.eigvals(H))[0])
    dE = (EA-EQ)/EA
    print("VQE estimation: {:.3f},\nClassical result: {:.3f},\nRelative error: {:.3f}".format(EQ, EA, dE))


if __name__ == '__main__':
    # compere_eigen_values(get_target_hamiltonian(), is_naive_ansatz=1)
    compere_eigen_values(get_reference_hamiltonian(), is_naive_ansatz=0) #use 0 or 1 to change ansatz
    plt.show()
