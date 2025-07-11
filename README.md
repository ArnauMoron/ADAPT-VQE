# ADAPT-VQE for the nuclear ground state problem

This is the code developed for my Bachelor's Thesis, which consists in the implementation and simulation of the **ADAPT-VQE** algorithm to compute the ground state energy of light atomic nuclei.

The algorithm has a classical framework that allows to select the operators and a quantum part that is in charge of building the circuits and measuring the energy. A new feature has been implemented to also compute the gradients on a quantum simulator. The code can be run in three different ways:

* **Classical**: Operator selection and parameter optimization are performed classically.
* **Quantum**: Both operator selection (via quantum gradient calculation) and parameter optimization are performed on a quantum simulator.
* **Mixed**: The operator ordering from a classical run is used to re-optimize the parameters in a quantum simulation environment.

## `Nucleus.py`

In this file, the `Nucleus` class is implemented. It is in charge of managing the data of a given nucleus. It takes the Hamiltonian, the basis states and the operators from data files. So the code can be easily used to simulate any nuclear shell by changing the number of qubits and the corresponding data files.

## `Ansatze.py`

This file defines the classes for the ansätze:

* `ADAPTAnsatz`: Implements the ADAPT ansatz for the classical version of the algorithm.
* `QuantumADAPTAnsatz`: Implements the ansatz for the **fully quantum version**, where both the energy and the gradients are computed with `qibo`.
* `ADAPT_mixed_Ansatz`: Implements the ansatz for the hybrid or mixed simulation, which uses a predefined operator ordering to optimize parameters on a quantum simulator.

## `Circuit.py`

This file contains the logic for **quantum circuit composition**. The `Circuits_Composser` class is the core of this section, handling:

* The transformation of fermionic operators into Pauli strings (Jordan-Wigner).
* The construction of the exponentials of Pauli operators (the ansatz) using the "staircase" algorithm.
* **Dynamic composition of circuits**: It assembles all the necessary circuits to measure each term of the Hamiltonian and the gradients of the operators.

## `Methods.py`

This file implements the main logic of the VQE algorithm. It contains the `ADAPTVQE`, `QuantumADAPTVQE` and `ADAPT_mixed_VQE` classes, which manage the optimization loop and interface with the corresponding ansatz. The new `QuantumADAPTVQE` class handles the entire quantum workflow, including calling the circuit composer for energy and gradient measurements.

## `ADAPT-VQE simulations.ipynb`

This is a Jupyter Notebook that shows the whole workflow of the project. It has examples on how to run the classical, full quantum and mixed simulations, as well as the analysis of the results. It also includes an analysis on how the shot noise affects the accuracy and standard deviation of the computed energy.

## `requirements.txt`

This file contains a list of all the Python dependencies needed to run the code.

## Acknowledgements

The foundational code for the classical ADAPT-VQE algorithm was developed by **Miquel Carrasco**. His original work can be found in his repository:
[miquel-carrasco/UCC_vs_ADAPT_p_shell](https://github.com/miquel-carrasco/UCC_vs_ADAPT_p_shell) Also, the Hamiltonian matrix elements and Many body basis files have been created with **Antonio Marquez Romero** code.

The adaptation of this classical framework and the **entire quantum implementation**—including circuit building with `qibo`, energy measurement protocols, and the hybrid simulation logic—were developed as part of my Bachelor's Thesis.

## Contact

For any questions, suggestions, or collaborations regarding this project, feel free to contact me at:

* **Arnau Morón**: [arnau.moron@gmail.com](mailto:arnau.moron@gmail.com)
