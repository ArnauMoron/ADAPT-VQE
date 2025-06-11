# ADAPT-VQE for Nuclear Ground State Simulations

This repository contains the code developed for my Bachelor's Thesis, focusing on the implementation and simulation of the **Variational Quantum Eigensolver (VQE)** algorithm using an adaptive ansatz, known as **ADAPT-VQE**. The goal is to calculate the ground-state energy of atomic nuclei, such as Beryllium-6 ($^6$Be).

The project combines a classical framework for operator selection with a quantum implementation for circuit construction and measurement, leveraging the `qibo` framework.

## Key Features

* **ADAPT-VQE Implementation**: Iteratively builds an efficient ansatz by only adding operators that contribute most significantly to lowering the energy.
* **Quantum Simulation**: Uses `qibo` and `qibojit` for quantum circuit simulations, supporting both exact state-vector calculations and shot-based measurements.
* **Nuclear Flexibility**: The code is designed to simulate any desired shell by adapting the number of qubits and the corresponding data files.
* **Shot Noise Analysis**: Includes analysis of how the number of shots in a simulation impacts the accuracy and standard deviation of the calculated energy.
* **Hybrid Approach**: Demonstrates a mixed workflow where operators are selected classically, and their parameters are then optimized within a quantum simulation environment.

## Project Structure

* `Nucleus.py`: A class that manages the data for a given nucleus (Hamiltonian, basis states, operators) from data files.
* `Ansatze.py`: Defines the classes for the ansätze, including `ADAPTAnsatz` (for the classical part) and `ADAPT_mixed_Ansatz` (for the quantum/mixed simulation).
* `Circuit.py`: Contains the logic for building quantum circuits. It handles the mapping from fermionic operators to Pauli strings (Jordan-Wigner transformation) and assembles the circuits using `qibo` via the `Circuits_Composser` class.
* `Methods.py`: Implements the main VQE logic. It contains the `ADAPTVQE` and `ADAPT_mixed_VQE` classes that manage the classical optimization loop and interface with the ansatz.
* `ADAPT-VQE simulations.ipynb`: A Jupyter Notebook demonstrating the complete workflow: from the initial classical ADAPT simulation to the quantum optimization with shots and results analysis.
* `requirements.txt`: A list of all necessary Python packages.

## Installation

To run this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/ArnauMoron/ADAPT-VQE

# Install the dependencies
pip install -r requirements.txt

```

## Usage and Workflow

The main workflow is demonstrated in the `ADAPT-VQE simulations.ipynb` notebook.

1. **Run Classical ADAPT**: The `ADAPT_minimization` function runs the ADAPT algorithm classically to determine the optimal sequence of operators for building the ansatz.

    ```python
    data, nucleo = ADAPT_minimization(nuc='Be6', ref_state=0, n_qubits=6, max_layers=3)
    ```

2. **Build Quantum Circuits**: Using the selected operators, `Circuits_Composser` constructs all the necessary quantum circuits to measure the different energy contributions.

3. **Measure Energy (Quantum Simulation)**: The `Qibo_measure_Energy` function calculates the total energy. It can operate in two modes:
    * **Exact**: By calculating the expectation value analytically from the final state vector.
    * **Shot-based**: By simulating actual measurements on the circuit to estimate probabilities and, from them, the energy.

    ```python
    # Exact measurement
    Et_exact = Qibo_measure_Energy(..., exact=True)

    # Measurement with 1000 shots
    Et_shots = Qibo_measure_Energy(..., exact=False, nshots=1000)
    ```

4. **Mixed Optimization**: The `ADAPT_mixed_minimization` function takes the operator ordering from the classical step and re-optimizes the operator parameters directly on the quantum simulator, taking shot noise into account.

    ```python
    data_mixed, nucleo = ADAPT_mixed_minimization(data=data, ..., exact=False, nshots=1000)
    ```

## Acknowledgements

The foundational code for the classical ADAPT-VQE algorithm was developed by **Miquel Carrasco**. His original work can be found in his repository:
*[miquel-carrasco/UCC_vs_ADAPT_p_shell](https://github.com/miquel-carrasco/UCC_vs_ADAPT_p_shell)

The adaptation of this classical framework and the **entire quantum implementation**—including circuit building with `qibo`, energy measurement protocols, and the hybrid simulation logic—were developed as part of this Bachelor's Thesis
