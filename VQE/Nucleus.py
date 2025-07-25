import numpy as np
from numpy import linalg as la
import os

class TwoBodyExcitationOperator():
    """
    Class to define an antihermitian operator corresponding to a two-body excitation. The data of the operator
    is taken from already processed files in the data folder for each nucleus.
    
    Attributes:

    """

    def __init__(self, 
                 label: int,
                 H2b: float, 
                 ijkl: list, 
                 matrix: np.ndarray, 
                 commutator: np.ndarray ) -> None:
        """
        Initializes the operator instance, according to its pre-generated data.

        Args:
            label (int): Label of the operator (as it appears on the data files).
            H2b (float): Value of the amplitude of the operator in the hamiltonian.
            ijkl (list): Indices of the two body excitation: (a'_i a'_j a_k a_l).
            matrix (np.ndarray): Matrix representation of the operator.
            commutator (np.ndarray): Commutator of the operator with the Hamiltonian ([H, T]).
        """
        self.label = label
        self.H2b = H2b
        self.ijkl = ijkl
        self.matrix = matrix
        self.commutator = commutator
        
        
class OneBodyExcitationOperator():
    """
    Class to define an antihermitian operator corresponding to one-body excitation. The data of the operator
    is taken from already processed files in the data folder for each nucleus.
    
    Attributes:

    """

    def __init__(self, 
                 eps: float, 
                 i: int, 
                 matrix: np.ndarray) -> None:
        """
        Initializes the operator instance, according to its pre-generated data.

        Args:
            eps (float): Value of the amplitude of the operator in the hamiltonian.
            i (list): Indices of the two body excitation: (a'_i a_i).
            matrix (np.ndarray): Matrix representation of the operator.
            
        """
        
        self.eps = eps
        self.i = i
        self.matrix = matrix
        

        

class Nucleus():
    """
    Class to define a nucleus with its Hamiltonian, eigenvalues and eigenvectors, 
    angular momentum and other properties.
        
    Attributes:
        name (str): Name of the nucleus.
        data_folder (str): Path to the folder with the data of the nucleus.
        states (list): List of the basis states of the nucleus.
        H (csc_matrix): Hamiltonian matrix of the nucleus.
        d_H (int): Dimension of the Hamiltonian matrix.
        eig_val (np.ndarray): Eigenvalues of the Hamiltonian.
        eig_vec (np.ndarray): Eigenvectors of the Hamiltonian.
        operators (list): List of all antihermitian operators corresponding to two-body excitations.

    Methods:
        hamiltonian_matrix: Returns the hamiltonian matrix of the nucleus.
        states_list: Returns the list of states of the many-body basis according to the indices of the single-particle states.
        operators_list: Returns the list of ALL antihermitian operators corresponding to two-body excitations.
        excitation_numbers: Returns the new state and parity of the excitation after the action of a two-body excitation operator

    """

    def __init__(self, nuc_name: str,
                 n_qubits:int =6) -> None:
        """Initializes the nucleus with its name, angular momentum and magnetic quantum number.
        
        Args:
            nuc_name (str): Name of the nucleus.
        """
        self.n_qubits=n_qubits
        self.name = nuc_name
        self.data_folder = os.path.join(f'nuclei/{self.name}_data')
        self.states = self.states_list()
        self.H = self.hamiltonian_matrix()
        self.d_H = self.H.shape[0]
        self.eig_val, self.eig_vec = la.eigh(self.H)
        self.operators = self.operators_list()
    

    def hamiltonian_matrix(self) -> np.ndarray:
        """
        Returns the hamiltonian matrix of the nucleus.
        
        Returns:
            np.ndarray: Hamiltonian matrix.
        """
        file_path = os.path.join(self.data_folder, f'{self.name}.dat')
        H = np.zeros((self.d_H, self.d_H))
        H_data = np.loadtxt(file_path,delimiter=' ', dtype=float)
        for line in H_data:
            H[int(line[0]), int(line[1])] = line[2]
        return H
    
    def states_list(self) -> list:
        """
        Returns the list of states of the many-body basis according to the indices of the single-particle states.

        Returns:
            list: List of the basis states.
        """
        states = []
        mb_path = os.path.join(self.data_folder, f'mb_basis_2.dat')
        file = open(mb_path, 'r')
        self.d_H = int(file.readline().strip())
        mb_data = np.loadtxt(mb_path, dtype=str, delimiter=' ',skiprows=1)
        for m in mb_data:
            sp_labels = []
            for i in range(1, len(m)):
                label = int(m[i].replace(',','').replace('(','').replace(')',''))
                sp_labels.append(int(label))
            states.append(tuple(sp_labels))
        return states

    
    def operators_list(self) -> list:
        """
        Returns the list of ALL antihermitian operators corresponding to two-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[TwoBodyExcitationOperators]: List of antihermitian operators, as TwoBodyExcitationOperator instances.        
        """
        operators = []
        H2b_path = os.path.join(self.data_folder, f'H2b.dat')
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        label = 1
        for h in H2b_data:
            indices = [int(h[1]), int(h[2]), int(h[3]), int(h[4])]
            i, j, k , l = indices
            H2b = float(h[0])
            if i < j and k < l and (i,j) <= (k,l):
                operator_matrix = np.zeros((self.d_H, self.d_H))
                for state in self.states:
                    new_state, parity = self.excitation_numbers(state, [i,j,l,k])
                    if new_state in self.states:
                        column = self.states.index(state)
                        row = self.states.index(new_state)
                        this_excitation = np.zeros((self.d_H, self.d_H))
                        this_excitation[row, column] = parity
                        operator_matrix += this_excitation
                        operator_matrix += -this_excitation.T
                
                if np.allclose(operator_matrix, np.zeros((self.d_H, self.d_H))) == False:
                    commutator = self.H.dot(operator_matrix) - operator_matrix.dot(self.H)
                    operators.append(TwoBodyExcitationOperator(label, H2b, indices, operator_matrix, commutator))
                    label += 1
        return operators


    def Ham_2_body_contributions(self) -> list:
        """
        Returns the list of ALL hermitian observables corresponding to two-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[TwoBodyExcitationOperators]: List of antihermitian operators, as TwoBodyExcitationOperator instances.        
        """
        operators = []
        H2b_path = os.path.join(self.data_folder, f'H2b.dat')
        H2b_data = np.loadtxt(H2b_path, dtype=str)
        label = 1
        for h in H2b_data:
            indices = [int(h[1]), int(h[2]), int(h[3]), int(h[4])]
            i, j, k, l = indices
            H2b = float(h[0])
            if max(indices) < self.n_qubits:
                if i < j and k < l and (i,j) <= (k,l):
                    operator_matrix = np.zeros((self.d_H, self.d_H))
                    for state in self.states:
                        new_state, parity = self.excitation_numbers(state, indices)
                        if new_state in self.states:
                            column = self.states.index(state)
                            row = self.states.index(new_state)
                            this_excitation = np.zeros((self.d_H, self.d_H))
                            this_excitation[row, column] = parity
                            operator_matrix += this_excitation
                            if row!=column:
                                operator_matrix += this_excitation.T
                    commutator = 0
                    
                    operators.append(TwoBodyExcitationOperator(label, H2b, indices, operator_matrix, commutator))
                    label += 1
        return operators
    
    def Ham_1_body_contributions(self) -> list:
        """
        Returns the list of ALL hermitian observables corresponding to one-body excitations.
        The indices of the avaliable operators are taken from the data files of the nucleus, since it only includes
        those operators that respect the selection rules.

        Returns:
            list[OneBodyExcitationOperators]: List of hermitian operators, as OneBodyExcitationOperator instances.        
        """
        
        operators = []
        sp_path = os.path.join(self.data_folder, f'sp.dat')
        sp_data = np.loadtxt(sp_path, dtype=str, skiprows=1)
        
        states = self.states
        monoparticular_energies = {}
        
        for row in sp_data:
            sp = row[0]
            if int(sp) < self.n_qubits:
                eps = float(row[-1])
                monoparticular_energies[sp] = eps
            
        
        for state in range(len(states)):
            state_energy=0
            matrix=np.zeros((self.d_H, self.d_H))
            matrix[state][state]=1
            
            for particle in states[state]:
                E_i=monoparticular_energies[str(particle)]
                state_energy+=E_i
            
            operator=OneBodyExcitationOperator(eps=state_energy, i=state, matrix=matrix)
            operators.append(operator)
            
        return operators, monoparticular_energies
                
    
    def excitation_numbers(self, state: tuple, indices: list) -> tuple:
        """
        Returns the new state and parity of the excitation after the action of a two-body excitation operator
        on a state of the basis.

        Args:
            state (tuple): Indices of the single-particle states of a given state of the many-body basis.
            indices (list): Indices of the two-body excitation: (a_i* a_j* a_k a_l).
        Returns:
            tuple: New state after the excitation.  
            int: Parity of the excitation.      
        """
        parity = 1
        if indices[2] in state and indices[3] in state:
            new_state = list(state)
        #Annihilation
            for i in [indices[2], indices[3]]:
                parity *= (-1)**(new_state.index(i))
                new_state.remove(i)
        #Creation
            for i in [indices[1], indices[0]]:
                new_state.append(i)
                new_state.sort()
                parity *= (-1)**(new_state.index(i))
            
            return tuple(new_state), parity
        else:
            return tuple(), 0