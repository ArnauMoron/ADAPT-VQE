import numpy as np
from VQE.Nucleus import Nucleus
from  scipy.sparse.linalg import expm_multiply, expm
from VQE.Circuit import Circuits_Composser, Qibo_measure_Energy


class Ansatz():
    """
    Parent class to define ansätze for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        all_operators (list): List of all the avaliable two-body excitation operators for a given nucleus.
        operator_pool (list): List of operators used in the ansatz.
        fcalls (int): Number of function calls of a VQE procedure.
        count_fcalls (bool): If True, the number of function calls during a VQE procedure is counted.
        ansatz (np.ndarray): Ansatz state.
    
    Methods:
        reduce_operators: Returns the list of operators excluding the repeated excitations.
        only_acting_operators: Returns the list of operators, only including the ones that have a non-zero action on the ref. state.
    """
    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the Ansatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool. Available formats ['All', 'Reduced', 'Only acting', 'Custom'].
            operators_list (list): List of operators to be used in the ansatz, in case the pool format is 'Custom'.
        """
        self.nucleus = nucleus
        self.ref_state = ref_state
       
        self.all_operators = self.nucleus.operators
        
        self.operator_pool = nucleus.operators
        
        self.fcalls = 0
        self.count_fcalls = False
        self.ansatz = self.ref_state
  
class ADAPTAnsatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        self.E0 = self.energy([])


    def build_ansatz(self, parameters: list) -> np.ndarray:
        """
        Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.

        Returns:
            np.ndarray: Ansatz state.        
        """
        ansatz=self.ref_state

        for i,op in enumerate(self.added_operators):
            ansatz = expm_multiply(parameters[i]*op.matrix, ansatz, traceA = 0.0)
        return ansatz


    def energy(self, parameters: list) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        if len(parameters) != 0:
            if self.count_fcalls == True:
                self.fcalls += 1
            new_ansatz = self.build_ansatz(parameters)
            E = new_ansatz.conj().T.dot(self.nucleus.H.dot(new_ansatz))
            return E
        else:
            E = self.ansatz.conj().T.dot(self.nucleus.H.dot(self.ansatz))
            return E


    def choose_operator(self) -> tuple:
        """
        Returns the next operator and its gradient, after an ADAPT iteration.

        Returns:
            TwoBodyExcitationOperator: Next operator.
            float: Gradient of the next operator.
        """

        gradients = []
        sigma = self.nucleus.H.dot(self.ansatz)
        gradients = [abs(2*(sigma.conj().T.dot(op.matrix.dot(self.ansatz))).real) for op in self.operator_pool]
        max_gradient = max(gradients)
        max_operator = self.operator_pool[gradients.index(max_gradient)]
        
        return max_operator,max_gradient
    
class ADAPT_mixed_Ansatz(Ansatz):
    """
    Child Ansatz class to define the ADAPT ansatz for VQE.

    Attributes:
        nucleus (Nucleus): Nucleus object.
        ref_state (np.ndarray): Reference state of the ansatz.
        pool_format (str): Format of the operator pool.
        operators_list (list): List of operators to be used in the ansatz.
        added_operators (list): List of operators added to the ansatz.
        minimum (bool): If True, the ansatz has reached the minimum energy.
        E0 (float): Energy of the ansatz without any excitation operators.

    Methods:
        build_ansatz: Returns the state of the ansatz on a given VQE iteratioin, after building it with the given paramters and the operators in the pool.
        energy: Returns the energy of the ansatz on a given VQE iteration.
        choose_operator: Returns the next operator and its gradient, after an ADAPT iteration.
    """

    def __init__(self,
                 nucleus: Nucleus,
                 ref_state: np.ndarray, 
                 data: dict,
                 exact:bool=True,
                 nshots:int = 1000) -> None:
        """
        Initialization of the ADAPTAnsatz object.

        Args:
            nucleus (Nucleus): Nucleus object.
            ref_state (np.ndarray): Reference state of the ansatz.
            pool_format (str): Format of the operator pool.
            operators_list (list): List of operators to be used in the ansatz (optional).
        """
        super().__init__(nucleus, ref_state)
        self.added_operators = []
        self.minimum = False
        self.data=data
        self.exact=exact
        self.nshots=nshots
        self.E0 = self.energy([])
        self.capas=0


    def energy(self, parameters, **kwargs) -> float:
        """
        Returns the energy of the ansatz on a given VQE iteration.

        Args:
            parameters (list): Values of the parameters of a given VQE iteration.
        
        Returns:
            float: Energy of the ansatz.
        """
        data = self.data
        monoparticular_energies = data['monoparticular']
        two_index = data['two_index']
        used_operators = data['used_operators'][0:len(parameters)]
        operator_pool = data['ham_pool']
        name = data['name']
        ref_state = data['ref_state']
        n_qubits = 12
        print(ref_state)
        
        composer = Circuits_Composser(operator_pool=operator_pool, operators_used=used_operators, n_qubits=n_qubits, ref_state=ref_state, name=name, parameters=parameters)
        Qibo_circuits=composer.Qibo_all_circuits()

        Et = Qibo_measure_Energy(monoparticular_energies, two_index, Qibo_circuits, exact=self.exact, nshots=self.nshots)
        
        return Et

    def choose_operator(self):
        i=self.capas
        return self.data['used_operators'][i]