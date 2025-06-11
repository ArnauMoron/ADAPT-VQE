from scipy.optimize import minimize
import numpy as np

from VQE.Nucleus import Nucleus
from VQE.Ansatze import ADAPTAnsatz, ADAPT_mixed_Ansatz
from VQE.Circuit import Circuits_Composser


class OptimizationConvergedException(Exception):
    pass

class VQE():
    """
    Parent class to define the Variational Quantum Eigensolvers (VQEs).

    Attributes:
        method (str): Optimization method.
        test_threshold (float): Threshold to stop the optimization.
        stop_at_threshold (bool): If True, the optimization stops when the threshold is reached.
        fcalls (list): List of function calls.
        energy (list): List of energies.
        rel_error (list): List of relative errors.
        success (bool): If True, the optimization was successful.
        tot_operations (list): List of total operations.
        options (dict): Optimization options.
    
    Methods:
        update_options: Update the optimization options.
    """

    def __init__(self,
                 test_threshold: float = 1e-4,
                 method: str = 'L-BFGS-B',
                 ftol: float = 1e-7,
                 gtol: float = 1e-3,
                 rhoend: float = 1e-5,
                 stop_at_threshold: bool = True) -> None:
        """
        Initialization of the VQE object.

        Args:
            test_threshold (float): Threshold to stop the optimization.
            method (str): Optimization method.
            ftol (float): Tolerance for the energy.
            gtol (float): Tolerance for the gradient.
            rhoend (float): Tolerance for the constraints.
            stop_at_threshold (bool): If True, the optimization stops when the threshold is reached.
        """
        self.method = method
        self.test_threshold = test_threshold
        self.stop_at_threshold = stop_at_threshold
        self.fcalls = []
        self.energy = []
        self.rel_error = []
        self.success = False 
        self.tot_operations = [0]
        try:
            self.method = method
        except method not in ['SLSQP', 'COBYLA','L-BFGS-B','BFGS']:
            print('Invalid optimization method, try: SLSQP, COBYLA, L-BFGS-B or BFGS')
            exit()
        self.options={}
        if self.method in ['SLSQP','L-BFGS-B']:
            self.options.setdefault('ftol',ftol)
        if self.method in ['L-BFGS-B','BFGS']:
            self.options.setdefault('gtol',gtol)
        if self.method == 'COBYLA':
            self.options.setdefault('tol',rhoend)

    def update_options(self,ftol,gtol,rhoend) -> None:
        """Update the optimization options"""

        if self.method in ['SLSQP','L-BFGS-B']:
            self.options['ftol']=ftol
        if self.method in ['L-BFGS-B','BFGS']:
            self.options['gtol']=gtol
        if self.method == 'COBYLA':
            self.options['rhoend']=rhoend

class ADAPTVQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        tot_operators (int): Total number of operators.
        layer_fcalls (list): List of function calls per layer.
        state_layers (list): List of states per layer.
        parameter_layers (list): List of parameters per layer.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self, 
                 ansatz: ADAPTAnsatz,
                 method: str = 'L-BFGS-B',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-4,
                 stop_at_threshold: bool = True,
                 max_layers: int = 100) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method, stop_at_threshold = stop_at_threshold)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.tot_operators = 0
        self.layer_fcalls = []
        self.state_layers = []
        self.parameter_layers = []
        self.max_layers = max_layers

        try:
            self.conv_criterion = conv_criterion
        except conv_criterion not in ['Repeated op', 'Gradient','None']:
            print('Invalid minimum criterion. Choose between "Repeated op", "Gradient" and "None"')
            exit()
    
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")

    

        
        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=self.fcalls[-1]*len(self.ansatz.added_operators)
        print('Initial Energy: ',E0)
        next_operator,next_gradient = self.ansatz.choose_operator()
        
        gradient_layers = []
        opt_grad_layers = []
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
        fcalls_layers = [self.fcalls[-1]]
        self.state_layers.append(self.ansatz.ansatz)
        while self.ansatz.minimum == False and len(self.ansatz.added_operators)<self.max_layers:
            self.ansatz.added_operators.append(next_operator)
            gradient_layers.append(next_gradient)
            self.parameter_layers.append([])
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.parameters.append(0.0)
            self.ansatz.count_fcalls = True
            try:
                result = minimize(self.ansatz.energy,
                                  self.parameters,
                                  method=self.method,
                                  callback=self.callback,
                                  options=self.options)
                self.parameters = list(result.x)

                print(self.parameters)

                nf = result.nfev
                
                if self.method!='COBYLA':
                    opt_grad= np.linalg.norm(result.jac)
                else:
                    opt_grad=0
                opt_grad_layers.append(opt_grad)

                self.ansatz.count_fcalls = False
                self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
                

                next_operator,next_gradient = self.ansatz.choose_operator()
                if self.conv_criterion == 'Repeated op' and next_operator == self.ansatz.added_operators[-1]:
                    self.ansatz.minimum = True
                elif self.conv_criterion == 'Gradient' and next_gradient < 1e-7:
                    self.ansatz.minimum = True
                else:
                    energy_layers.append(self.energy[-1])
                    rel_error_layers.append(self.rel_error[-1])
                    fcalls_layers.append(self.fcalls[-1])
                print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
                print('Energy: ',energy_layers[-1])
                print('Rel. Error: ',rel_error_layers[-1])
                print('New Operator: ',self.ansatz.added_operators[-1].ijkl,'    Theta:', self.parameters[-1])
            except OptimizationConvergedException:
                opt_grad_layers.append('Manually stopped')
            self.state_layers.append(self.ansatz.ansatz)
            
            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])      
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            if rel_error < self.test_threshold and self.stop_at_threshold:
                self.success = True

                self.ansatz.minimum = True
                break

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
        print('Energy: ',energy_layers[-1])
        print('Rel. Error: ',rel_error_layers[-1])
        print('New operator: ',self.ansatz.added_operators[-1].ijkl,'    Theta:', self.parameters[-1])
    
       


        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(f"Layer {i}: Operator {op.ijkl}, Theta = {self.parameters[i]}, Gradient = {gradient_layers[i]}")
            

        self.ansatz.ansatz = self.ansatz.build_ansatz(self.parameters)
        print('\n Ground state aproximation:')
        print([(idx, value) for idx, value in enumerate(self.ansatz.ansatz) if value != 0])

        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {self.rel_error[-1]}' )

        
        
        if self.conv_criterion == 'None' and self.ansatz.minimum == False:
            self.ansatz.minimum = True
            opt_grad_layers.append('Manually stopped')
        
        
        data={'parameters':self.parameters,
            'used_operators':[op for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
            
        return data
        

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        self.tot_operations.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters = params
            raise OptimizationConvergedException

def ADAPT_minimization(nucleus: str,
                       ref_state: int = 0,
                       opt_method: str = "L-BFGS-B",
                       threshold: float = 1e-6,
                       stop_at_threshold: bool = True,
                       max_layers: int = 20,
                       n_qubits: int = 6):

    ref_state_dict = {'ref_state':ref_state}
    nuc = Nucleus(nucleus, n_qubits=n_qubits)
    ref_state = np.eye(nuc.d_H)[ref_state]
    ansatz = ADAPTAnsatz(nucleus = nuc,
                       ref_state = ref_state)
    
    vqe = ADAPTVQE(ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   stop_at_threshold = stop_at_threshold,
                   max_layers = max_layers)

    
    ham = nuc.Ham_2_body_contributions()
    

    two_index=[]
    ham_pool=[]

    for op in ham:
        if len(set(op.ijkl))==2:
            two_index.append(op)
        else:
            ham_pool.append(op)


    data=vqe.run()
    
    one_body, monoparticular = nuc.Ham_1_body_contributions()
    diag_two_body_ops = {'two_index':two_index}
    ham_pool_ops = {'ham_pool':ham_pool}
    one_body = {'one_body':one_body}
    monoparticular = {'monoparticular':monoparticular}
    name = {'name':nucleus}
    
    
    data.update(monoparticular)    

    data.update(diag_two_body_ops)    

    data.update(ham_pool_ops)
    
    data.update(one_body)
    
    data.update(name)
    
    data.update(ref_state_dict)
    
    
    return data, nuc


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
        return data['used_operators'][i]

class ADAPT_mixed_VQE(VQE):
    """
    Child class to define the ADAPT VQE.

    Attributes:
        ansatz (ADAPTAnsatz): ADAPT Ansatz object.
        nucleus (Nucleus): Nucleus object.
        parameters (list): List of parameters.
        tot_operators (int): Total number of operators.
        layer_fcalls (list): List of function calls per layer.
        state_layers (list): List of states per layer.
        parameter_layers (list): List of parameters per layer.
        max_layers (int): Maximum number of layers.
    
    Methods:
        run: Runs the ADAPT VQE algorithm.
        callback: Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is
    """
    def __init__(self,
                 data: dict, 
                 ansatz: ADAPT_mixed_Ansatz,
                 method: str = 'COBYLA',
                 conv_criterion: str = 'Repeated op',
                 test_threshold: float = 1e-4,
                 stop_at_threshold: bool = True,
                 max_layers: int = 100,
                 exact:bool = True) -> None:
        
        super().__init__(test_threshold = test_threshold, method = method, stop_at_threshold = stop_at_threshold)
        self.ansatz = ansatz
        self.nucleus = ansatz.nucleus
        self.parameters = []
        self.tot_operators = 0
        self.layer_fcalls = []
        self.parameter_layers = []
        self.max_layers = max_layers
        self.data = data
        self.exact = exact
        operators_used=data['used_operators']
        self.operators_used=operators_used
        
        self.options={}

        
    def run(self) -> tuple:
        """
        Runs the ADAPT VQE algorithm and returns the data of the optimization.

        Returns:
            list: List of the selected operator per layer.
            list: List of energy gradient after optimization per layer.
            list: List of energies per layer.
            list: List of relative errors per layer.
            list: List of function calls per layer.        
        """
 
        print(" --------------------------------------------------------------------------")
        print("                            ADAPT for ", self.nucleus.name)                 
        print(" --------------------------------------------------------------------------\n")


        self.ansatz.fcalls = 0
        E0 = self.ansatz.energy(self.parameters)
        self.energy.append(E0)
        self.rel_error.append(abs((E0 - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=self.fcalls[-1]*len(self.ansatz.added_operators)
        print('Initial Energy: ',E0)
        next_operator = self.ansatz.choose_operator()
        
        
        opt_grad_layers = []
        energy_layers = [E0]
        rel_error_layers = [self.rel_error[-1]]
        fcalls_layers = [self.fcalls[-1]]
        
        
        while self.ansatz.capas<len(self.data['used_operators']):
            self.ansatz.capas += 1
            self.ansatz.added_operators.append(next_operator)
            
            self.parameter_layers.append([])
            self.layer_fcalls.append(self.ansatz.fcalls)
            self.parameters.append(0.0)
            self.ansatz.count_fcalls = True
            scaling=[0.3 for _ in range(len(self.parameters)-1)]
            scaling.append(1)
            try:
            
                result = minimize(self.ansatz.energy,
                                    self.parameters,
                                    method=self.method,
                                    callback=self.callback,
                                    options=self.options,
                                    bounds=[(-np.pi, np.pi) for _ in range(len(self.parameters))])
                self.parameters = list(result.x)

                print(self.parameters)
            except Exception as e:
                print(e)
                
                print('final parameters: ', self.parameters)
                break
              


            for a in range(len(self.parameters)):
                self.parameter_layers[a].append(self.parameters[a])      
            rel_error = abs((self.energy[-1] - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0])
            if rel_error < self.test_threshold and self.stop_at_threshold:
                self.success = True

                self.ansatz.minimum = True
                break

        energy_layers.append(self.energy[-1])
        rel_error_layers.append(self.rel_error[-1])
        fcalls_layers.append(self.fcalls[-1])
        print(f"\n------------ LAYER {len(energy_layers)-1} ------------")
        print('Energy: ',energy_layers[-1])
        print('Rel. Error: ',rel_error_layers[-1])
        print('New operator: ',self.ansatz.added_operators[-1].ijkl,'    Theta:', self.parameters[-1])
    
       


        print("\nOperators used for each layer:")
        for i, op in enumerate(self.ansatz.added_operators):
            print(f"Layer {i}: Operator {op.ijkl}, Theta = {self.parameters[i]}")

        
        print('\n Ground state aproximation:')
        print([(idx, value) for idx, value in enumerate(self.ansatz.ansatz) if value != 0])

        print(f'\n Final energy result: {energy_layers[-1]}\t', f'Final relative error is {self.rel_error[-1]}' )

        
    
        
        
        data={'parameters':self.parameters,
            'used_operators':[op.ijkl for op in self.ansatz.added_operators],
            'operator_pool':self.ansatz.operator_pool,
            'Energy': energy_layers[-1]}
            
        return data
        

    def callback(self, params: list) -> None:
        """
        Callback function to store the energy and parameters at each iteration and stop the optimization if the threshold is reached.
        """
        self.ansatz.count_fcalls = False
        E = self.ansatz.energy(params)
        self.ansatz.count_fcalls = True
        self.energy.append(E)
        self.rel_error.append(abs((E - self.ansatz.nucleus.eig_val[0])/self.ansatz.nucleus.eig_val[0]))
        self.fcalls.append(self.ansatz.fcalls)
        self.tot_operators+=(self.fcalls[-1]-self.fcalls[-2])*len(self.ansatz.added_operators)
        self.tot_operations.append(self.tot_operators)
        if self.rel_error[-1] < self.test_threshold and self.stop_at_threshold:
            self.success = True
            self.ansatz.minimum = True
            self.parameters = params
            raise OptimizationConvergedException

def ADAPT_mixed_minimization(data: dict,
                            nucleus: str,
                       ref_state: int = 0,
                       opt_method: str = "L-BFGS-B",
                       threshold: float = 1e-6,
                       stop_at_threshold: bool = True,
                       max_layers: int = 20,
                       n_qubits: int = 6,
                       exact:bool =True,
                       nshots:int = 1000):

    
    nuc = Nucleus(nucleus, n_qubits=n_qubits)
    ref_state = np.eye(nuc.d_H)[ref_state]
    
    ansatz = ADAPT_mixed_Ansatz(data= data,
                                nucleus = nuc,
                                ref_state = ref_state,
                                exact=exact,
                                nshots=nshots)
    
    vqe = ADAPT_mixed_VQE(data=data,
                          ansatz = ansatz,
                   method = opt_method,
                   test_threshold = threshold,
                   stop_at_threshold = stop_at_threshold,
                   max_layers = max_layers,
                   exact=exact)

    
    ham = nuc.Ham_2_body_contributions()
    

    two_index=[]
    ham_pool=[]

    for op in ham:
        if len(set(op.ijkl))==2:
            two_index.append(op)
        else:
            ham_pool.append(op)


    data=vqe.run()
    
    one_body, monoparticular = nuc.Ham_1_body_contributions()
    diag_two_body_ops = {'two_index':two_index}
    ham_pool_ops = {'ham_pool':ham_pool}
    one_body = {'one_body':one_body}
    monoparticular = {'monoparticular':monoparticular}
    name = {'name':nucleus}
    ref_state_dict = {'ref_state':ref_state}
    
    
    data.update(monoparticular)    

    data.update(diag_two_body_ops)    

    data.update(ham_pool_ops)
    
    data.update(one_body)
    
    data.update(name)
    
    data.update(ref_state_dict)
    
    
    return data, nuc
