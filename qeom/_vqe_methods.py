
import time, copy
import numpy as np
import scipy
import openfermion, cirq, qeom
# from qeom import SSQUARE
# from ._qchem_ansatz import SSQUARE, UCCSD

class Adapt_VQE():

    """
    Adapt-VQE ALGORITHM:
    Grimsley, H.R., Economou, S.E., Barnes, E. et al.
    An adaptive variational algorithm for exact molecular simulations on a quantum computer.
    Nat Commun 10, 3007 (2019). https://doi.org/10.1038/s41467-019-10988-2
    Code reference: https://github.com/asthanaa/adapt-vqe
    """
    def __init__(self, mol=None):
        self._thresh = 1.0e-12
        self._n_electrons = mol.n_electrons
        self._n_qubits = mol.n_qubits
        self._n_orbitals = mol.n_orbitals
        self._nuclear_repulsion = mol.nuclear_repulsion
        self._hf_energy = mol.hf_energy
        self._one_body_integrals = mol.one_body_integrals
        self._two_body_integrals = mol.two_body_integrals

    def run(self,n_tol=1.0e-3,max_iter=50):
        # Prepare operators
        hamiltonian = openfermion.InteractionOperator(self.nuclear_repulsion,\
                                                      self.one_body_integrals,\
                                                      0.25*self.two_body_integrals)
        self._sparse_ham = openfermion.get_sparse_operator(openfermion.jordan_wigner(hamiltonian),\
                                                           n_qubits=self.n_qubits)
        self._hf_ref = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(range(self.n_electrons)),\
                                                                                  self.n_qubits)).transpose()
        self._spin_operator = qeom.SSQUARE(self.n_electrons,self.n_qubits).generate()
        self.singlet_GSD()

        # Run ground state
        curr_state = copy.deepcopy(self.hf_ref)
        curr_energy = self.hf_energy
        operators = []
        parameters = []
        conv = False
        print("*****************************************************")
        print("*              Start ADAPT-VQE algorithm            *")
        print("* Grimsley, H.R., Economou, S.E., Barnes, E. et al. *")
        print("*              Nat Commun 10, 3007 (2019).          *")
        print("*     https://doi.org/10.1038/s41467-019-10988-2    *")
        print("*       https://github.com/asthanaa/adapt-vqe       *")
        print("*****************************************************")
        print(" ------------------------------------------------")
        print("      Iter   Energy (a.u.)  Gnorm      <S^2>     ")
        print(" ------------------------------------------------")
        vqe_start = time.time()
        for iter in range(max_iter):
            norm_grad,max_indx = self.grow_ansatz(curr_state)

            if norm_grad < n_tol:
                conv = True
            if conv:
                break

            parameters.insert(0,0)
            operators.insert(0,self.operator_pool[max_indx])
            result = scipy.optimize.minimize(self.vqe_energy,parameters,jac=self.trotter_gradient,\
                                            options={'gtol':1.0e-5,'disp':False},\
                                            args=(operators),method='BFGS')
            curr_energy = result.fun
            parameters = result.x.tolist()
            curr_state = self.prepare_state(parameters,operators)
            s2 = np.dot(curr_state.transpose().conj(), np.dot(self.spin_operator, curr_state))[0,0].real
            print("    {:3d}     {:.8f}     {:.3f}     {:>5.2f}".format(iter+1, curr_energy, norm_grad, s2))

        if conv:
            print("    {:3d}     {:.8f}*    {:.3f}     {:>5.2f}".format(iter+1, curr_energy, norm_grad, s2))
            self._ground_energy = curr_energy
            self._ground_params = parameters
            self._ground_operators = operators
            print(' ------------------------------------------------')
            print(" SCF energy                   = {:.8f}".format(self.hf_energy))
            print(" ADAPT-VQE correlation energy = {:.8f}".format(curr_energy-self.hf_energy))
            print(" ADAPT-VQE energy             = {:.8f}".format(curr_energy))
            print("\n ADAPT-VQE calculation: {:.2f} s".format(time.time()-vqe_start))
            print(' ================================================\n')
        else:
            raise Exception("ADAPT-VQE Failed -- Iterations exceeded")

    def vqe_energy(self,parameters,operators):
        psi = self.prepare_state(parameters,operators)
        energy = np.dot(psi.transpose().conj(), np.dot(self.sparse_ham, psi))[0,0]
        assert(np.isclose(energy.imag,0))
        return energy.real

    def prepare_state(self,parameters,operators):
        psi = copy.deepcopy(self.hf_ref)
        for i in reversed(range(len(parameters))):
            """
            |psi(n+1)> = exp(On)exp(On-1)...exp(O1)|psi(n)>, k = # of excitation operators
            *energy by trotterization depends on the order of operators (operators do not commute each other)
            """
            jw_sparse = openfermion.linalg.get_sparse_operator(operators[i],self.n_qubits)
            psi       = scipy.sparse.linalg.expm_multiply((parameters[i]*jw_sparse),psi)
        return psi

    def trotter_gradient(self,parameters,operators):
        grad = []
        psi = copy.deepcopy(self.hf_ref)
        for i in reversed(range(len(parameters))):
            jw_sparse = openfermion.linalg.get_sparse_operator(operators[i],self.n_qubits)
            psi = scipy.sparse.linalg.expm_multiply((parameters[i]*jw_sparse), psi)
        psi_dagger = psi.transpose().conj()

        term = 0
        ket = copy.deepcopy(psi)
        bra = np.dot(psi_dagger,self.sparse_ham) # <bra|H
        grad = self.recurse(operators,parameters,grad,ket,bra,term)
        return np.asarray(grad)

    def recurse(self,operators,parameters,grad,ket,bra,term):
        if term == 0:
            ket = ket
            bra = bra
        else:
            jw_sparse = openfermion.linalg.get_sparse_operator(operators[term-1],self.n_qubits)
            ket = scipy.sparse.linalg.expm_multiply(jw_sparse.transpose().conj()*parameters[term-1],ket)
            bra = scipy.sparse.linalg.expm_multiply(jw_sparse.transpose().conj()*parameters[term-1],\
                                                    bra.transpose().conj()).transpose().conj()

        jw_sparse = openfermion.linalg.get_sparse_operator(operators[term],self.n_qubits)
        grad.append(2 * np.dot(bra, np.dot(jw_sparse, ket))[0,0].real)
        if term < len(parameters)-1:
            term += 1
            self.recurse(operators,parameters,grad,ket,bra,term)
        return np.asarray(grad)

    def grow_ansatz(self,curr_state):
        h_ket = np.dot(self.sparse_ham, curr_state)
        norm_grad = 0.0
        max_grad = 0
        max_indx = None
        for i in range(len(self.operator_pool)):
            op_i = openfermion.linalg.get_sparse_operator(self.operator_pool[i],self.n_qubits)
            grad = 2.0 * np.dot(h_ket.transpose().conj(), np.dot(op_i, curr_state))[0,0]
            assert(np.isclose(grad.imag,0))
            grad = grad.real
            norm_grad += np.power(grad,2)
            if abs(grad) > abs(max_grad):
                max_grad = grad
                max_indx = i
        norm_grad = np.sqrt(norm_grad)
        return norm_grad, max_indx

    def singlet_GSD(self):
        self.operator_pool = []

        # Singles
        for p in range(self.n_orbitals):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(p,self.n_orbitals):
                qa  = 2 * q
                qb  = 2 * q + 1
                t1  =  openfermion.FermionOperator(((pa,1),(qa,0)))
                t1 += openfermion.FermionOperator(((pb,1),(qb,0)))
                t1 -= openfermion.hermitian_conjugated(t1)
                t1  = openfermion.normal_ordered(t1)
                #Normalization
                coeff = 0
                for t in t1.terms:
                    coeff_t = t1.terms[t]
                    coeff  += coeff_t * coeff_t
                if t1.many_body_order() > 0:
                    t1 = t1 / np.sqrt(coeff)
                    self.operator_pool.append(t1)

        # Doubles
        pq = -1
        for p in range(self.n_orbitals):
            pa = 2 * p
            pb = 2 * p + 1
            for q in range(p,self.n_orbitals):
                qa  = 2 * q
                qb  = 2 * q + 1
                pq += 1
                rs  = -1
                for r in range(self.n_orbitals):
                    ra = 2 * r
                    rb = 2 * r + 1
                    for s in range(r,self.n_orbitals):
                        sa  = 2 * s
                        sb  = 2 * s + 1
                        rs += 1
                        if(pq > rs):
                            continue

                        t2a  = openfermion.FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        t2a += openfermion.FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        t2a += openfermion.FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        t2a += openfermion.FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        t2a += openfermion.FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        t2a += openfermion.FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        t2b  = openfermion.FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        t2b += openfermion.FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        t2b += openfermion.FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        t2b += openfermion.FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                        t2a -= openfermion.hermitian_conjugated(t2a)
                        t2b -= openfermion.hermitian_conjugated(t2b)
                        t2a  = openfermion.normal_ordered(t2a)
                        t2b  = openfermion.normal_ordered(t2b)

                        #Normalization
                        coeffa = 0
                        coeffb = 0
                        for t in t2a.terms:
                            coeff_t = t2a.terms[t]
                            coeffa += coeff_t * coeff_t
                        for t in t2b.terms:
                            coeff_t = t2b.terms[t]
                            coeffb += coeff_t * coeff_t


                        if t2a.many_body_order() > 0:
                            t2a = t2a / np.sqrt(coeffa)
                            self.operator_pool.append(t2a)

                        if t2b.many_body_order() > 0:
                            t2b = t2b / np.sqrt(coeffb)
                            self.operator_pool.append(t2b)

    def define_operator(self):
        """ spin_preserved generalized uccsd ansatz """
        alpha = [orb for orb in range(self.n_qubits) if orb % 2 == 0]
        beta  = [orb for orb in range(self.n_qubits) if orb % 2 == 1]
        self.operator_pool = []

        """ single excitation """
        for p in alpha:
            for q in [ele for ele in alpha if ele > p]:
                # if abs(self.one_body[p,q]) > self.thresh:
                t1  = openfermion.FermionOperator(((q,1),(p,0)))-openfermion.FermionOperator(((p,1),(q,0)))
                t1 += openfermion.FermionOperator(((q+1,1),(p+1,0)))-openfermion.FermionOperator(((p+1,1),(q+1,0)))
                t1  = openfermion.normal_ordered(t1)
                self.operator_pool.append(t1)

        """ double excitations """
        #alpha -> alpha
        pq = 0
        for p in alpha:
            for q in [ele for ele in alpha if ele > p]:
                rs = 0
                for r in alpha:
                    for s in [ele for ele in alpha if ele > r]:
                        if pq<rs:
                            continue
                        # if abs(self.two_body[p,q,s,r]) > self.thresh:
                        t2 = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))\
                            -openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        t2 += openfermion.FermionOperator(((r+1,1),(p+1,0),(s+1,1),(q+1,0)))\
                            -openfermion.FermionOperator(((q+1,1),(s+1,0),(p+1,1),(r+1,0)))
                        t2  = openfermion.normal_ordered(t2)
                        self.operator_pool.append(t2)
                        rs += 1
                pq += 1
        #alpha -> beta
        pq = 0
        for p in alpha:
            for q in [ele for ele in beta if ele > p]:
                rs = 0
                for r in alpha:
                    for s in beta:
                        if pq<rs:
                            continue
                        # if abs(self.two_body[p,q,s,r]) > self.thresh:
                        t2 = openfermion.FermionOperator(((r,1),(p,0),(s,1),(q,0)))\
                            -openfermion.FermionOperator(((q,1),(s,0),(p,1),(r,0)))
                        t2 += openfermion.FermionOperator(((s-1,1),(q-1,0),(r+1,1),(p+1,0)))\
                            -openfermion.FermionOperator(((p+1,1),(r+1,0),(q-1,1),(s-1,0)))
                        t2  = openfermion.normal_ordered(t2)
                        self.operator_pool.append(t2)
                        rs += 1
                pq += 1

    def generate_hucc(self):
        """
        Prepare Tilde(H):
        exp{-A1}exp{-A2}exp{-A3}...exp{-An}Hexp{A1}exp{A2}exp{A3}...exp{An}
        """
        self._hucc = copy.deepcopy(self.sparse_ham)
        for m in range(len(self.ground_params)):
            jw_sparse = openfermion.linalg.get_sparse_operator(self.ground_operators[m],self.n_qubits)
            self._hucc = scipy.sparse.linalg.expm_multiply((-1 * self.ground_params[m] * jw_sparse), self._hucc)
            self._hucc = np.dot(self._hucc,scipy.sparse.linalg.expm(self.ground_params[m] * jw_sparse))
    
    @property
    def thresh(self):
        return self._thresh
    @property
    def n_electrons(self):
        return self._n_electrons
    @property
    def n_qubits(self):
        return self._n_qubits
    @property
    def n_orbitals(self):
        return self._n_orbitals
    @property
    def nuclear_repulsion(self):
        return self._nuclear_repulsion
    @property
    def hf_energy(self):
        return self._hf_energy
    @property
    def one_body_integrals(self):
        return self._one_body_integrals
    @property
    def two_body_integrals(self):
        return self._two_body_integrals
    @property
    def hf_ref(self):
        return self._hf_ref
    @property
    def sparse_ham(self):
        return self._sparse_ham
    @property
    def spin_operator(self):
        return self._spin_operator
    @property
    def ground_energy(self):
        return self._ground_energy
    @property
    def ground_params(self):
        return self._ground_params
    @property
    def ground_operators(self):
        return self._ground_operators
    @property
    def hucc(self):
        """The hucc property."""
        return self._hucc
    # @hucc.setter
    # def hucc(self, value):
    #     self._hucc = value

if __name__ == "__main__":
    import openfermionqchem
    multiplicity = 1.0
    basis = '6-31g'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.))]
    molecule = openfermion.MolecularData(geometry, basis, multiplicity)  
    path = "/Users/yongbinkim/Desktop/venv/publish/qeom-davidson/OpenFermion-QChem/qeom/test_files/"
    molecule = openfermionqchem.run_qchem(molecule,file_directory=path+'1.1/',output_name='test_qis')
    gs = Adapt_VQE(mol=molecule)
    gs.run()
    gs.generate_hucc()
    # print(gs.hucc)
