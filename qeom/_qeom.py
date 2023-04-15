import time, itertools, copy, operator
from collections import defaultdict
from functools import reduce  # omit on Python 2
import numpy as np
import scipy
import openfermion, cirq, qeom

au2ev = 27.211324570273
class qEOM(object):
    """
    qEOM master class
    """
    def __init__(self,gs=None):
        self._gs = gs
        self._qubits = cirq.LineQubit.range(self.gs.n_qubits)
        self._n_occ = self.gs.n_electrons
        self._n_vir = self.gs.n_qubits-self.gs.n_electrons
        self.gs.generate_hucc()

    def operator_pool(self,spin="singlet",level=2):
        nCm = list(itertools.combinations(range(self.gs.n_qubits),self.gs.n_electrons))
        dets = [det for det in nCm if self.is_valid_basis(det,level)]
        basis_pool = defaultdict(list)
        for det in dets:
            # basis det can be generated only with n NOT gates, n: # of electrons
            psi = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(det),self.gs.n_qubits))
            Hii = np.dot(psi,np.dot(self.gs.hucc,psi.transpose().conj()))[0,0]
            assert(np.isclose(Hii.imag,0))
            basis_pool[np.round(Hii.real,10)].append(list(det))

        pool,diag = [],[]
        for Hii in basis_pool:
            # degeneracy
            degenerate_dets = basis_pool.get(Hii)
            if spin == "triplet" and len(degenerate_dets) == 1:
                continue
            elif len(degenerate_dets) == 4:
                temp = self.degeneracy_split(degenerate_dets)
                pool.extend(temp)
                diag.extend([Hii for _ in range(len(temp))])
            else:
                pool.append(degenerate_dets)
                diag.append(Hii)
        indx = np.array(diag).argsort()
        pool = np.array(pool,dtype=object)[indx].tolist()
        diag = np.array(diag,dtype=object)[indx].tolist()
        return pool,diag

    def is_valid_basis(self,det,level):
        ref = range(self.gs.n_electrons)
        ref_ms = sum([0.5 if spin % 2 == 0 else -0.5 for spin in list(ref)])
        det_ms = sum([0.5 if spin % 2 == 0 else -0.5 for spin in det])
        dms = det_ms - ref_ms
        LoE = len(list(set(ref)-set(det)))
        if set(det) == set(ref):
            return False # no need HF reference
        elif dms != 0:
            return False # spin preserved excitations
        elif LoE > level:
            return False # more than given level of excitations
        else:
            return True

    def degeneracy_split(self,dets):
        # hard coding
        confs = defaultdict(list)
        for det in dets:
            spatial = [occ // 2 for occ in det]
            confs[sum(spatial)].append(det)
        if len(confs.keys()) == 2:
            return list(confs.values())
        confs = defaultdict(list)
        for det in dets:
            xdif = reduce(operator.__sub__, sorted(det)[:2])
            confs[xdif].append(det)
        if len(confs.keys()) == 2:
            return list(confs.values())
    
    def residual(self,pool,curr_state,curr_energy,diag_enes,r_tol=1.0e-3,tol=1.0e-9):
        """ 
        Davidson Part I. (Convergency check)
        residual = <Phi_{i,j,..}^{a,b,...}|(hucc|curr_state> - curr_energy|curr_state>)
        """
        corr_ops,corr_amp,corr_amp_var,resnorm = [],[],[],[]
        for i in range(len(curr_state)):
            kroot,kresn,kresn_var,knorm = [],[],[],[]
            for j in range(len(pool)):
                temp1,temp2,temp3 = [],[],[]
                for det in pool[j]:
                    psi = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(list(det),self.gs.n_qubits))
                    res = np.dot(psi,np.dot(self.gs.hucc,curr_state[i]))[0,0] - curr_energy[i]*np.dot(psi,curr_state[i])[0,0]
                    if abs(res.real) < tol:
                        continue
                    temp1.append(det)
                    temp2.append(res.real / (curr_energy[i]-diag_enes[i]))
                    temp3.append(res.real)
                    knorm.append(res.real)
                if len(temp1) > 0:
                    kroot.append(temp1)
                    kresn.append(temp2)
                    kresn_var.append(temp3)
            if np.linalg.norm(knorm) < r_tol:
                continue
            resnorm.extend(knorm)
            corr_ops.append(kroot)
            corr_amp.append(kresn)
            corr_amp_var.append(kresn_var)
        return np.linalg.norm(resnorm),corr_ops,corr_amp,corr_amp_var
    
    def qubit_to_text(self,qubit_repr):
        comp,ampl = scipy.sparse.find(qubit_repr)[0],scipy.sparse.find(qubit_repr)[2]
        text_repr = ""
        for i in range(len(comp)):
            if abs(ampl[i]) > 1.0e-1:
                string = '{0:0b}'.format(comp[i]).zfill(self.gs.n_qubits)
                sign = "+"
                if np.sign(ampl[i]) == -1.0:
                    sign = "-"
                text_repr += " {}{:>8.5f}|{}>".format(sign,abs(ampl[i].real),string)
        return text_repr

    def watermark(self,var=False):
        print("*****************************************************")
        print("*           Start qEOM-DAVIDSON algorithm           *")
        print("*          Yongbin Kim and Anna I. Krylov           *")
        print("*****************************************************")
        if var:
            print(" -------------------------------------------------------------------------------------")
            print("      Iter  Root  NVecs  ResNorm     Total energy (a.u.)    Excitation energy (eV.)   ")
            print(" -------------------------------------------------------------------------------------")
        else:
            print("-"*80)
            print("     Iter   NVecs  ResNorm     Total energy (a.u.)   ")
            print("-"*80)


    def display(self,nroot,eom_start):
        print(" -------------------------------------------------------------------------------------")
        print("   Root #    a.u.       eV       <S^2>    Wavefunction")
        print(" -------------------------------------------------------------------------------------")
        for k in range(nroot):
            print("{:>6}  {:>10.6f}  {:>10.6f}  {:>5.2f}".format(k+1, self.energy[k+1],\
                   (self.energy[k+1]-self.energy[0])*au2ev, self.s2[k]),self.qubit_to_text(self.state[k+1]))
        print("\n qEOM-Davidson calculation: {:.2f} s".format(time.time()-eom_start))
        print(" =====================================================================================\n")

    @property
    def n_occ(self):
        return self._n_occ
    @n_occ.setter
    def n_occ(self, value):
        self._n_occ = value
    @property
    def n_vir(self):
        return self._n_vir
    @n_vir.setter
    def n_vir(self, value):
        self._n_vir = value
    @property
    def gs(self):
        return self._gs
    @property
    def qubits(self):
        return self._qubits
 
class Davidson(qEOM):
    """
    Diagonalization-based
    """
    def __init__(self,gs=None,):
        super().__init__(gs=gs)

    def run(self,nroot=1,spin='singlet',level=2,user_guess=None,r_tol=1.0e-3,max_iter=100):
        eom_start = time.time()
        # fact = {"singlet":-1,"triplet":1}
        # phase = fact.get(spin)
        # s2_expt = {"singlet":0.0,"triplet":2.0}
        phase = {"singlet":-1,"triplet":1}.get(spin)
        self.energy = [self.gs.ground_energy]
        self.state = [self.gs.hf_ref]
        self.watermark() 

        if user_guess != None:
            nroot = len(user_guess)
        L = nroot
        pool,diag = self.operator_pool(spin=spin,level=level)
        pool,diag,operators,operator_enes = self.iguess(pool,diag,nroot=nroot,user_guess=user_guess)
        guess_space,guess_amps = self.iguess_to_qubit(operators,phase)
        curr_energy,curr_state,curr_s2 = self.compute_energy(guess_space,nroot)
        for eom_iter in range(max_iter):
            resnorm,corr_ops,corr_amp,_ = self.residual(pool,curr_state,curr_energy,operator_enes,r_tol=r_tol)
            corr_ops,corr_amp = [sum(ops,[]) for ops in corr_ops],[sum(amp,[]) for amp in corr_amp]
            self.log(nroot,L,resnorm,curr_energy,eom_iter)
            if resnorm < r_tol:
                self.energy.extend(curr_energy)
                self.state.extend(curr_state)
                self.s2 = curr_s2
                break
            # guess_space,guess_ops,guess_amps = self.gram_schmidt(guess_space,guess_ops,guess_amps,corr_ops,corr_amp)
            guess_space,guess_amps,operators = self.gram_schmidt(guess_space,guess_amps,operators,corr_ops,corr_amp)
            curr_energy,curr_state,curr_s2 = self.compute_energy(guess_space,nroot)
            L = len(guess_space)
        self.display(nroot,eom_start)

    def compute_energy(self,guess_space,nroot):
        N = len(guess_space)
        Hsub = np.zeros((N,N))
        for i in range(N):
            for j in range(i,N):
                Hsub[i,j] = np.dot(guess_space[i].transpose().conj(),np.dot(self.gs.hucc,guess_space[j]))[0,0].real
                if i != j:
                    Hsub[j,i] = Hsub[i,j]
        e,v = np.linalg.eigh(Hsub)
        curr_energy = e[:nroot]
        curr_state = []
        curr_s2 = []
        for i in range(nroot):
            psi = guess_space[0] * v[:,i][0]
            for j in range(1,N):
                psi += guess_space[j] * v[:,i][j]
            s2 = np.round(np.dot(psi.transpose().conj(),np.dot(self.gs.spin_operator,psi))[0,0].real,3)
            curr_state.append(psi)
            curr_s2.append(s2)
        return curr_energy,curr_state,curr_s2
 

    def gram_schmidt(self,guess_space,guess_amps,guess_ops,corr_ops,corr_amp):
        """ 
        Davidson Part II. (Correction vectors)
        Compute correction vectors "classically"
        ** One can guess the quantum states based on the operators and amplitudes
        """
        for i in range(len(corr_ops)):
            # check orthogonality
            ri = corr_ops[i]
            ci = corr_amp[i]
            overlap,orthogonal = self.compute_overlap(ri,ci,guess_ops,guess_amps)
            if orthogonal:
                ci /= np.linalg.norm(ci)
                parameters = self.compute_parameters(amplitudes=ci)
                corr_vec = self.prepare_state(parameters,ri,ci)
            else:
                corr_vec,ri,ci = self.compute_orthogonal_vector(overlap,ri,ci,guess_ops,guess_amps)
            guess_space.append(corr_vec)
            guess_ops.append(ri)
            guess_amps.append(ci)
        return guess_space,guess_amps,guess_ops
    
    def compute_overlap(self,ri,ci,guess_ops,guess_amps):
        overlap = []
        orthogonal = True
        for i in range(len(guess_ops)):
            bi = guess_ops[i]
            dot_prd = 0.0
            for j in range(len(ri)):
                if ri[j] in bi:
                    indx = bi.index(ri[j])
                    dot_prd += guess_amps[i][indx] * ci[j]
            overlap.append(dot_prd)
            if dot_prd != 0.0:
                orthogonal = False
        return overlap,orthogonal
    
    def compute_orthogonal_vector(self,overlap,corr_ops,corr_amp,guess_ops,guess_amps):
        """
        Gram-Schmidt Process
        W - <W|V> * V, we know what the quatum states are based on the operators and measured amplitudes
        Guess vectors are already normalized
        """
        operators = defaultdict(list)
        amplitude = defaultdict(list)
        for i in range(len(overlap)):
            if overlap[i] == 0.0:
                continue
            for j in range(len(guess_ops[i])):
                operators[str(guess_ops[i][j])].append(guess_ops[i][j])
                amplitude[str(guess_ops[i][j])].append(-overlap[i]*guess_amps[i][j])
        new_operators = []
        new_amplitude = []
        for i in range(len(corr_ops)):
            if str(corr_ops[i]) in operators.keys():
                new_operators.append(operators.get(str(corr_ops[i]))[0])
                new_amplitude.append(corr_amp[i]+sum(amplitude.get(str(corr_ops[i]))))
                operators.pop(str(corr_ops[i]))
                amplitude.pop(str(corr_ops[i]))
            else:
                new_operators.append(corr_ops[i])
                new_amplitude.append(corr_amp[i])
        for k in amplitude.keys():
            new_amplitude.append(sum(amplitude.get(k)))
            new_operators.append(operators.get(k)[0])

        new_amplitude /= np.linalg.norm(new_amplitude)
        parameters = self.compute_parameters(amplitudes=new_amplitude)
        psi = self.prepare_state(parameters,new_operators,new_amplitude)
        return psi,new_operators,new_amplitude

    def iguess(self,pool,diag,nroot=1,user_guess=None):
        if user_guess == None:
            # Koopamans'
            bi = pool[:nroot]
            bi_ene = diag[:nroot]
            del pool[:nroot]
            del diag[:nroot]
        else:
            # User defined guess
            guess_indx = []
            for guess in user_guess:
                guess_indx.append(pool.index(guess))
            bi = [pool[indx] for indx in guess_indx]
            bi_ene = [diag[indx] for indx in guess_indx]
            pool = [pool[indx] for indx in range(len(pool)) if indx not in guess_indx]
            diag = [diag[indx] for indx in range(len(diag)) if indx not in guess_indx]
        # vector space (bi excluded), vector space energies, guess, guess energies
        return pool,diag,bi,bi_ene

    def iguess_to_qubit(self,operators,phase):
        guess_space,guess_amps = [],[]
        print(" Initial guess vectors")
        i = 1
        for bi in operators:
            amplitude = [1.0] # |a>
            if len(bi) == 2: # (|a> +- |b>) / sqrt(2)
                amplitude = [np.sin(np.pi/4),np.sin(phase*np.pi/4)]
            param = self.compute_parameters(amplitude)
            state = self.prepare_state(param,bi,amplitude)
            guess_space.append(state)
            guess_amps.append(amplitude)
            print(" |b{}> =".format(i), self.qubit_to_text(state))
            i += 1
        return guess_space,guess_amps

    def prepare_state(self,parameters,operators,amplitude):
        killer = np.pi
        cos = 1.0
        for param in parameters:
            cos *= np.cos(param/2.0)
        if np.sign(amplitude[-1]) != np.sign(cos):
            killer = -np.pi
        ansatz = qeom.GrayGate(self.n_occ,self.n_vir)
        ansatz.operator = operators
        ansatz.params = parameters
        ansatz.killer = killer
        circuit = ansatz.generate()
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit,qubit_order=self.qubits)
        psi = scipy.sparse.csc_matrix(result.final_state_vector).transpose().conj()
        # print(circuit.to_text_diagram(transpose=True,use_unicode_characters=False)) # YB
        return psi

    def compute_parameters(self,amplitudes=None):
        # From amplitudes -> compute gray code circuit parameters
        parameters = []
        for i in range(len(amplitudes)-1):
            theta = amplitudes[i]
            for prev_theta in parameters:
                theta /= np.cos(prev_theta/2)
            if theta > 1.0 or theta < -1.0:
                if abs(theta)-1.0 > 1.0e-2:
                    theta = np.pi # kill operator
                else:
                    theta = np.round(theta,2)
            parameters.append(2.0 * np.arcsin(np.round(theta,10)))
        return parameters

    def log(self,nroot,L,resnorm,curr_energy,eom_iter,conv=False):
        output = "{:>6} {:>6} {:>11.4f}".format(eom_iter+1,L,resnorm)
        for i in range(len(curr_energy[:nroot])):
            output += "{:>18.10f}".format(curr_energy[i])
        print(output)

class DavidsonVar(qEOM):
    """
    Diagonalization-based
    """
    def __init__(self,gs=None,):
        super().__init__(gs=gs)

    def run(self,nroot=1,spin='singlet',level=2,user_guess=None,r_tol=1.0e-3,beta=10.0,max_iter=100):
        eom_start = time.time()
        fact = {"singlet":-1,"triplet":1}.get(spin)
        self.tot_spin = {"singlet":0.0,"triplet":2.0}.get(spin)
        self.energy = [self.gs.ground_energy]
        self.state = [self.gs.hf_ref]
        self.s2 = []

        pool,diag = self.operator_pool(spin=spin,level=level)
        if user_guess != None:
            nroot = len(user_guess)
            space = []
            for guess in user_guess:
                space.append(pool.index(guess))
        else:
            space = list(range(nroot))
 
        self.watermark(var=True)
        l = 1 
        for k in space:
            self.ansatz = qeom.GrayGateVar(self.n_occ,self.n_vir)
            op_pool,operators,parameters,phases,op_enes = self.iguess(pool,diag,fact,k)
            curr_energy,curr_state,curr_s2 = self.compute_ienergy(parameters,operators,phases)
            nvec = len(operators[0]) 

            print(" Initial guess vectors")
            print(" |b{}> =".format(l), self.qubit_to_text(curr_state))
            l += 1
            is_mixed = False
            m_indx = 1000
            
            for eom_iter in range(max_iter):

                # resnorm,corr_ops,_,corr_amp = self.residual(op_pool,[curr_state],[curr_energy],op_enes,r_tol=r_tol)
                resnorm,corr_ops,corr_amp,_ = self.residual(op_pool,[curr_state],[curr_energy],op_enes,r_tol=r_tol)
                corr_ops,corr_amp = sum(corr_ops,[]),sum(corr_amp,[])
                self.log(k,nvec,resnorm,curr_energy,eom_iter)

                if np.linalg.norm(resnorm) < r_tol:
                    self.energy.append(curr_energy)
                    self.state.append(curr_state)
                    self.s2.append(curr_s2)
                    break

                is_mixed,ops,phase = self.grow_ansatz(corr_ops,corr_amp)
                if is_mixed:
                    indx = [op_pool.index(op) for op in ops]
                    m_indx = len(operators)
                    parameters.append(np.pi)
                    parameters.extend([0.0 for _ in range(len(ops)-2)])
                    operators.extend(ops); op_pool = [ops for p,ops in enumerate(op_pool) if p not in indx]
                    phases.extend(phase)
                    self.ansatz = qeom.GrayGateVar(self.n_occ,self.n_vir,m_indx,is_mixed)
                else:
                    indx = op_pool.index(ops[0])
                    parameters.insert(0,0.0)
                    operators.insert(0,op_pool[indx]); op_pool.pop(indx)
                    if phase != None:
                        phases.insert(0,phase)

                    if self.ansatz.is_mixed:
                        self.ansatz.m_indx += 1

                bound  = [(-2*np.pi,2*np.pi) for _ in range(len(parameters))]
                result = scipy.optimize.minimize(self.compute_energy,parameters,\
                                                 args=(operators,phases,beta),\
                                                 method='L-BFGS-B',bounds=bound,\
                                                 options={'disp': None,'gtol':1e-05,'eps':1e-03})

                curr_energy = self.vqe_energy
                parameters = result.x.tolist()
                _,curr_state,curr_s2 = self.compute_ienergy(parameters,operators,phases)
                nvec = 0
                for p in range(len(operators)):
                    nvec += len(operators[p])

        idx = np.array(self.energy).argsort()
        self.energy = np.array(self.energy)[idx]
        self.state  = np.array(self.state)[idx]
        self.display(nroot,eom_start)

    def compute_energy(self,parameters,operators,phases,beta):
        psi = self.prepare_state(parameters,operators,phases)
        energy = np.dot(psi.transpose().conj(), np.dot(self.gs.hucc,psi))[0,0]
        assert(np.isclose(energy.imag,0))
        perturb = 0.0
        for state in self.state:
            perturb += beta * pow(np.dot(state.transpose().conj(),psi),2)[0,0]
        self.vqe_energy = energy.real
        return energy.real + perturb.real

    def compute_ienergy(self,parameters,operators,phases):
        curr_state = self.prepare_state(parameters,operators,phases)
        curr_energy = np.dot(curr_state.transpose().conj(),np.dot(self.gs.hucc,curr_state))[0,0].real
        curr_s2 = np.dot(curr_state.transpose().conj(),np.dot(self.gs.spin_operator,curr_state))[0,0].real
        return curr_energy,curr_state,curr_s2
    
    def iguess(self,pool,diag,phase,k):
        operator_pool = copy.deepcopy(pool)
        operator_enes = copy.deepcopy(diag)
        operators = [operator_pool[k]]; operator_pool.pop(k)
        diag_enes = [operator_enes[k]]; operator_enes.pop(k)
        parameters = []
        phases = []
        if len(operators[0]) == 2:
            phases = [np.pi/2 * phase]
        return operator_pool,operators,parameters,phases,diag_enes

    def grow_helper(self,ops,amp,phase=None):
        psi = scipy.sparse.csc_matrix((1,pow(2,self.gs.n_qubits)), dtype='float64')
        norm = amp / np.linalg.norm(amp)
        for i in range(len(ops)):
            psi += norm[i] * scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ops[i],self.gs.n_qubits))
        s2 = np.round(np.dot(psi,np.dot(self.gs.spin_operator,psi.transpose().conj()))[0,0].real,2)
        if len(ops) == 2:
            phase = np.pi / 2
            if np.sign(amp[0]) != np.sign(amp[1]):
                phase = -1.0 * np.pi / 2
        return s2,phase

    def grow_ansatz(self,corr_ops,corr_amp):
    
        amp_list = [abs(x) for amp in corr_amp for x in amp[:1]]
        max_indx = amp_list.index(max(amp_list))
        s2,phase = self.grow_helper(corr_ops[max_indx],corr_amp[max_indx])
  
        if self.tot_spin == s2:
            return False,[corr_ops[max_indx]],phase
        else:
            mix_ops,mix_indx,mix_phase = [corr_ops[max_indx]],[max_indx],[phase]
            for i in reversed(np.array(amp_list).argsort()[:-1]):
                s2,phase = self.grow_helper(corr_ops[i],corr_amp[i])
                if self.tot_spin == s2:
                    continue
                mix_indx.append(i)
                mix_ops.append(corr_ops[i])
                if phase != None:
                    mix_phase.append(phase)
            return True,mix_ops,mix_phase

    def prepare_state(self,parameters,operators,phases):
        self.ansatz.operator = operators
        self.ansatz.params = parameters
        self.ansatz.phase = phases
        circuit = self.ansatz.generate()
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit,qubit_order=self.qubits)
        psi = scipy.sparse.csc_matrix(result.final_state_vector).transpose().conj()
        return psi

    def log(self,k,nvec,resnorm,curr_energy,eom_iter):
        excitationE = (curr_energy - self.gs.ground_energy) * au2ev
        print("{:>7} {:>5} {:>5} {:>12.4f} {:>18.10f} {:>23.10f}".format(eom_iter+1,k+1,nvec,resnorm,\
                                                                        curr_energy,excitationE))
if __name__ == "__main__":
    import openfermionqchem

    basis        = 'sto-3g'
    multiplicity = 1
    geometry       = [('H', (0., 0., 0.)), ('H', (0., 0., 2.2))]
    molecule       = openfermion.MolecularData(geometry, basis, multiplicity)
    directory      = "/Users/yongbinkim/Desktop/venv/publish/qeom-davidson/OpenFermion-QChem/examples/"
    system         = 'H4/eom-ccsd/sto-3g/2.20/'
    molecule       = openfermionqchem.run_qchem(molecule,file_directory=directory+system,output_name='test_qis')

    # ground state
    gs = qeom.Adapt_VQE(molecule)
    gs.run()

    # qeom_davidson = qeom.Davidson(gs=gs)
    qeom_davidson = qeom.DavidsonVar(gs=gs)

    # singlet
    user_guess = [[[0,1,2,5],[0,1,3,4]],[[0,1,2,7],[0,1,3,6]],[[0,1,4,5]]]
    qeom_davidson.run(nroot=3,user_guess=user_guess,spin='singlet',level=4)
