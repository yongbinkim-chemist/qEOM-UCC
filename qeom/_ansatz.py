import numpy as np
import copy
import openfermion
import cirq

class GrayCode:
    """
    *************************************************************************
    *                               GRAY CODE                               *
    *               Michael A. Nielsen and Issac L. Chuang                  *
    * Quantum Computation and Quantum Information: 10th Anniversary Edition *
    *                     doi:10.1017/CBO9780511976667                      *
    *************************************************************************
    """
    def __init__(self,n_occ,n_vir):
        self.n_occ = n_occ
        self.n_vir = n_vir
        self.qubits = cirq.LineQubit.range(self.n_occ+self.n_vir)
        self.ref = [1 for _ in range(self.n_occ)]+[0 for _ in range(self.n_vir)]
        self.ref_occ = list(range(self.n_occ))

    def control_bits_and_values(self,ref,ai,aa):
        target = copy.deepcopy(ref)
        c_qubits,c_values = [],[]
        for i in ai:
            occ = np.where(np.array(target)==1)[0]
            ctrl = [1 if q in occ else 0 for q in range(len(self.qubits))]
            ctrl.pop(i)
            c_values.append(ctrl)
            c_qubits.append([self.qubits[q] for q in range(len(self.qubits)) if q != i])
            target[i] = 0
        for a in aa:
            occ = np.where(np.array(target)==1)[0]
            ctrl = [1 if q in occ else 0 for q in range(len(self.qubits))]
            ctrl.pop(a)
            c_values.append(ctrl)
            c_qubits.append([self.qubits[q] for q in range(len(self.qubits)) if q != a])
            target[a] = 1
        return c_qubits,c_values

    def generate_circuit(self,ref,ai,aa,param):
        c_qubits,c_values = self.control_bits_and_values(ref,ai,aa)
        for i in range(len(ai)):
            yield cirq.X(self.qubits[ai[i]]).controlled_by(*c_qubits[i],control_values=c_values[i])
        for a in range(len(aa)-1):
            yield cirq.X(self.qubits[aa[a]]).controlled_by(*c_qubits[len(ai)+a],control_values=c_values[len(ai)+a])
        yield cirq.ry(param).on(self.qubits[aa[-1]]).controlled_by(*c_qubits[-1],control_values=c_values[-1])
        for a in reversed(range(len(aa)-1)):
            yield cirq.X(self.qubits[aa[a]]).controlled_by(*c_qubits[len(ai)+a],control_values=c_values[len(ai)+a])
        for i in reversed(range(len(ai))): 
            yield cirq.X(self.qubits[ai[i]]).controlled_by(*c_qubits[i],control_values=c_values[i])

class GrayGate(GrayCode):
    
    def __init__(self,n_occ,n_vir):
        super().__init__(n_occ,n_vir)

    def generate(self):
        circuit = [cirq.X(self.qubits[q]) for q in range(self.n_occ)] # HF

        for i in range(len(self.operator)):
            # maximum two configurations
            if i == len(self.operator)-1:
                param = self.killer
            else:
                param = self.params[i]
            ai = sorted(list(set(self.ref_occ)-set(self.operator[i])))
            aa = sorted(list(set(self.operator[i])-set(self.ref_occ)))
            circuit.extend(self.generate_circuit(self.ref,ai,aa,param))

        return cirq.Circuit(circuit)
        # return cirq.Circuit(circuit,strategy=cirq.InsertStrategy.NEW)

    @property
    def params(self):
        return self._params
    @params.setter
    def params(self,theta):
        self._params = theta
    @property
    def operator(self):
        return self._operator
    @operator.setter
    def operator(self,op):
        self._operator = op
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self,fact):
        self._phase = fact
    @property
    def killer(self):
        return self._killer
    @killer.setter
    def killer(self,kill):
        self._killer = kill

class GrayGateVar(GrayCode):
    
    def __init__(self,n_occ,n_vir,m_indx=1000,is_mixed=False):
        super().__init__(n_occ,n_vir)
        self.m_indx = m_indx
        self.is_mixed = is_mixed

    def generate(self):
        circuit = [cirq.X(self.qubits[q]) for q in range(self.n_occ)] # HF
        spin_complete = []
        ref = self.ref
        ref_occ = self.ref_occ
        u = 0
        for i in range(len(self.operator)):
            if self.is_mixed:
                if i == self.m_indx:
                    # param = self.sign * np.pi # kill reference
                    param = np.pi # kill reference
                elif i == len(self.operator)-1:
                    param = self.compute_killer(self.params[self.m_indx:])
                else:
                    param = self.params[u]
                    u += 1
                if i > self.m_indx:
                    ref_occ = self.operator[self.m_indx][0]
                    ref = [0 for _ in range(self.n_occ+self.n_vir)]
                    for occ in ref_occ:
                        ref[occ] = 1
            else:
                if i == len(self.operator)-1:
                    # param = self.sign * np.pi # kill reference
                    param = np.pi # kill reference
                else:
                    param  = self.params[i]

            ai = sorted(list(set(ref_occ)-set(self.operator[i][0])))
            aa = sorted(list(set(self.operator[i][0])-set(ref_occ)))
            circuit.extend(self.generate_circuit(ref,ai,aa,param))

            if len(self.operator[i]) == 2:
                if self.operator[i] in spin_complete:
                    continue
                spin_complete.append(self.operator[i])

        for i in range(len(spin_complete)):
            ai = sorted(list(set(spin_complete[i][0])-set(spin_complete[i][1])))
            aa = sorted(list(set(spin_complete[i][1])-set(spin_complete[i][0])))
            ref = [0 for _ in range(self.n_occ+self.n_vir)]
            for occ in spin_complete[i][0]:
                ref[occ] = 1
            circuit.extend(self.generate_circuit(ref,ai,aa,self.phase[i]))
        return cirq.Circuit(circuit)
        # return cirq.Circuit(circuit,strategy=cirq.InsertStrategy.NEW)

    def compute_killer(self,params):
        theta = 0.0
        for i in range(len(params)):
            t = np.tan(params[i]/2)
            for j in range(i+1,len(params)):
                t /= np.cos(params[j]/2)
            theta -= t

        x = theta / np.sqrt(2)
        if abs(x) > 1.0:
            x = 1.0
        return 2 * np.arcsin(x) - np.pi/2

    @property
    def params(self):
        return self._params
    @params.setter
    def params(self,theta):
        self._params = theta
    @property
    def operator(self):
        return self._operator
    @operator.setter
    def operator(self,op):
        self._operator = op
    @property
    def phase(self):
        return self._phase
    @phase.setter
    def phase(self,phi):
        self._phase = phi


class SSQUARE(object):
    """
    Generate S^2 Operator
    """
    def __init__(self,n_electrons=None,n_qubits=None):
        self.n_electrons = n_electrons
        self.n_qubits    = n_qubits

    def generate(self):
        """
        Circuit will be implemented in the future
        """
        sz    = np.where(np.arange(self.n_qubits) % 2 == 0, 0.5, -0.5)
        table = self.s2_matrix(sz)
        coeff = np.array([3 / 4 * self.n_electrons])
        ops   = [[]]
        for t in table:
            coeff = np.concatenate((coeff, np.array([t[4]])))
            ops.append([int(t[0]), int(t[1]), int(t[2]), int(t[3])])
        s2 = openfermion.FermionOperator()
        for op in range(len(ops)):
            if len(ops[op]) == 0:
                s2 += coeff[op]
            else:
                p,q,r,s = ops[op]
                s2 += openfermion.FermionOperator(((p,1), (q,1), (r,0), (s,0)), coeff[op])

        s2 = openfermion.jordan_wigner(s2)
        s2.compress()
        return openfermion.linalg.get_sparse_operator(s2,self.n_qubits)

    def s2_matrix(self,sz):

        N     = np.arange(sz.size)
        alpha = N.reshape(-1, 1, 1, 1)
        beta  = N.reshape(1, -1, 1, 1)
        gamma = N.reshape(1, 1, -1, 1)
        delta = N.reshape(1, 1, 1, -1)

        # we only care about indices satisfying the following boolean mask
        mask = np.logical_and(alpha // 2 == delta // 2, beta // 2 == gamma // 2)

        # diagonal elements
        diag_mask    = np.logical_and(sz[alpha] == sz[delta], sz[beta] == sz[gamma])
        diag_indices = np.argwhere(np.logical_and(mask, diag_mask))
        diag_values  = (sz[alpha] * sz[beta]).flatten()
        diag         = np.vstack([diag_indices.T, diag_values]).T

        # off-diagonal elements
        m1 = np.logical_and(sz[alpha] == sz[delta] + 1, sz[beta] == sz[gamma] - 1)
        m2 = np.logical_and(sz[alpha] == sz[delta] - 1, sz[beta] == sz[gamma] + 1)

        off_diag_mask    = np.logical_and(mask, np.logical_or(m1, m2))
        off_diag_indices = np.argwhere(off_diag_mask)
        off_diag_values  = np.full([len(off_diag_indices)], 0.5)
        off_diag         = np.vstack([off_diag_indices.T, off_diag_values]).T

        # combine the off diagonal and diagonal tables into a single table
        return np.vstack([diag, off_diag])

class UCCSD(object):
    def __init__(self,n_occ,n_vir):
        self.n_occ  = n_occ
        self.n_vir  = n_vir
        self.qubits = cirq.LineQubit.range(self.n_occ+self.n_vir)

    def quantum_gate(self,gate,v):
        if 'H' in gate:
            _, q = gate.split()
            yield cirq.H(self.qubits[int(q)])
        elif 'CNOT' in gate:
            _, q1, q2 = gate.split()
            yield cirq.CNOT(self.qubits[int(q1)],self.qubits[int(q2)])
        elif 'Rx' in gate:
            _, theta, q = gate.split()
            yield cirq.rx(float(theta)).on(self.qubits[int(q)])
        elif 'Rz' in gate:
            _, theta, q = gate.split()
            yield cirq.rz(float(v)).on(self.qubits[int(q)])
            # yield cirq.rz(float(theta)).on(self.qubits[int(q)])

    def to_pauli_string(self,ops,v):
        string = ""
        for pauli in ops:
            string += str(pauli[1])+str(pauli[0])+" "
        return v * openfermion.QubitOperator(string)

    def generate(self):
        # circuit = [cirq.X(self.qubits[q]) for q in range(self.n_occ)]
        circuit = []

        jw  = self.params[0] * openfermion.jordan_wigner(self.operator[0])
        for i in range(1,len(self.params),1):
            jw += self.params[i] * openfermion.jordan_wigner(self.operator[i])
        for k, v in jw.terms.items():
            pauli = self.to_pauli_string(k,v)
            qasm  = openfermion.circuits.trotterize_exp_qubop_to_qasm(pauli)
            for gate in qasm:
                circuit.extend(self.quantum_gate(gate,-2.0*v.imag))
        # for i in range(len(self.params)):
        #     jw = self.params[i] * openfermion.jordan_wigner(self.operator[i])
        #     for k, v in jw.terms.items():
        #         pauli = self.to_pauli_string(k,v)
        #         qasm  = openfermion.circuits.trotterize_exp_qubop_to_qasm(pauli)
        #         for gate in qasm:
        #             circuit.extend(self.quantum_gate(gate,-2.0*v.imag))
        # return cirq.Circuit(circuit,strategy=cirq.InsertStrategy.NEW)
        return cirq.Circuit(circuit)

    @property
    def params(self):
        return self._params
    @params.setter
    def params(self,theta):
        self._params = theta
    @property
    def operator(self):
        return self._operator
    @operator.setter
    def operator(self,op):
        self._operator = op

class BASIS(object):
    def __init__(self,n_occ,n_vir):
        self.n_occ  = n_occ
        self.n_vir  = n_vir
        self.qubits = cirq.LineQubit.range(self.n_occ+self.n_vir)

    def generate(self):
        circuit = [cirq.X(self.qubits[q]) for q in self.operator]
        return cirq.Circuit(circuit)

    @property
    def operator(self):
        return self._operator
    @operator.setter
    def operator(self,op):
        self._operator = op
