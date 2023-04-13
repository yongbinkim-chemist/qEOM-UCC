import numpy as np
import openfermion
from openfermion.chem import MolecularData
from openfermionqchem import run_qchem
import qeom

import pyscf
import pyscf_helper
from pyscf import lib
from pyscf import gto, scf, mcscf, fci, ao2mo, lo,  cc
from pyscf.cc import ccsd

def print_result(bond,energy):   
    string1,string2 = "Total Energy\n","Excitation Energy\n"
    
    for i in range(len(bond)):
        string1 += "{:4.2f}".format(bond[i])
        string2 += "{:4.2f}".format(bond[i])

        for j in range(len(energy[i])):
            string1 += "{:15.10f}".format(energy[i][j])
        string1 += '\n'

        for j in range(1,len(energy[i])):
            string2 += "{:15.10f}".format((energy[i][j] - energy[i][0]) * 27.211324570273)
        string2 += '\n'
        
    print(string1+'\n'+string2)

def test():
    #geometry = [('O', (0,0,0)), ('H', ( 0.95700111, 0.00000,  0.00000000)), ('H', (-0.23961394, 0.00000,  0.92651836))]

    singlet = []
    triplet = []
    bond_lengths = []

    for rrr in range(1):
        r0 = 1.3 + 0.1 * rrr
        geometry = '''
        O
        H   1   {}
        H   1   {}   2   104.5
        '''.format(r0,r0)

        charge = 0
        spin   = 0
        basis  = 'sto3g'
        [n_orb, n_a, n_b, h, g, mol, E_nuc, E_scf, C, S] = pyscf_helper.init(geometry,charge,spin,basis,n_frzn_occ=1,n_act=6)
        sq_ham = pyscf_helper.SQ_Hamiltonian()
        sq_ham.init(h, g, C, S)
        print(" HF Energy: %12.8f" %(E_nuc + sq_ham.energy_of_determinant(range(n_a),range(n_b))))
        fermi_ham   = sq_ham.export_FermionOperator()
        fermi_ham  += openfermion.FermionOperator((),E_nuc)
        hamiltonian = openfermion.get_sparse_operator(fermi_ham)

        geometry  = [('H', (0., 0., 0.)), ('H', (0., 0., r0))]
        molecule  = MolecularData(geometry, basis, 1)
        directory = '/scratch/brown/kim2096/qeom/OpenFermion-QChem/examples/'
        system    = 'H2O/eom-ccsd/sto-3g/'+str(round(r0,2))+'/'
        molecule  = run_qchem(molecule,file_directory=directory+system,output_name='test_qis')

        print("r = {:4.2f}".format(r0))
        bond_lengths.append(r0)

        gs = qeom.Adapt_VQE(molecule,frozen_core=True)
        gs.run(hamiltonian=hamiltonian)
        qeom_davidson = qeom.Davidson(gs=gs)

        # singlet
        qeom_davidson.run(nroot=2,spin='singlet',level=3)
        singlet.append(qeom_davidson.energy)   

        # triplet
        qeom_davidson.run(nroot=2,spin='triplet',level=3,r_tol=1.0e-4,max_iter=150)
        triplet.append(qeom_davidson.energy)

    singlet      = np.array(singlet)
    triplet      = np.array(triplet)
    bond_lengths = np.array(bond_lengths)

    print("Singlet")
    print_result(bond_lengths,singlet)

    print("Triplet")
    print_result(bond_lengths,triplet)

if __name__== "__main__":
    test()
