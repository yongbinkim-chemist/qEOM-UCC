import numpy as np
from openfermion import MolecularData
from openfermionqchem import run_qchem
from openfermionqchem import test_files
from qeom import Adapt_VQE
from qeom import Davidson, DavidsonVar

def print_test(description=None,bond_length=0.75,fidelity=None):
    judge = "Success!"
    if not all(fidelity):
        judge = "Fail!"
    print(" {} at {} Angstrom: {} {}".format(description,bond_length,fidelity,judge))

def run_test():

    multiplicity = 1.0
    basis = '6-31g'
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., 0.))]
    molecule = MolecularData(geometry, basis, multiplicity)  
    path = str(test_files.__path__._path[0])+'/'
    refe1 = np.array([-0.5669486344,-0.0388271148])
    refe2 = np.array([-0.7762007030,-0.6063834643])

    molecule1 = run_qchem(molecule,file_directory=path+'h2-321g/',output_name='test_qis')
    gs1 = Adapt_VQE(molecule1)
    gs1.run()
    qeom1 = Davidson(gs=gs1)
    # qeom1 = DavidsonVar(gs=gs1)
    qeom1.run(nroot=2,spin='singlet')
    
    molecule2 = run_qchem(molecule,file_directory=path+'h2-6311g/',output_name='test_qis')
    gs2 = Adapt_VQE(molecule2)
    gs2.run()
    qeom2 = Davidson(gs=gs2)
    # qeom2 = DavidsonVar(gs=gs2)
    qeom2.run(nroot=2,spin='triplet')
    
    print(" -------------------------------------------------------------------------------------")
    print_test(description="H2/3-21G",fidelity=np.isclose(refe1,qeom1.energy[1:]))
    print_test(description="H2/6-311G",fidelity=np.isclose(refe2,qeom2.energy[1:]))
    print(" -------------------------------------------------------------------------------------")

if __name__== "__main__":
    run_test()
