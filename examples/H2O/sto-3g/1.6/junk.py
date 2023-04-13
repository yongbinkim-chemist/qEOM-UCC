import sys
import math

def main(inputfile):
    ifile    = open(inputfile,'r')
    reflines = ifile.readlines()

    lineN = 0
    for obj in reflines:
        line  = obj.strip()
        lineN += 1
        if "OPTIMIZATION CONVERGED" in line:
            st = lineN+4
            ed = lineN+8
            break

    lineN   = 0
    string  = "$comment\n"
    string += "Water molecule\n"
    string += "$end\n\n"
    string += "$molecule\n"
    string += "0 1\n"
    coord  = []
    for obj in reflines:
        line  = obj.strip()
        lineN += 1
        if lineN > st and lineN < ed:
            string += line[2:]+'\n'
            coord.append(line.split()[2:])
    string += "$end\n\n"
    string += "$rem\n"
    string += "   n_frozen_core = 0\n"
    string += "   print_qis = true\n"
    string += "   BASIS                = sto-3g\n"
    string += "   SYMMETRY             = false\n"
    string += "   CC_SYMMETRY          = false\n"
    string += "   METHOD               = eom-ccsd\n"
    string += "   EE_SINGLETS          = [5]  ! Compute two EOM-EE singlet excited states\n"
    string += "   EE_TRIPLETS          = [5]  ! Compute two EOM-EE triplet excited states\n"
    string += "   CC_REF_PROP          = true ! Compute ground state properties\n"
    string += "   CC_EOM_PROP          = true ! Compute excited state properties\n"
    string += "   CC_OSFNO             = true\n"
    string += "   GUI                  = 2\n"
    string += "$end"


    h1x,h1y,h1z = float(coord[0][0]), float(coord[0][1]), float(coord[0][2])
    o1x,o1y,o1z = float(coord[1][0]), float(coord[1][1]), float(coord[1][2])
    h2x,h2y,h2z = float(coord[2][0]), float(coord[2][1]), float(coord[2][2])

    dx1,dy1,dz1 = o1x-h1x, o1y-h1y, o1z-h1z
    dx2,dy2,dz2 = o1x-h2x, o1y-h2y, o1z-h2z

    ofile = open('test_qis.inp','w')
    ofile.write(string)

    print(round(math.sqrt(dx1*dx1 + dy1*dy1 + dz1*dz1),5))
    print(round(math.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2),5))


main(sys.argv[1])
