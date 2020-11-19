import argparse
import numpy as np
from Bio.PDB.PDBParser import PDBParser
import Bio.PDB.vectors as Vectors


parser = argparse.ArgumentParser()
parser.add_argument("-p=", "--pdb=", type=str, required=True, dest="pdb", help="path to pdb file")
parser.add_argument("-s=", "--starting_id=", type=int, required=False, dest="starting_id", default=1, help="start index")
parser.add_argument("-c=", "--cstfile=", type=str, required=False, default=None, dest="cstfile", help="path to constraint file")
parser.add_argument("--range=", type=str, required=False, dest="range", default=None, help="range (hyphen separated)")
parser.add_argument("--types=", type=str, required=False, dest="types", default=None, help="constraint types (theta,phi,dist,omega)")
args = parser.parse_args()

if args.range:
    range_split = args.range.split("-")
    if len(range_split) == 2:
        left = int(range_split[0])
        right = int(range_split[1])
    else:
        raise Exception("invalid pdb number range!")
else:
    left = -np.inf
    right = np.inf

if args.types:
    types = set(args.types.split(","))
else:
    types = {"theta", "phi", "dist", "omega"}


indices, CA, CB, N = [], [], [], []
pdb_parser = PDBParser()
for res in pdb_parser.get_structure("", args.pdb)[0]["A"].get_residues():
    idx = res.get_id()[1]
    if left <= idx <= right:
        indices.append(idx)
        ca = res["CA"].get_vector()
        n = res["N"].get_vector()
        if "CB" in res:
            cb = res["CB"].get_vector()
        else:
            c = res["C"].get_vector()
            x2 = (ca - n).get_array()
            x3 = (c - ca).get_array()
            x1 = np.cross(x2, x3)
            cb = Vectors.Vector(-0.58273431 * x1 + 0.56802827 * x2 - 0.54067466 * x3 + ca.get_array())
        CA.append(ca)
        CB.append(cb)
        N.append(n)


n_indices = len(indices)
theta_lines, phi_lines, dist_lines, omega_lines = [], [], [], []
for i in range(n_indices):
    for j in range(n_indices):
        if i != j:
            if "theta" in types:
                theta_lines.append(f"theta   {indices[i] + args.starting_id - 1}   {indices[j] + args.starting_id - 1}   {Vectors.calc_dihedral(N[i], CA[i], CB[i], CB[j])}\n")
            if "phi" in types:
                phi_lines.append(f"phi   {indices[i] + args.starting_id - 1}   {indices[j] + args.starting_id - 1}   {Vectors.calc_angle(CA[i], CB[i], CB[j])}\n")
            if "dist" in types:
                dist_lines.append(f"dist   {indices[i] + args.starting_id - 1}   {indices[j] + args.starting_id - 1}   {np.sqrt(np.sum(np.array((CB[i] - CB[j]).get_array()) ** 2))}\n")
            if "omega" in types:
                omega_lines.append(f"omega   {indices[i] + args.starting_id - 1}   {indices[j] + args.starting_id - 1}   {Vectors.calc_dihedral(CA[i], CB[i], CB[j], CA[j])}\n")


if args.cstfile:
    cstfile = args.cstfile.rstrip(".cstfile") + ".cstfile"
else:
    cstfile = args.pdb.rstrip(".pdb") + ".cstfile"


with open(cstfile, "w") as fw:
    fw.writelines(theta_lines)
    fw.writelines(phi_lines)
    fw.writelines(dist_lines)
    fw.writelines(omega_lines)
