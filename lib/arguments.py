import argparse
import time
import os
import numpy as np
from lib import utils


def hallucinate_get_arguments():   # tested
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length=", type=int, required=False, dest="length", default=60, help="sequence length")
    parser.add_argument("-a", "--aln=", type=str, required=False, dest="aln", default="", help="path to starting alignment file")
    parser.add_argument("-r", "--resfile=", type=str, required=False, dest="resfile", default="", help="path to resfile")
    parser.add_argument("-c", "--cstfile=", type=str, required=False, dest="cstfile", default="", help="path to constraint file")
    parser.add_argument("-o", "--output", type=str, required=False, dest="output", default=f"output_{time.strftime('%Y%m%d%H%M%S')}.csv", help="path to output file")
    parser.add_argument("-z", "--npz", type=str, required=False, dest="npz", default=None, help="path to npz file")
    parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory", default="../models/trmodel", help="path to trRosetta network weights")
    parser.add_argument("--background=", type=str, required=False, dest="background_directory", default="../models/bkgrd", help="path to background network weights")
    parser.add_argument("--aa_weight=", type=float, required=False, dest="aa_weight", default=0.0, help="weight for the aa composition biasing loss term")
    parser.add_argument("--cst_weight=", type=float, required=False, dest="cst_weight", default=1.0, help="weight for the constraints loss term")
    parser.add_argument("--schedule=", type=str, required=False, dest='schedule',
                        default="0.1,20000,2.0,5000", help="simulated annealing schedule: 'T0,n_steps,decrease_factor,decrease_range'")
    args = parser.parse_args()
    return args


def predict_get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a=", "--aln=", type=str, required=False, dest="aln", default="", help="path to alignment file")
    parser.add_argument("-z=", "--npz=", type=str, required=False, dest="npz", default=None, help="path to npz file")
    parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory",
                        default="../models/trmodel", help="path to trRosetta network weights")
    args = parser.parse_args()
    return args


def parse_aln(starting_aln):   # tested
    """
    :usage: get amino acid sequences from starting alignment file (or fasta file)
    :param: starting_aln, str
    :return: msa, list of str
    """
    if starting_aln:
        if os.path.isfile(starting_aln):
            with open(starting_aln) as fr:
                lines = fr.readlines()
                msa = []
                for i in range(len(lines) // 2):
                    if len(lines[2 * i]) > 2 and lines[2 * i][0] == ">" and len(lines[2 * i + 1]) > 1:
                        seq = lines[2 * i + 1].rstrip()
                        unk_aa = set(seq) - {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"}
                        assert not unk_aa, "File {}, line {}, {} cannot be recognized!".format(starting_aln, 2 * i + 2, ",".join(list(unk_aa)))
                        msa.append(seq)
                    else:
                        raise Exception(f"File {starting_aln} cannot be recognized!")
        else:
            raise Exception(f"File {starting_aln} does not exist!")
        return msa


def parse_resfile(resfile, L):   # tested
    if resfile:
        if os.path.isfile(resfile):
            aa_allowed = np.ones((L, 20))
            with open(resfile) as fr:
                for line_idx, line in enumerate(fr.readlines()):
                    split = line.split()
                    assert len(split) == 2, f"Line {line_idx + 1} in {resfile} is badly formatted!"
                    pdbnum = int(split[0])
                    assert pdbnum <= L, f"pdb number in line {line_idx + 1} in {resfile} is larger than the length of the sequence!"
                    aas = set(split[1])
                    total_aas = {"A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"}
                    unk_aa = aas - total_aas
                    assert not unk_aa, "{} in line {} in {} cannot be recognized!".format(",".join(list(unk_aa)), line_idx + 1, resfile)
                    for aa in total_aas - aas:
                        aa_allowed[pdbnum-1][utils.aa2idx[aa]] = 0
            aa_allowed = np.array(aa_allowed)
            return aa_allowed
        else:
            raise Exception(f"{resfile} does not exist!")


def parse_cstfile(cstfile, L):
    if cstfile:
        if os.path.isfile(cstfile):
            constraints, constraints_ = {}, {}
            with open(cstfile) as fr:
                for line in fr.readlines():
                    split = line.split()
                    type_ = split[0]
                    i = int(split[1])
                    j = int(split[2])
                    metric = float(split[3])
                    if type_ not in constraints_:
                        constraints_[type_] = []
                    if type_ == "theta":
                        constraints_["theta"].append((i - 1, j - 1, utils.mtx2bins(metric, -np.pi, np.pi, 25)))
                    elif type_ == "phi":
                        constraints_["phi"].append((i - 1, j - 1, utils.mtx2bins(metric, 0.0, np.pi, 13)))
                    elif type_ == "dist":
                        constraints_["dist"].append((i - 1, j - 1, utils.mtx2bins(metric, 2.0, 20.0, 37)))
                    elif type_ == "omega":
                        constraints_["omega"].append((i - 1, j - 1, utils.mtx2bins(metric, -np.pi, np.pi, 25)))
            for type_ in constraints_:
                mask = np.zeros((L, L)).astype(bool)
                cst = []
                for i, j, metric1hot in sorted(constraints_[type_], key=lambda item_: (item_[0], item_[1])):
                    mask[i][j] = True
                    cst.append(metric1hot)
                constraints[type_] = mask, np.array(cst)
            return constraints
        else:
            raise Exception(f"{cstfile} does not exist!")
