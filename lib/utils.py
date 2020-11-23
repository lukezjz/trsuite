import os
import numpy as np
import tensorflow as tf


# database
aa_probabilities = np.array([[0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
                              0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]])
aa_bkgr = tf.constant(np.array([0.07892653, 0.04979037, 0.0451488, 0.0603382, 0.01261332,
                                0.03783883, 0.06592534, 0.07122109, 0.02324815, 0.05647807,
                                0.09311339, 0.05980368, 0.02072943, 0.04145316, 0.04631926,
                                0.06123779, 0.0547427, 0.01489194, 0.03705282, 0.0691271]), dtype=tf.float32)

aa2idx = {"A": 0, "R": 1, "N": 2, "D": 3, "C": 4, "Q": 5, "E": 6, "G": 7, "H": 8, "I": 9, "L": 10,
          "K": 11, "M": 12, "F": 13, "P": 14, "S": 15, "T": 16, "W": 17, "Y": 18, "V": 19, "-": 20}
idx2aa = ["A", "R", "N", "D", "C", "Q", "E", "G", "H", "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V", "-"]
aa1_aa3 = {"A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
           "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
           "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
           "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}


def seq_aa2idx(seq_aa):
    return np.array([aa2idx[aa] for aa in seq_aa])


def seq_idx2aa(seq_idx):
    return "".join([idx2aa[idx] for idx in seq_idx])


def msa_aa2idx(msa_aa):
    return np.array([seq_aa2idx(seq_aa) for seq_aa in msa_aa])


def msa_idx2aa(msa_idx):
    return [seq_idx2aa(seq_idx) for seq_idx in msa_idx]


def mtx2bins(x_ref, start, end, nbins, mask=None):
    bins = np.linspace(start, end, nbins)
    x_true = np.digitize(x_ref, bins).astype(np.uint8)
    if mask:
        x_true[mask] = 0
    return np.eye(nbins + 1)[x_true][..., :-1]


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


def parse_resfile(resfile, L):
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
                        aa_allowed[pdbnum-1][aa2idx[aa]] = 0
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
                        constraints_["theta"].append((i - 1, j - 1, mtx2bins(metric, -np.pi, np.pi, 25)))
                    elif type_ == "phi":
                        constraints_["phi"].append((i - 1, j - 1, mtx2bins(metric, 0.0, np.pi, 13)))
                    elif type_ == "dist":
                        constraints_["dist"].append((i - 1, j - 1, mtx2bins(metric, 2.0, 20.0, 37)))
                    elif type_ == "omega":
                        constraints_["omega"].append((i - 1, j - 1, mtx2bins(metric, -np.pi, np.pi, 25)))
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
