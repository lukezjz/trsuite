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
