import os
import sys
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.disable(logging.WARNING)
import argparse
from lib import arguments, utils, networks
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-a=", "--aln=", type=str, required=False, dest="aln", default="", help="path to alignment file")
parser.add_argument("-z=", "--npz=", type=str, required=False, dest="npz", default=None, help="path to npz file")
parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory",
                    default="../models/trmodel", help="path to trRosetta network weights")
args = parser.parse_args()

msa_aa = arguments.parse_aln(args.aln)
msa_idx = utils.msa_aa2idx(msa_aa)
print(msa_aa[0])

get_features = networks.GetFeatures(args.trmodel_directory, len(msa_idx))
features = get_features.predict(msa_idx)

np.savez_compressed(args.npz, theta=features["p_theta"], phi=features["p_phi"], dist=features["p_dist"], omega=features["p_omega"])
