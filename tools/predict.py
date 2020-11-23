import os
import sys
import logging
os.environ["TF_CPMIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.disable(logging.WARNING)
import time
import argparse
import numpy as np
from lib import utils, networks


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a=", "--aln=", type=str, required=False, dest="aln", default="", help="path to alignment file")
    parser.add_argument("-z=", "--npz=", type=str, required=False, dest="npz", default=None, help="path to npz file")
    parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory",
                        default="../models/trmodel", help="path to trRosetta network weights")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    msa_aa = utils.parse_aln(args.aln)
    msa_idx = utils.msa_aa2idx(msa_aa)
    print(msa_aa[0])

    start_time = time.time()
    get_features = networks.GetFeatures(args.trmodel_directory, len(msa_idx))
    middle_time = time.time()
    print(f"Model loading takes {round(middle_time - start_time)}s.")
    features = get_features.predict(msa_idx)
    end_time = time.time()
    print(f"Prediction takes {round(end_time - middle_time)}s.")
    np.savez_compressed(args.npz, theta=features["theta"], phi=features["phi"], dist=features["dist"], omega=features["omega"])
    print("done.")


if __name__ == "__main__":
    main()
