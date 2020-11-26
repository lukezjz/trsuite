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
    parser.add_argument("-a=", "--aln=", type=str, required=False, dest="aln", default=None, help="path to alignment file")
    parser.add_argument("--list=", type=str, required=False, dest="list", default=None, help="path to npz file")
    parser.add_argument("--folder", "--folder=", type=str, required=False, dest="folder", default="./", help="npz file folder")
    parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory",
                        default="../models/trmodel", help="path to trRosetta network weights")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    assert (args.aln and not args.list) or (args.list and not args.aln)

    aln_list = []
    if args.aln:
        aln_list.append(args.aln)
    else:
        with open(args.list) as fr:
            for aln in fr.readlines():
                aln_list.append(aln.strip())

    if not os.path.exists(args.folder):
        os.mkdir(args.folder)

    start_time = time.time()
    get_features = networks.GetFeatures(args.trmodel_directory, len(utils.parse_aln(aln_list[0])))
    middle_time1 = time.time()
    print(f"Model loading takes {round(middle_time1 - start_time)}s.")

    for aln in aln_list:
        msa_aa = utils.parse_aln(aln)
        msa_idx = utils.msa_aa2idx(msa_aa)
        print(msa_aa[0])
        middle_time2 = time.time()
        features = get_features.predict_(msa_idx)
        end_time = time.time()
        print(f"Prediction takes {round(end_time - middle_time2)}s.")

        npz = aln.split("/")[-1].rstrip(".fasta").rstrip(".aln") + ".npz"
        np.savez_compressed(os.path.join(args.folder, npz), theta=features["theta"], phi=features["phi"], dist=features["dist"], omega=features["omega"])

    print("done.")


if __name__ == "__main__":
    main()
