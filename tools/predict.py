import os
import sys
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.disable(logging.WARNING)
from lib import arguments, utils, networks
import numpy as np
import time


def main():
    args = arguments.predict_get_arguments()

    msa_aa = arguments.parse_aln(args.aln)
    msa_idx = utils.msa_aa2idx(msa_aa)
    print(msa_aa[0])

    start_time = time.time()
    get_features = networks.GetFeatures(args.trmodel_directory, len(msa_idx))
    middle_time = time.time()
    print(f"Model loading takes {round(middle_time - start_time)}s.")
    features = get_features.predict(msa_idx)
    end_time = time.time()
    print(f"Prediction takes {round(end_time - middle_time)}s.")
    np.savez_compressed(args.npz, theta=features["p_theta"], phi=features["p_phi"], dist=features["p_dist"], omega=features["p_omega"])
    print("done.")


if __name__ == "__main__":
    main()
