import os
import sys
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.disable(logging.WARNING)
import time
import argparse
import numpy as np
from lib import utils, networks, losses, optimizers


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--length=", type=int, required=False, dest="length", default=60, help="sequence length")
    parser.add_argument("-a", "--aln=", type=str, required=False, dest="aln", default="", help="path to starting alignment file")
    parser.add_argument("-r", "--resfile=", type=str, required=False, dest="resfile", default=None, help="path to resfile")
    parser.add_argument("-c", "--cstfile=", type=str, required=False, dest="cstfile", default=None, help="path to constraint file")
    parser.add_argument("-o", "--output", type=str, required=False, dest="output", default=f"output_{time.strftime('%Y%m%d%H%M%S')}.csv", help="path to output file")
    parser.add_argument("-z", "--npz", type=str, required=False, dest="npz", default=None, help="path to npz file")
    parser.add_argument("--trmodel=", type=str, required=False, dest="trmodel_directory", default="../models/trmodel", help="path to trRosetta network weights")
    parser.add_argument("--background=", type=str, required=False, dest="background_directory", default="../models/bkgrd", help="path to background network weights")
    parser.add_argument("--bkgrd_loss_weights_file=", type=str, required=False, dest="bkgrd_loss_weights_file", default=None, help="path of background loss weights file")
    parser.add_argument("--cst_bias=", type=float, required=False, dest="cst_bias", default=0.2, help="bias of the constraints loss term")
    parser.add_argument("--aa_weight=", type=float, required=False, dest="aa_weight", default=0.0, help="weight for the aa composition biasing loss term")
    parser.add_argument("--schedule=", type=str, required=False, dest='schedule',
                        default="0.1,20000,2.0,5000", help="simulated annealing schedule: 'T0,n_steps,decrease_factor,decrease_range'")
    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    ### aln: msa_aa, msa_idx, L
    ### length: L
    msa_aa = None
    msa_idx = None
    if args.aln:
        msa_aa = utils.parse_aln(args.aln)
        msa_idx = utils.msa_aa2idx(msa_aa)
        L = msa_idx.shape[1]
    else:
        L = args.length

    ### aa_probabilities_L
    aa_probabilities_L = np.tile(utils.aa_probabilities, (L, 1))
    if args.resfile:
        aa_allowed = utils.parse_resfile(args.resfile, L)
        aa_probabilities_L *= aa_allowed
        aa_probabilities_L /= np.sum(aa_probabilities_L, axis=1).reshape((-1, 1))

    ### length: msa_idx, msa_aa
    if not args.aln:
        aa_idx = np.arange(20)
        msa_idx = np.array([[np.random.choice(aa_idx, p=aa_probabilities_pos) for aa_probabilities_pos in aa_probabilities_L]])
        msa_aa = utils.msa_idx2aa(msa_idx)

    print(f"starting sequence: {msa_aa[0]}")

    ### model
    get_features = networks.GetFeatures(args.trmodel_directory, L)

    ### loss_function
    bkgrd = networks.GetBackground(args.background_directory, L).generate()
    if args.bkgrd_loss_weights_file:
        bkgrd_loss_weights = {"theta": np.full((L, L), 0.0), "phi": np.full((L, L), 0.0), "dist": np.full((L, L), 0.0), "omega": np.full((L, L), 0.0)}
        for type_, weights in utils.parse_bkgrd_loss_weights_file(args.bkgrd_loss_weights_file, L).items():
            bkgrd_loss_weights[type_] = weights
    else:
        bkgrd_loss_weights = {"theta": np.full((L, L), 1.0), "phi": np.full((L, L), 1.0), "dist": np.full((L, L), 1.0), "omega": np.full((L, L), 1.0)}
    
    if args.cstfile:
        constraints = utils.parse_cstfile(args.cstfile, L)

    def loss_function(msa):
        features = get_features.predict(msa)
        background_loss = losses.background_loss(bkgrd, features, bkgrd_loss_weights)
        aa_loss = losses.aa_loss(msa)
        total_loss = background_loss + args.aa_weight * aa_loss
        if args.cstfile:
            cst_loss = losses.constraints_loss(constraints, features)
            return {"background_loss": background_loss, "aa_loss": aa_loss, "total_loss": total_loss, "constraints_loss": cst_loss}
        else:
            return {"background_loss": background_loss, "aa_loss": aa_loss, "total_loss": total_loss}

    ### T0, n_steps, decrease_factor, decrease_range
    tmp = args.schedule.split(",")
    schedule = [float(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3])]

    ### optimizer
    optimizer = optimizers.SimulatedAnnealing(schedule, aa_probabilities_L, cst_bias=args.cst_bias)

    ### generate seq
    final_msa_idx = optimizer.iterator(msa_idx, loss_function, args.output)

    print(f"final sequence: {utils.msa_idx2aa(final_msa_idx)[0]}")

    '''
    # print("args: ", args)
    # print("seq: ", seq)
    # print("seq_idx: ", seq_idx)
    # print("L: ", L)
    # print("aa_probabilities_L.shape: ", aa_probabilities_L.shape)
    # print("bkgrd['p_theta'].shape: ", bkgrd["p_theta"].shape)
    # print("bkgrd['p_phi'].shape: ", bkgrd["p_phi"].shape)
    # print("bkgrd['p_dist'].shape: ", bkgrd["p_dist"].shape)
    # print("bkgrd['p_omega'].shape: ", bkgrd["p_omega"].shape)
    # print("loss: ", loss_function(seq_msa))
    # print(schedule)
    '''


if __name__ == '__main__':
    main()
