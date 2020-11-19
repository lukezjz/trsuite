import os
import sys
import logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
logging.disable(logging.WARNING)
from lib import arguments, utils, networks, losses, optimizers
import numpy as np


"""
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
environment variable
"""


# python tr_suite/hallucinate.py --background hallucinate2/models/bkgrd --model hallucinate2/models/trmodel
# python tr_suite/hallucinate.py -s hallucinate2/test/A.fasta -r A.resfile --background hallucinate2/models/bkgrd --model hallucinate2/models/trmodel
# python tr_suite/hallucinate.py -l 102 -r s3l100.resfile --background hallucinate2/models/bkgrd --model hallucinate2/models/trmodel
def main():
    args = arguments.hallucinate_get_arguments()

    ### aln: msa_aa, msa_idx, L
    ### length: L
    msa_aa = None
    msa_idx = None
    if args.aln:
        msa_aa = arguments.parse_aln(args.aln)
        msa_idx = utils.msa_aa2idx(msa_aa)
        L = msa_idx.shape[1]
    else:
        L = args.length

    ### aa_probabilities_L
    aa_probabilities_L = np.tile(utils.aa_probabilities, (L, 1))
    if args.resfile:
        aa_allowed = arguments.parse_resfile(args.resfile, L)
        aa_probabilities_L *= aa_allowed
        aa_probabilities_L /= np.sum(aa_probabilities_L, axis=1).reshape((-1, 1))

    ### length: msa_idx, msa_aa
    if not args.aln:
        aa_idx = np.arange(20)
        msa_idx = np.array([[np.random.choice(aa_idx, p=aa_probabilities_pos) for aa_probabilities_pos in aa_probabilities_L]])
        msa_aa = utils.msa_idx2aa(msa_idx)

    print(f"starting sequence: {msa_aa[0]}")

    ### bkgrd
    bkgrd = networks.GetBackground(args.background_directory, L).generate()

    ### model
    get_features = networks.GetFeatures(args.trmodel_directory, L)

    ### loss_function
    def loss_function(msa):
        features = get_features.predict(msa)
        background_loss = losses.background_loss(bkgrd, features)
        aa_loss = losses.aa_loss(msa)
        total_loss = background_loss + args.aa_weight * aa_loss   # + constraints_loss
        if args.cstfile:
            constraints = arguments.parse_cstfile(args.cstfile)
            cst_loss = losses.constraints_loss(constraints, features)
            total_loss += args.cst_weight * cst_loss
            return {"background_loss": background_loss, "aa_loss": aa_loss, "constraints_loss": cst_loss, "total_loss": total_loss}
        else:
            return {"background_loss": background_loss, "aa_loss": aa_loss, "total_loss": total_loss}

    ### T0, n_steps, decrease_factor, decrease_range
    tmp = args.schedule.split(",")
    schedule = [float(tmp[0]), int(tmp[1]), float(tmp[2]), int(tmp[3])]

    ### optimizer
    optimizer = optimizers.SimulatedAnnealing(schedule, aa_probabilities_L)

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
