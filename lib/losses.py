import tensorflow as tf
from lib import utils


def background_loss(background_features, predicted_features, mask=None):
    bt, bp, bd, bo = background_features["p_theta"], background_features["p_phi"], background_features["p_dist"], background_features["p_omega"]
    pt, pp, pd, po = predicted_features["p_theta"], predicted_features["p_phi"], predicted_features["p_dist"], predicted_features["p_omega"]
    if mask is None:
        length = bp.shape[0]
        mask = tf.cast(tf.ones((length, length)), dtype=tf.bool)
    theta_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pt * tf.math.log(pt / bt), axis=-1)[mask])
    phi_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pp * tf.math.log(pp / bp), axis=-1)[mask])
    dist_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pd * tf.math.log(pd / bd), axis=-1)[mask])
    omega_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(po * tf.math.log(po / bo), axis=-1)[mask])
    bkgrd_loss = phi_bkgrd_loss + theta_bkgrd_loss + dist_bkgrd_loss + omega_bkgrd_loss
    return bkgrd_loss


def aa_loss(msa):   # ?
    msa1hot = tf.one_hot(msa, 21, dtype=tf.float32)[:, :20]
    length_ = msa1hot.shape[1]
    aa_samp = tf.reduce_mean(msa1hot[0, :, :20], axis=0) / tf.cast(length_, dtype=tf.float32) + 1e-7
    aa_samp = aa_samp / tf.reduce_sum(aa_samp)
    loss_aa = tf.reduce_sum(aa_samp * tf.math.log(aa_samp / utils.aa_bkgr))
    return loss_aa

# def constraints_loss(predicted_features, constraints)
