import tensorflow as tf
from lib import utils


def background_loss(background_features, features):   # , mask=None):
    bt, bp, bd, bo = background_features["theta"], background_features["phi"], background_features["dist"], background_features["omega"]
    pt, pp, pd, po = features["theta"], features["phi"], features["dist"], features["omega"]
    # if mask is None:
    #     length = bp.shape[0]
    #     mask = tf.cast(tf.ones((length, length)), dtype=tf.bool)
    theta_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pt * tf.math.log(pt / bt), axis=-1))   # [mask])
    phi_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pp * tf.math.log(pp / bp), axis=-1))   # [mask])
    dist_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pd * tf.math.log(pd / bd), axis=-1))   # [mask])
    omega_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(po * tf.math.log(po / bo), axis=-1))   # [mask])
    bkgrd_loss = phi_bkgrd_loss + theta_bkgrd_loss + dist_bkgrd_loss + omega_bkgrd_loss
    return bkgrd_loss


def aa_loss(msa):
    msa1hot = tf.one_hot(msa, 21, dtype=tf.float32)[:, :20]
    length_ = msa1hot.shape[1]
    aa_samp = tf.reduce_mean(msa1hot[0, :, :20], axis=0) / tf.cast(length_, dtype=tf.float32) + 1e-7
    aa_samp = aa_samp / tf.reduce_sum(aa_samp)
    aa_loss_ = tf.reduce_sum(aa_samp * tf.math.log(aa_samp / utils.aa_bkgr))
    return aa_loss_


def constraints_loss(constraints, features):
    cst_loss = 0
    for type_ in constraints:
        mask, cst = constraints[type_]
        cst_loss += tf.reduce_mean(tf.keras.losses.categorical_crossentropy(cst, features[type_][mask]))
    return cst_loss
