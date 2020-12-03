import tensorflow as tf
from lib import utils


def background_loss(background_features, features, weights):
    bt, bp, bd, bo = background_features["theta"], background_features["phi"], background_features["dist"], background_features["omega"]
    pt, pp, pd, po = features["theta"], features["phi"], features["dist"], features["omega"]
    theta_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pt * tf.math.log(pt / bt + 1e-9), axis=-1) * weights["theta"])
    phi_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pp * tf.math.log(pp / bp + 1e-9), axis=-1) * weights["phi"])
    dist_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(pd * tf.math.log(pd / bd + 1e-9), axis=-1) * weights["dist"])
    omega_bkgrd_loss = -tf.math.reduce_mean(tf.math.reduce_sum(po * tf.math.log(po / bo + 1e-9), axis=-1) * weights["omega"])
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


def loop_loss(features, weights):
    return tf.math.reduce_mean(tf.math.reduce_sum(features["dist"] * tf.math.log(features["dist"] + 1e-9), axis=-1) * weights)   # no negative symbol


def domain_loss(features_A, features_B, left, right, features):
    return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(features_A["dist"], features["dist"][:left - 1, :left - 1])) + \
           tf.reduce_mean(tf.keras.losses.categorical_crossentropy(features_B["dist"], features["dist"][right:, right:]))
