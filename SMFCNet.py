import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import tensorflow as tf


def SMFC_Net():
    shape_list = []
    for i in range(10):
        shape_list.append(tf.keras.Input(shape=(120, 120, 1)))
    h_list = []
    for j in range(10):
        h = shape_list[j]
        h = tf.keras.layers.Conv2D(kernel_size=[1, 120], filters=64, padding='valid',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))(h)
        h = tf.keras.layers.Conv2D(kernel_size=[120, 1], filters=32, padding='valid',
                                   kernel_initializer='he_normal',
                                   kernel_regularizer=tf.keras.regularizers.l2(0.00001))(h)
        h = tf.keras.layers.Reshape([1, 32])(h)
        h_list.append(h)
    megred = tf.keras.layers.concatenate(h_list, axis=1)
    megred_flatten = tf.keras.layers.Flatten()(megred)
    drop1 = tf.keras.layers.Dropout(0.2)(megred_flatten)
    mlp1 = tf.keras.layers.Dense(128)(drop1)
    drop2 = tf.keras.layers.Dropout(0.2)(mlp1)
    mlp2 = tf.keras.layers.Dense(64)(drop2)
    out = tf.keras.layers.Dense(2, activation='softmax')(mlp2)
    model = tf.keras.Model(inputs=shape_list, outputs=out)
    return model