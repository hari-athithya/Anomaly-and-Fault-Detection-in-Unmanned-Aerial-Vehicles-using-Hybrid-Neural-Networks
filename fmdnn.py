"""The three classifiers compared in the FMDNN paper."""
from __future__ import annotations
import tensorflow as tf


def _compile(model: tf.keras.Model, learning_rate: float) -> tf.keras.Model:
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def build_mlp(n_features: int, n_classes: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(n_features,), name="sensor_vector")
    x = tf.keras.layers.Dense(128, activation="relu", name="mlp_dense_1")(inp)
    x = tf.keras.layers.Dropout(0.20, name="mlp_dropout")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="mlp_features")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax_classifier")(x)
    return _compile(tf.keras.Model(inp, out, name="MLP"), learning_rate)


def build_cnn1d(n_features: int, n_classes: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    inp = tf.keras.Input(shape=(n_features,), name="sensor_vector")
    x = tf.keras.layers.Reshape((n_features, 1), name="cnn_reshape")(inp)
    x = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same", name="cnn_conv")(x)
    x = tf.keras.layers.MaxPooling1D(2, name="cnn_pool")(x)
    x = tf.keras.layers.Flatten(name="cnn_flatten")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="cnn_features")(x)
    x = tf.keras.layers.Dropout(0.20, name="cnn_dropout")(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax_classifier")(x)
    return _compile(tf.keras.Model(inp, out, name="CNN1D"), learning_rate)


def build_fmdnn(n_features: int, n_classes: int, learning_rate: float = 1e-3) -> tf.keras.Model:
    """Parallel MLP/CNN intermediate features fused before classification.

    Both branches receive the same processed raw or IFHS-fused input vector;
    predictions are never concatenated.
    """
    inp = tf.keras.Input(shape=(n_features,), name="fused_sensor_vector")
    mlp = tf.keras.layers.Dense(128, activation="relu", name="mlp_dense")(inp)
    mlp = tf.keras.layers.Dropout(0.20, name="mlp_dropout")(mlp)
    mlp_features = tf.keras.layers.Dense(64, activation="relu", name="mlp_features")(mlp)
    cnn = tf.keras.layers.Reshape((n_features, 1), name="cnn_reshape")(inp)
    cnn = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same", name="cnn_conv")(cnn)
    cnn = tf.keras.layers.MaxPooling1D(2, name="cnn_pool")(cnn)
    cnn = tf.keras.layers.Flatten(name="cnn_flatten")(cnn)
    cnn_features = tf.keras.layers.Dense(64, activation="relu", name="cnn_features")(cnn)
    fused = tf.keras.layers.Concatenate(name="feature_fusion")([mlp_features, cnn_features])
    fused = tf.keras.layers.Dense(128, activation="relu", name="fusion_dense")(fused)
    fused = tf.keras.layers.Dropout(0.20, name="fusion_dropout")(fused)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="softmax_classifier")(fused)
    return _compile(tf.keras.Model(inp, out, name="FMDNN"), learning_rate)


BUILDERS = {"mlp": build_mlp, "cnn": build_cnn1d, "fmdnn": build_fmdnn}
