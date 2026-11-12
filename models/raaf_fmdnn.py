"""Reliability-Aware Adaptive Feature Fusion FMDNN (RAAF-FMDNN).

The MLP and CNN feature extractors intentionally mirror ``build_fmdnn``.  The
novelty is confined to the fusion head and auxiliary branch supervision.
"""
from __future__ import annotations

import tensorflow as tf


def _branches(n_features: int):
    inp = tf.keras.Input(shape=(n_features,), name="fused_sensor_vector")
    mlp = tf.keras.layers.Dense(128, activation="relu", name="mlp_dense")(inp)
    mlp = tf.keras.layers.Dropout(0.20, name="mlp_dropout")(mlp)
    h_m = tf.keras.layers.Dense(64, activation="relu", name="mlp_features")(mlp)
    cnn = tf.keras.layers.Reshape((n_features, 1), name="cnn_reshape")(inp)
    cnn = tf.keras.layers.Conv1D(16, 3, activation="relu", padding="same", name="cnn_conv")(cnn)
    cnn = tf.keras.layers.MaxPooling1D(2, name="cnn_pool")(cnn)
    cnn = tf.keras.layers.Flatten(name="cnn_flatten")(cnn)
    h_c = tf.keras.layers.Dense(64, activation="relu", name="cnn_features")(cnn)
    return inp, h_m, h_c


def build_raaf_fmdnn(n_features: int, n_classes: int, learning_rate: float = 1e-3,
                     fusion_head: str = "raaf", projection_dim: int = 64) -> tf.keras.Model:
    """Build concat, adaptive, or full RAAF fusion with multitask loss."""
    if fusion_head not in {"concat", "adaptive", "raaf"}:
        raise ValueError("fusion_head must be concat, adaptive, or raaf")
    inp, h_m, h_c = _branches(n_features)
    p_m = tf.keras.layers.Dense(n_classes, activation="softmax", name="aux_m")(h_m)
    p_c = tf.keras.layers.Dense(n_classes, activation="softmax", name="aux_c")(h_c)
    if fusion_head == "concat":
        fused = tf.keras.layers.Concatenate(name="feature_fusion")([h_m, h_c])
        fused = tf.keras.layers.Dense(128, activation="relu", name="fusion_dense")(fused)
        fused = tf.keras.layers.Dropout(0.20, name="fusion_dropout")(fused)
    else:
        eps = tf.keras.backend.epsilon()
        log_classes = tf.math.log(tf.cast(n_classes, tf.float32) + eps)
        u_m = tf.keras.layers.Lambda(lambda p: -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1, keepdims=True) / log_classes, name="entropy_m")(p_m)
        u_c = tf.keras.layers.Lambda(lambda p: -tf.reduce_sum(p * tf.math.log(p + eps), axis=-1, keepdims=True) / log_classes, name="entropy_c")(p_c)
        r_m = tf.keras.layers.Lambda(lambda u: 1.0 - u, name="reliability_m")(u_m)
        r_c = tf.keras.layers.Lambda(lambda u: 1.0 - u, name="reliability_c")(u_c)
        proj_m = tf.keras.layers.Dense(projection_dim, name="proj_m")(h_m)
        proj_c = tf.keras.layers.Dense(projection_dim, name="proj_c")(h_c)
        gate_inputs = tf.keras.layers.Concatenate(name="gate_inputs")([h_m, h_c, r_m, r_c])
        gates = tf.keras.layers.Dense(2, activation="softmax", name="adaptive_gates")(gate_inputs)
        h_adapt = tf.keras.layers.Add(name="adaptive_fusion")([
            tf.keras.layers.Multiply()([tf.keras.layers.Lambda(lambda g: g[:, 0:1])(gates), proj_m]),
            tf.keras.layers.Multiply()([tf.keras.layers.Lambda(lambda g: g[:, 1:2])(gates), proj_c]),
        ])
        if fusion_head == "adaptive":
            fused = h_adapt
        else:
            inter_m = tf.keras.layers.Dense(projection_dim, activation="gelu", name="interaction_m")(h_m)
            inter_c = tf.keras.layers.Dense(projection_dim, activation="gelu", name="interaction_c")(h_c)
            h_inter = tf.keras.layers.Multiply(name="cross_feature_interaction")([inter_m, inter_c])
            h_res = tf.keras.layers.Dense(projection_dim, name="residual_shortcut")(tf.keras.layers.Concatenate()([h_m, h_c]))
            fused = tf.keras.layers.LayerNormalization(name="fusion_layernorm")(tf.keras.layers.Add()([h_adapt, tf.keras.layers.Lambda(lambda x: 0.5 * x)(h_inter), h_res]))
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="output_main")(fused)
    if fusion_head == "concat":
        model = tf.keras.Model(inp, out, name="FMDNN_concat")
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        model.fusion_head = fusion_head
        return model
    model = tf.keras.Model(inp, {"output_main": out, "aux_m": p_m, "aux_c": p_c}, name=f"RAAF_FMDNN_{fusion_head}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss={"output_main": "sparse_categorical_crossentropy", "aux_m": "sparse_categorical_crossentropy", "aux_c": "sparse_categorical_crossentropy"},
        loss_weights={"output_main": 1.0, "aux_m": 0.2, "aux_c": 0.2},
        metrics={"output_main": ["accuracy"]},
    )
    model.fusion_head = fusion_head
    model.multitask_outputs = True
    return model


def main_output(prediction):
    """Extract the primary classifier output from any RAAF/baseline prediction."""
    if isinstance(prediction, dict):
        return prediction["output_main"]
    if isinstance(prediction, (list, tuple)):
        return prediction[0]
    return prediction
