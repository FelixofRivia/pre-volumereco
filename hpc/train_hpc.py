import h5py
import numpy as np
import tensorflow as tf 
from tensorflow.keras.saving import register_keras_serializable
import optuna
from optuna.integration import TFKerasPruningCallback
from sklearn.model_selection import train_test_split
import json
import os
import argparse

def normalize_events(events):
    events_max = events.max(axis=1, keepdims=True)  # max per event
    return events / events_max

def normalize_truths(truths):
    truths_max = truths.max(axis=(1, 2, 3), keepdims=True)
    return truths / truths_max

@register_keras_serializable()
class MaxNormalize1D(tf.keras.layers.Layer):
    def call(self, inputs):
        max_val = tf.reduce_max(inputs, axis=1, keepdims=True)
        return inputs / (max_val + 1e-6)
    
# Objective function for OPTUNA
def objective(trial):
    global best_model, best_history, best_val_loss, best_trial_number

    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"])
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 1, 4)

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))

    for i in range(num_layers):
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dropout(dropout_rate))

    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(np.prod(output_shape), activation='relu'))
    model.add(MaxNormalize1D())
    model.add(tf.keras.layers.Reshape(output_shape))

    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=batch_size,
        verbose=0,
        callbacks=[TFKerasPruningCallback(trial, "val_loss")]
    )

    val_loss = min(history.history["val_loss"])

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        best_history = history.history
        best_trial_number = trial.number

    return val_loss

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Dataset (.h5)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="model", help="Model name")
    args = parser.parse_args()

    with h5py.File(args.input_file, 'r') as f:
        events = f['inputs'][:]
        truths = f['targets'][:]

    X = normalize_events(events)
    y = normalize_truths(truths)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

    input_dim = len(X[0])
    output_shape = y[0].shape

    # Global tracking variables
    best_model = None
    best_history = None
    best_val_loss = float("inf")
    best_trial_number = None

    # Run OPTUNA study
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=3)

    # Best trial
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # === SAVE BEST MODEL, HISTORY, PARAMS ===
    os.makedirs(args.output_dir, exist_ok=True)

    # Save model
    best_model.save(os.path.join(args.output_dir, f"{args.model_name}.keras"))

    # Save training history
    with open(os.path.join(args.output_dir, f"{args.model_name}_history.json"), "w") as f:
        json.dump(best_history, f)

    # Save best parameters
    with open(os.path.join(args.output_dir, f"{args.model_name}_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

    print(f"\nBest model, history, and parameters saved in {args.output_dir}/")