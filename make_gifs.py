import argparse
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.saving import register_keras_serializable
import os

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Dataset (.h5)")
    parser.add_argument("--model", type=str, help="Trained model (.keras)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset
    with h5py.File(args.input_file, 'r') as f:
        events = f['inputs'][:]
        truths = f['targets'][:]

    X = normalize_events(events)
    y = normalize_truths(truths)

    # Plot histograms
    plt.figure(figsize=(14, 5))

    # Histogram of normalized event values (left)
    plt.subplot(1, 2, 1)
    plt.hist(X.flatten(), bins=50, color='salmon', edgecolor='black', log=True)
    plt.xlabel('Normalized camera times')
    plt.ylabel('Frequency')
    plt.title('Normalized camera times (features)')
    plt.grid(True)

    # Histogram of normalized truth values (right)
    plt.subplot(1, 2, 2)
    plt.hist(y.flatten(), bins=50, color='skyblue', edgecolor='black', log=True)
    plt.xlabel('Normalized voxel score')
    plt.ylabel('Frequency')
    plt.title('Normalized voxel score (truths)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "histograms.png"))
    plt.close()

    # Make animation with input and expected output
    n_samples = min(10, len(X))

    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    line, = ax1.plot([], [], marker='o')
    ax1.set_ylim(0, 1)
    ax1.set_xlim(0, X.shape[1])
    ax1.set_xlabel('Camera index')
    ax1.set_ylabel('Normalized time')
    ax1.grid(True)

    cmap = plt.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mappable = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    fig.colorbar(mappable, ax=ax2, shrink=0.6, pad=0.1).set_label('Normalized score')

    def update(frame):
        sample_idx = frame
        truth_sample = y[sample_idx]
        event_features = X[sample_idx]

        line.set_data(np.arange(len(event_features)), event_features)
        ax1.set_title(f'Camera times (features) - Event {sample_idx}')

        ax2.cla()
        normed = truth_sample / truth_sample.max()
        filled = normed > 0.01
        colors = cmap(norm(normed))
        ax2.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.3)

        ax2.set_xlim([0, y.shape[1]])
        ax2.set_ylim([0, y.shape[2]])
        ax2.set_zlim([0, y.shape[3]])
        ax2.set_xlabel('voxel_ID_x')
        ax2.set_ylabel('voxel_ID_y')
        ax2.set_zlabel('voxel_ID_z')
        ax2.set_title(f'Voxel score (truths) - Event {sample_idx}')

        return [line]

    ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=1500, blit=False)
    ani.save(os.path.join(args.output_dir, "features_vs_truth.gif"), writer="pillow", fps=1)
    plt.close()

    # Load model
    model = load_model(
        args.model,
        custom_objects={"MaxNormalize1D": MaxNormalize1D},
        compile=True,
    )

    X_vis = X[:10]
    y_true = y[:10]
    y_pred = model.predict(X_vis)

    fig = plt.figure(figsize=(12, 6))
    ax_pred = fig.add_subplot(1, 2, 1, projection='3d')
    ax_true = fig.add_subplot(1, 2, 2, projection='3d')

    cmap = plt.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=[ax_pred, ax_true], shrink=0.6, pad=0.1).set_label('Normalized Value')

    def update_pred(frame):
        ax_pred.cla()
        ax_true.cla()

        pred = y_pred[frame]
        true = y_true[frame]

        pred_filled = pred > 0.01
        true_filled = true > 0.01

        pred_colors = cmap(norm(pred))
        true_colors = cmap(norm(true))

        ax_pred.voxels(pred_filled, facecolors=pred_colors, edgecolor='k', linewidth=0.2)
        ax_true.voxels(true_filled, facecolors=true_colors, edgecolor='k', linewidth=0.2)

        for ax, title in zip([ax_pred, ax_true], ['Prediction', 'Truth']):
            ax.set_xlim([0, y.shape[1]])
            ax.set_ylim([0, y.shape[2]])
            ax.set_zlim([0, y.shape[3]])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'{title} - Event {frame}')

        return [ax_pred, ax_true]

    ani = animation.FuncAnimation(fig, update_pred, frames=len(X_vis), interval=1500, blit=False)
    ani.save(os.path.join(args.output_dir, "prediction_vs_truth.gif"), writer="pillow", fps=1)
    plt.close()