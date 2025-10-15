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


def set_axes_equal(ax):
    """Ensure equal voxel scaling in 3D (1x1x1 cubes)."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    ax.set_box_aspect((x_limits[1] - x_limits[0],
                       y_limits[1] - y_limits[0],
                       z_limits[1] - z_limits[0]))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_file", type=str, help="Dataset (.h5)")
    parser.add_argument("--model", type=str, help="Trained model (.keras)")
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--n_frames", type=int, default=10, help="Number of frames in GIF")
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
    n_samples = min(args.n_frames, len(X))

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

        # --- Update line plot ---
        line.set_data(np.arange(len(event_features)), event_features)
        ax1.set_title(f'Camera times (features) - Event {sample_idx}')

        # --- Clear and redraw 3D plot ---
        ax2.cla()

        # Voxel data
        normed = truth_sample / truth_sample.max()
        filled = normed > 0.01
        colors = cmap(norm(normed))
        ax2.voxels(filled, facecolors=colors, edgecolor='k', linewidth=0.3)

        # --- Restore consistent axes ---
        x_size, y_size, z_size = y.shape[1], y.shape[2], y.shape[3]
        ax2.set_xlim([0, x_size])
        ax2.set_ylim([0, y_size])
        ax2.set_zlim([0, z_size])

        set_axes_equal(ax2)  # Ensures cubic aspect ratio

        # --- Restore labels and ticks ---
        ax2.set_xlabel('z')
        ax2.set_ylabel('x')
        ax2.set_zlabel('y')
        ax2.set_xticks(np.arange(0, x_size + 1, 1))
        ax2.set_yticks(np.arange(0, y_size + 1, 1))
        ax2.set_zticks(np.arange(0, z_size + 1, 1))
        ax2.set_title(f'Voxel score (truths) - Event {sample_idx}')

        ax2.view_init(elev=20, azim=-60)

        return [ax1, ax2]


    ani = animation.FuncAnimation(fig, update, frames=n_samples, interval=1500, blit=False)
    ani.save(os.path.join(args.output_dir, "features_vs_truth.gif"), writer="pillow", fps=1)
    plt.close()

    # Load model
    model = load_model(
        args.model,
        custom_objects={"MaxNormalize1D": MaxNormalize1D},
        compile=True,
    )

    X_vis = X[:args.n_frames]
    y_true = y[:args.n_frames]
    y_pred = model.predict(X_vis)

    fig = plt.figure(figsize=(12, 6))
    ax_pred = fig.add_subplot(1, 2, 1, projection='3d')
    ax_true = fig.add_subplot(1, 2, 2, projection='3d')

    cmap = plt.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=[ax_pred, ax_true], orientation='horizontal', fraction=0.05, pad=0.1).set_label('Normalized Value')

    def update_pred(frame):
        ax_pred.cla()
        ax_true.cla()

        pred = y_pred[frame]
        true = y_true[frame]

        # Only voxels with value > 0.01
        pred_filled = pred > 0.01
        true_filled = true > 0.01

        # Color mapping
        pred_colors = cmap(norm(pred))
        true_colors = cmap(norm(true))

        # Voxel coordinates
        x_size, y_size, z_size = pred.shape[0], pred.shape[1], pred.shape[2]
        x = np.arange(x_size + 1)
        y_coords = np.arange(y_size + 1)
        z = np.arange(z_size + 1)

        # Plot voxels with explicit coordinates
        ax_pred.voxels(pred_filled, facecolors=pred_colors, edgecolor='k', linewidth=0.2)
        ax_true.voxels(true_filled, facecolors=true_colors, edgecolor='k', linewidth=0.2)


        for ax, title in zip([ax_pred, ax_true], ['Prediction', 'Truth']):
            # Set axis limits starting from 0
            ax.set_xlim([0, x_size])
            ax.set_ylim([0, y_size])
            ax.set_zlim([0, z_size])

            # Keep voxels cubic
            set_axes_equal(ax)

            # Labels
            ax.set_xlabel('z')
            ax.set_ylabel('x')
            ax.set_zlabel('y')
            ax.set_title(f'{title} - Event {frame}')

            # Set tick labels to match voxel indices
            ax.set_xticks(np.arange(0, x_size + 1, 1))
            ax.set_yticks(np.arange(0, y_size + 1, 1))
            ax.set_zticks(np.arange(0, z_size + 1, 1))

            # Camera angle
            ax.view_init(elev=20, azim=-60)

        return [ax_pred, ax_true]

    ani = animation.FuncAnimation(fig, update_pred, frames=len(X_vis), interval=1500, blit=False)
    ani.save(os.path.join(args.output_dir, "prediction_vs_truth.gif"), writer="pillow", fps=1)
    plt.close()

        # --- Combined GIF: truth | data | prediction ---
    fig = plt.figure(figsize=(18, 6))
    ax_data = fig.add_subplot(1, 3, 1)
    ax_truth = fig.add_subplot(1, 3, 2, projection='3d')
    ax_pred = fig.add_subplot(1, 3, 3, projection='3d')

    cmap = plt.get_cmap('magma')
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(mappable, ax=[ax_truth, ax_pred], shrink=0.6, pad=0.1).set_label('Normalized Value')

    line_data, = ax_data.plot([], [], 'o-', color='tab:blue')
    ax_data.set_xlim(0, X.shape[1])
    ax_data.set_ylim(0, 1)
    ax_data.set_xlabel('Camera index')
    ax_data.set_ylabel('Normalized time')
    ax_data.set_title('Camera times (data)')
    ax_data.grid(True)

    def update_all(frame):
        ax_truth.cla()
        ax_pred.cla()

        # --- Data plot ---
        event_features = X[frame]
        line_data.set_data(np.arange(len(event_features)), event_features)
        ax_data.set_title(f'Data - Event {frame}')

        # --- 3D Truth and Prediction plots ---
        true = y_true[frame]
        pred = y_pred[frame]

        true_filled = true > 0.01
        pred_filled = pred > 0.01
        true_colors = cmap(norm(true))
        pred_colors = cmap(norm(pred))

        ax_truth.voxels(true_filled, facecolors=true_colors, edgecolor='k', linewidth=0.2)
        ax_pred.voxels(pred_filled, facecolors=pred_colors, edgecolor='k', linewidth=0.2)

        for ax, title in zip([ax_truth, ax_pred], ['Truth', 'Prediction']):
            ax.set_xlim([0, y.shape[1]])
            ax.set_ylim([0, y.shape[2]])
            ax.set_zlim([0, y.shape[3]])
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            ax.set_title(f'{title} - Event {frame}')

        return [line_data, ax_truth, ax_pred]

    ani = animation.FuncAnimation(fig, update_all, frames=len(X_vis), interval=1500, blit=False)
    ani.save(os.path.join(args.output_dir, "truth_data_prediction.gif"), writer="pillow", fps=1)
    plt.close()
