import argparse
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("input_history", type=str, help="Training history (.json)")
    parser.add_argument("--output_file", type=str, default="training_history.png", help="Output image file")
    args = parser.parse_args()

    # Load training history
    with open(args.input_history, "r") as f:
        history = json.load(f)

    # Create subplots: 1 row, 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- Plot Loss ----
    axes[0].plot(history["loss"], label="Training Loss")
    axes[0].plot(history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training vs Validation Loss")
    axes[0].legend()
    axes[0].grid(True)

    # ---- Plot MAE (if available) ----
    if "mae" in history and "val_mae" in history:
        axes[1].plot(history["mae"], label="Training MAE")
        axes[1].plot(history["val_mae"], label="Validation MAE")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Absolute Error")
        axes[1].set_title("Training vs Validation MAE")
        axes[1].legend()
        axes[1].grid(True)
    else:
        axes[1].axis("off")  # hide if MAE not present

    plt.tight_layout()
    plt.savefig(args.output_file, dpi=300)
    print(f"Saved training history plots to {args.output_file}")
