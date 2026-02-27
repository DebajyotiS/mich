import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch


def plot_layers(
    pred_bold: torch.Tensor,
    true_bold: torch.Tensor,
    pred_neural: torch.Tensor,
    true_neural: torch.Tensor,
) -> Figure:
    """Plots the predicted and true BOLD signals and neural activity for a single sample."""
    fig, ax = plt.subplots(nrows=3, figsize=(8, 12), constrained_layout=True)

    n_layers = pred_neural.shape[0]
    times = np.arange(0, pred_bold.shape[-1]) * 0.1  # Assuming TR=0.1s for x-axis

    pred_bold_np = pred_bold.cpu().numpy()
    true_bold_np = true_bold.cpu().numpy()
    pred_neural_np = pred_neural.cpu().numpy()
    true_neural_np = true_neural.cpu().numpy()
    layer_names = ["Deep", "Middle", "Superficial"]

    for i in range(n_layers):
        ax[n_layers - i - 1].plot(
            times,
            pred_bold_np[i],
            label="Predicted BOLD",
            color="orange",
            linestyle="--",
            alpha=0.7,
        )
        ax[n_layers - i - 1].plot(
            times, true_bold_np[i], label="True BOLD", color="orange", alpha=0.9
        )
        ax2 = ax[n_layers - i - 1].twinx()
        ax2.plot(
            times,
            pred_neural_np[i],
            label="Predicted Neural",
            color="blue",
            linestyle="--",
            alpha=0.7,
        )
        ax2.plot(times, true_neural_np[i], label="True Neural", color="blue", alpha=0.9)
        ax[n_layers - i - 1].set_title(f"{layer_names[i]} Layer", font="monospace")
        ax[n_layers - i - 1].set_xlabel("Time (s)", font="monospace")
        ax[n_layers - i - 1].set_ylabel("BOLD Signal", font="monospace")
        ax2.set_ylabel("Neural Activity", font="monospace")
        ax[n_layers - i - 1].legend(loc="upper left")
        ax2.legend(loc="upper right")

        # set tick labels to monospace font
        x_ticks, x_tick_labels = (
            ax[n_layers - i - 1].get_xticks(),
            ax[n_layers - i - 1].get_xticklabels(),
        )
        y_ticks, y_tick_labels = (
            ax[n_layers - i - 1].get_yticks(),
            ax[n_layers - i - 1].get_yticklabels(),
        )
        ax[n_layers - i - 1].set_xticks(x_ticks)
        ax[n_layers - i - 1].set_xticklabels(x_tick_labels, font="monospace")
        ax[n_layers - i - 1].set_yticks(y_ticks)
        ax[n_layers - i - 1].set_yticklabels(y_tick_labels, font="monospace")
        ax2.set_yticks(y_ticks)
        ax2.set_yticklabels(y_tick_labels, font="monospace")

    # return wand loggable image
    return fig
