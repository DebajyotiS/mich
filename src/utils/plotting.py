import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

LAYER_NAMES = ["Deep", "Middle", "Superficial"]


def plot_neural_bold_layers(
    pred_bold: torch.Tensor,
    true_bold: torch.Tensor,
    pred_neural: torch.Tensor,
    true_neural: torch.Tensor,
    tr: float = 0.1,
) -> Figure:
    """
    Plot predicted and true BOLD and neural activity for a single sample.

    Args:
        pred_bold:    [L, T]
        true_bold:    [L, T]
        pred_neural:  [L, T]
        true_neural:  [L, T]
        tr:           repetition time in seconds (default 0.1s)
    """
    n_layers = pred_bold.shape[0]
    times = np.arange(pred_bold.shape[-1]) * tr

    pred_bold_np = pred_bold.detach().cpu().numpy()
    true_bold_np = true_bold.detach().cpu().numpy()
    pred_neural_np = pred_neural.detach().cpu().numpy()
    true_neural_np = true_neural.detach().cpu().numpy()

    fig, axes = plt.subplots(nrows=n_layers, figsize=(10, 4 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for i in range(n_layers):
        ax_bold = axes[i]
        ax_neural = ax_bold.twinx()

        # BOLD on left axis
        ax_bold.plot(times, true_bold_np[i], color="orange", alpha=0.9, label="True BOLD")
        ax_bold.plot(
            times,
            pred_bold_np[i],
            color="orange",
            alpha=0.7,
            label="Predicted BOLD",
            linestyle="--",
        )

        # Neural on right axis
        ax_neural.plot(times, true_neural_np[i], color="blue", alpha=0.9, label="True Neural")
        ax_neural.plot(
            times,
            pred_neural_np[i],
            color="blue",
            alpha=0.7,
            label="Predicted Neural",
            linestyle="--",
        )

        ax_bold.set_title(LAYER_NAMES[i], fontfamily="monospace")
        ax_bold.set_xlabel("Time (s)", fontfamily="monospace")
        ax_bold.set_ylabel("BOLD Signal", fontfamily="monospace")
        ax_neural.set_ylabel("Neural Activity", fontfamily="monospace")

        ax_bold.legend(loc="upper left")
        ax_neural.legend(loc="upper right")

        # Set monospace font on tick labels without overriding tick positions.
        # ax_neural manages its own ticks independently from ax_bold.
        for ax in (ax_bold, ax_neural):
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("monospace")

    return fig


SIGNAL_NAMES = ["s", "f", "v", "q", "v*", "q*"]


def plot_latent_layers(
    pred_s: torch.Tensor,
    pred_f: torch.Tensor,
    pred_v: torch.Tensor,
    pred_q: torch.Tensor,
    pred_v_star: torch.Tensor,
    pred_q_star: torch.Tensor,
    tr: float = 0.1,
    title: str = "Latent States",
) -> Figure:
    """
    Plot all 7 Heinzle latent signals across 3 cortical layers for a single sample.

    Each input tensor is [L, T].
    Layout: rows = layers (Deep → Superficial), cols = signals (x, s, f, v, q, v*, q*).
    """
    signals = [
        pred_s.detach().cpu().numpy(),
        pred_f.detach().cpu().numpy(),
        pred_v.detach().cpu().numpy(),
        pred_q.detach().cpu().numpy(),
        pred_v_star.detach().cpu().numpy(),
        pred_q_star.detach().cpu().numpy(),
    ]

    n_layers = signals[0].shape[0]
    n_signals = len(signals)
    times = np.arange(signals[0].shape[-1]) * tr

    fig, axes = plt.subplots(
        nrows=n_layers,
        ncols=n_signals,
        figsize=(3 * n_signals, 3 * n_layers),
        constrained_layout=True,
    )
    fig.suptitle(title, fontfamily="monospace", fontsize=13)

    for row, layer_name in enumerate(LAYER_NAMES):
        for col, (sig_array, sig_name) in enumerate(zip(signals, SIGNAL_NAMES, strict=True)):
            ax = axes[row, col]
            ax.plot(times, sig_array[row], color="steelblue", linewidth=0.9)

            # column header on top row only
            if row == 0:
                ax.set_title(sig_name, fontfamily="monospace", fontsize=11)

            # layer label on leftmost column only
            if col == 0:
                ax.set_ylabel(layer_name, fontfamily="monospace")

            # time axis label on bottom row only
            if row == n_layers - 1:
                ax.set_xlabel("Time (s)", fontfamily="monospace", fontsize=9)

            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("monospace")
                label.set_fontsize(8)

    return fig
