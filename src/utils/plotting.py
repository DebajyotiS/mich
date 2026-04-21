import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.figure import Figure

LAYER_NAMES = ["Superficial", "Middle", "Deep"]
COLOR_HEX = [
    "#9750a1",
    "#46c19a",
    "#bb4d3e",
    "#b94a73",
    "#6777cf",
    "#af7f3b",
    "#67a54f",
    "#bca73a",
]
SIGNALS_LIST = ["bold", "x", "s", "f", "v", "q", "vstar", "qstar"]
LATENT_NAMES = ["s", "f", "v", "q", "vstar", "qstar"]


def plot_neural_bold_layers(
    pred_bold: torch.Tensor,
    true_bold: torch.Tensor,
    pred_neural: torch.Tensor,
    true_neural: torch.Tensor,
    source_layer: torch.Tensor,
    source_pos: torch.Tensor,
    tr: float = 1.0,
) -> Figure:
    """
    Plot predicted and true BOLD and neural activity for a single sample.

    Args:
        pred_bold:    [L, T]
        true_bold:    [L, T]
        pred_neural:  [L, T]
        true_neural:  [L, T]
        source_layer: [L]
        source_pos:   [L, 2]
        tr:           repetition time in seconds (default 0.1s)
    """
    n_layers = pred_bold.shape[0]
    times = np.arange(pred_bold.shape[-1]) * tr

    pred_bold_np = pred_bold.cpu().numpy()
    true_bold_np = true_bold.cpu().numpy()
    pred_neural_np = pred_neural.cpu().numpy()
    true_neural_np = true_neural.cpu().numpy()
    source_layer = source_layer.cpu().numpy()
    source_pos = source_pos.cpu().numpy()

    bold_min, bold_max = true_bold_np.min(), true_bold_np.max()
    bold_pad = (bold_max - bold_min) * 0.05
    neural_min, neural_max = true_neural_np.min(), true_neural_np.max()
    neural_pad = (neural_max - neural_min) * 0.05

    fig, axes = plt.subplots(nrows=n_layers, figsize=(10, 4 * n_layers), constrained_layout=True)
    if n_layers == 1:
        axes = [axes]

    for i in range(n_layers):
        ax_bold = axes[i]
        ax_neural = ax_bold.twinx()

        # BOLD on left axis
        ax_bold.plot(
            times,
            true_bold_np[n_layers - i - 1],
            color=COLOR_HEX[SIGNALS_LIST.index("bold")],
            label="True BOLD",
            ls="-",
        )
        ax_bold.plot(
            times,
            pred_bold_np[n_layers - i - 1],
            color=COLOR_HEX[SIGNALS_LIST.index("bold")],
            label="Predicted BOLD",
            ls="-.",
            alpha=0.8,
        )

        # Neural on right axis
        ax_neural.plot(
            times,
            true_neural_np[n_layers - i - 1],
            color=COLOR_HEX[SIGNALS_LIST.index("x")],
            label="True Neural",
            ls="-",
        )
        ax_neural.plot(
            times,
            pred_neural_np[n_layers - i - 1],
            color=COLOR_HEX[SIGNALS_LIST.index("x")],
            label="Predicted Neural",
            ls="-.",
            alpha=0.8,
        )

        ax_bold.set_title(LAYER_NAMES[i], fontfamily="monospace")
        ax_bold.set_xlabel("Time (s)", fontfamily="monospace")
        ax_bold.set_ylabel("BOLD Signal", fontfamily="monospace")
        ax_neural.set_ylabel("Neural Activity", fontfamily="monospace")
        ax_bold.set_ylim(bold_min - bold_pad, bold_max + bold_pad)
        ax_neural.set_ylim(neural_min - neural_pad, neural_max + neural_pad)

        ax_bold.legend(loc="upper left", frameon=False)
        ax_neural.legend(loc="upper right", frameon=False)

        for ax in (ax_bold, ax_neural):
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily("monospace")

    return fig


def plot_latent_layers(
    pred_s: torch.Tensor,
    true_s: torch.Tensor,
    pred_f: torch.Tensor,
    true_f: torch.Tensor,
    pred_v: torch.Tensor,
    true_v: torch.Tensor,
    pred_q: torch.Tensor,
    true_q: torch.Tensor,
    pred_v_star: torch.Tensor,
    true_v_star: torch.Tensor,
    pred_q_star: torch.Tensor,
    true_q_star: torch.Tensor,
    tr: float = 1.0,
    title: str = "Latent States",
) -> Figure:
    """
    Plot all 7 Heinzle latent signals across 3 cortical layers for a single sample.

    Each input tensor is [L, T].
    Layout: rows = layers (Deep -> Superficial), cols = signals (x, s, f, v, q, v*, q*).
    """
    pred_signals = [
        pred_s.cpu().numpy(),
        pred_f.cpu().numpy(),
        pred_v.cpu().numpy(),
        pred_q.cpu().numpy(),
        pred_v_star.cpu().numpy(),
        pred_q_star.cpu().numpy(),
    ]
    true_signals = [
        true_s.cpu().numpy(),
        true_f.cpu().numpy(),
        true_v.cpu().numpy(),
        true_q.cpu().numpy(),
        true_v_star.cpu().numpy(),
        true_q_star.cpu().numpy(),
    ]

    n_layers = pred_signals[0].shape[0]
    n_signals = len(pred_signals)
    total_duration = pred_signals[0].shape[-1] * tr
    pred_times = np.linspace(0, total_duration, pred_signals[0].shape[-1])
    true_times = np.linspace(0, total_duration, true_signals[0].shape[-1])

    sig_ylims = []
    for true_arr in true_signals:
        lo, hi = true_arr.min(), true_arr.max()
        pad = (hi - lo) * 0.05
        sig_ylims.append((lo - pad, hi + pad))

    fig, axes = plt.subplots(
        nrows=n_layers,
        ncols=n_signals,
        figsize=(3 * n_signals, 3 * n_layers),
        constrained_layout=True,
    )
    fig.suptitle(title, fontfamily="monospace", fontsize=13)

    for row, layer_name in enumerate(LAYER_NAMES):
        for col, (pred_arr, true_arr, sig_name) in enumerate(
            zip(pred_signals, true_signals, LATENT_NAMES, strict=True)
        ):
            ax = axes[row, col]
            color = COLOR_HEX[SIGNALS_LIST.index(sig_name)]
            layer_idx = n_layers - row - 1
            ax.plot(true_times, true_arr[layer_idx], color=color, ls="-", label="True")
            ax.plot(
                pred_times,
                pred_arr[layer_idx],
                color=color,
                linewidth=0.9,
                ls="-.",
                alpha=0.6,
                label="Pred",
            )
            ax.legend(loc="upper right", frameon=False, fontsize=7)
            ax.set_ylim(*sig_ylims[col])

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
