from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

DEFAULT_PALETTE = ["#3498db", "#e74c3c", "#9b59b6", "#2ecc71", "#f39c12"]


def set_style(dpi=150, font_scale=1.05, style='whitegrid', palette=None):
    """Set a consistent, publication-friendly plotting style."""
    if palette is None:
        palette = DEFAULT_PALETTE
    sns.set_theme(style=style, palette=palette, font_scale=font_scale)
    plt.rcParams['figure.dpi'] = dpi
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['legend.frameon'] = False


def nice_boxplot(ax, data, x, y, palette=None, orient='v', showfliers=False, order=None):
    """Draw a stylized boxplot with jittered stripplot overlay and counts annotation."""
    if palette is None:
        palette = DEFAULT_PALETTE
    sns.boxplot(data=data, x=x, y=y, ax=ax, showfliers=showfliers, palette=palette, order=order)
    sns.stripplot(data=data, x=x, y=y, ax=ax, color='0.15', size=3, jitter=0.25, order=order)

    # Annotate counts above each box
    if x is not None:
        groups = data[x].unique() if order is None else order
        for i, g in enumerate(groups):
            try:
                cnt = len(data[data[x] == g])
            except Exception:
                cnt = np.nan
            ax.text(i, ax.get_ylim()[1] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f'n={cnt}', ha='center', va='bottom', fontsize=9)

    ax.grid(axis='y', linestyle='--', alpha=0.4)
    return ax


def nice_violin(ax, data, x, y, palette=None, order=None, inner='quartile'):
    """Draw a violin + swarm plot for richer distributions."""
    if palette is None:
        palette = DEFAULT_PALETTE
    sns.violinplot(data=data, x=x, y=y, ax=ax, palette=palette, inner=inner, order=order)
    sns.swarmplot(data=data, x=x, y=y, ax=ax, color='0.15', size=3, order=order)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    return ax


def bar_with_values(ax, x, height, color=None, xlabel=None, ylabel=None, title=None, rotation=0):
    if color is None:
        color = DEFAULT_PALETTE
    bars = ax.bar(x, height, color=color)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01 * max(height), f'{h:.1f}' if isinstance(h, float) else f'{int(h)}',
                ha='center', va='bottom', fontsize=9)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_xticklabels(x, rotation=rotation)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    return ax


def savefig(fig, path, dpi=300, tight=True):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(str(p), dpi=dpi)


# Convenience example: apply at top of notebook
# from src.plot_utils import set_style, nice_boxplot, nice_violin, savefig
# set_style(dpi=150, font_scale=1.05)
