"""Plot PSC-IMB permeability convergence for the local fines window."""

from __future__ import annotations

import csv
from pathlib import Path
from math import ceil

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "convergence" / "permeability" / "permeability_convergence_raw.csv"
OUT = ROOT / "figures"

plt.rcParams.update(
    {
        "font.family": "Arial",
            "font.size": 8.4,
            "axes.labelsize": 8.8,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def read_rows() -> list[dict[str, str]]:
    with IN.open(newline="") as fh:
        return list(csv.DictReader(fh))


def readable_xticks(resolutions: list[int]) -> list[int]:
    """Keep final-PDF tick labels readable as adaptive high-resolution points grow."""
    if len(resolutions) <= 7:
        return resolutions
    keep = {resolutions[0], resolutions[-1]}
    stride = max(1, ceil((len(resolutions) - 2) / 5))
    keep.update(resolutions[1:-1:stride])
    return [n for n in resolutions if n in keep]


def annotate_unresolved_ratio(ax: plt.Axes, ratio_plotted: dict[str, tuple[list[int], list[float]]]) -> None:
    """Mark the ratio panel as a same-resolution, unresolved trend.

    The high-resolution branch is intentionally presented as evidence of
    grid-sensitivity, not as a converged permeability-loss estimate.  Keeping
    this cue in the plotting script prevents future data merges from producing
    a visually overconfident convergence figure.
    """
    packed = ratio_plotted.get("packed_slug")
    if not packed:
        return
    n, ratio = packed
    if len(n) < 3:
        return
    monotone_nonincreasing = all(b <= a + 1e-12 for a, b in zip(ratio, ratio[1:]))
    if not monotone_nonincreasing:
        return
    ax.annotate(
        "not converged",
        xy=(n[-1], ratio[-1]),
        xytext=(-98, -16),
        textcoords="offset points",
        color="#8A3B12",
        fontsize=8.0,
        arrowprops={
            "arrowstyle": "->",
            "lw": 0.8,
            "color": "#8A3B12",
            "shrinkA": 2,
            "shrinkB": 2,
        },
        bbox={
            "boxstyle": "round,pad=0.22",
            "fc": "white",
            "ec": "#D55E00",
            "lw": 0.6,
            "alpha": 0.92,
        },
    )
    ax.text(
        0.50,
        0.94,
        "same-resolution ratios only",
        transform=ax.transAxes,
        fontsize=8.0,
        color="0.32",
        ha="center",
        va="top",
        bbox={
            "boxstyle": "round,pad=0.20",
            "fc": "white",
            "ec": "0.82",
            "lw": 0.5,
            "alpha": 0.92,
        },
    )


def main() -> None:
    rows = read_rows()
    cases = ["skeleton", "baseline", "packed_slug"]
    labels = {"skeleton": "skeleton", "baseline": "passive fines", "packed_slug": "preloaded fine slug"}
    colors = {"skeleton": "#222222", "baseline": "#0072B2", "packed_slug": "#D55E00"}
    markers = {"skeleton": "o", "baseline": "s", "packed_slug": "^"}
    linestyles = {"skeleton": "-", "baseline": "--", "packed_slug": "-"}
    all_resolutions = sorted({int(r["resolution"]) for r in rows})
    width = 7.2 if len(all_resolutions) <= 6 else min(7.75, 7.15 + 0.055 * len(all_resolutions))
    fig, axes = plt.subplots(1, 2, figsize=(width, 2.55), constrained_layout=True)
    plotted: dict[str, tuple[list[int], list[float]]] = {}
    ratio_plotted: dict[str, tuple[list[int], list[float]]] = {}
    for case in cases:
        case_rows = sorted([r for r in rows if r["case"] == case], key=lambda r: int(r["resolution"]))
        n = [int(r["resolution"]) for r in case_rows]
        k = [float(r["permeability_lu"]) for r in case_rows]
        plotted[case] = (n, k)
        axes[0].plot(
            n,
            k,
            marker=markers[case],
            color=colors[case],
            linestyle=linestyles[case],
            lw=1.55,
            ms=5.2,
            mec="white",
            mew=0.45,
            label=labels[case],
        )
        if case != "skeleton":
            ratio_rows = [r for r in case_rows if r.get("K_over_skeleton_same_resolution")]
            n_ratio = [int(r["resolution"]) for r in ratio_rows]
            ratio = [float(r["K_over_skeleton_same_resolution"]) for r in ratio_rows]
            if ratio:
                ratio_plotted[case] = (n_ratio, ratio)
                axes[1].plot(
                    n_ratio,
                    ratio,
                    marker=markers[case],
                    color=colors[case],
                    linestyle=linestyles[case],
                    lw=1.7,
                    ms=5.4,
                    mec="white",
                    mew=0.45,
                    label=labels[case],
                )
    axes[0].set_xlabel(r"grid resolution $N^3$")
    axes[0].set_ylabel(r"$K_\mathrm{PSC}$ (lattice units)")
    axes[1].set_xlabel(r"grid resolution $N^3$")
    axes[1].set_ylabel(r"$K/K_\mathrm{skeleton}$")
    axes[1].axhline(1.0, color="0.55", lw=0.85, ls=":", zorder=0)
    plotted_ratios = [
        float(r["K_over_skeleton_same_resolution"])
        for r in rows
        if r["case"] != "skeleton" and r.get("K_over_skeleton_same_resolution")
    ]
    if plotted_ratios:
        low = min(plotted_ratios) - 0.025
        high = max(plotted_ratios) + 0.015
        axes[1].set_ylim(min(0.815, low), max(1.01, high))
    tick_resolutions = readable_xticks(all_resolutions)
    for ax in axes:
        ax.set_xticks(tick_resolutions)
        if len(all_resolutions) > len(tick_resolutions):
            ax.set_xlim(min(all_resolutions) - 6, max(all_resolutions) + 34)
        ax.grid(True, axis="y", color="0.90", lw=0.55)
        ax.tick_params(direction="out", length=3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    for case, (n, values) in plotted.items():
        if n and values:
            dy = {"skeleton": 0.006, "baseline": -0.001, "packed_slug": -0.008}.get(case, 0.0)
            axes[0].text(n[-1] + 4, values[-1] + dy, labels[case], color=colors[case], va="center", fontsize=8.0)
    for case, (n, values) in ratio_plotted.items():
        if n and values:
            axes[1].text(n[-1] + 4, values[-1], labels[case], color=colors[case], va="center", fontsize=8.0)
    annotate_unresolved_ratio(axes[1], ratio_plotted)
    for ax, label in zip(axes, ("a", "b")):
        ax.text(-0.12, 1.03, label, transform=ax.transAxes, fontweight="bold", va="top", fontsize=8.8)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "psc_permeability_convergence.pdf")
    fig.savefig(OUT / "psc_permeability_convergence.png", dpi=300)
    fig.savefig(OUT / "psc_permeability_convergence.svg")
    print(f"wrote {OUT / 'psc_permeability_convergence.pdf'}")


if __name__ == "__main__":
    main()
