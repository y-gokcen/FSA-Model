#!/usr/bin/env python3
"""Accuracy and PFCoutD content as a function of the number of G tokens.

Usage: python3 tools/gcount_figs.py [--ggplot] [--tag TEXT] <outdir> <test.csv> [<test.csv> ...]

--tag TEXT is appended to the ggplot-style titles (default "current model").
--pool NAME pools all the given files into one set of figures labelled NAME
instead of one set per file (note that different runs may encode the branch
with different PFC tokens, so pooled PFCoutD plots mix encodings).

With --ggplot the figures mimic the ggplot2 look of the original figures
(default hue colors, white background, no grid, percent axis, legend on top
for the accuracy plot) so they can be compared side by side; output files
get a _gg suffix.

For each *_test.csv file (hard task) separately -- runs differ in how PFC
encodes the branch, so pooling them muddles the token plots -- splits it into
sequences (F ... H/I), bins each sequence by its number of G tokens, and
writes three figures to <outdir>, with <label> = the file name without
"_test.csv":

  <label>_accuracy.png   accuracy at the C trial (H vs I), overall correctness
                         over all trials of the sequence, and overall validity
  <label>_pfcoutd_A.png  distribution of the PFCoutD token on the C trial,
                         A-branch sequences
  <label>_pfcoutd_B.png  the same for B-branch sequences

These correspond to the figures Jasemin produced from the original model
(Base -- Overall vs. C-point accuracy; PFCoutD A branch; PFCoutD B branch).
Requires matplotlib.
"""
import csv
import collections
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BINS = ["0", "1", "2", "3", "4", "5", "6-7", "8-9", "10-12", "13-19", "20+"]


def gbin(n):
    if n <= 5:
        return str(n)
    if n <= 7:
        return "6-7"
    if n <= 9:
        return "8-9"
    if n <= 12:
        return "10-12"
    if n <= 19:
        return "13-19"
    return "20+"


# ggplot2 default hue palette (scales::hue_pal()) for 2 and 7 series.
GG2 = ["#F8766D", "#00BFC4"]
GG7 = ["#F8766D", "#C49A00", "#53B400", "#00C094", "#00B6EB", "#A58AFF", "#FB61D7"]
GG = False
TAG = "current model"

# Categorical palette (validated, fixed order) and text/grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEXT1, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
TOKENS = ["A", "B", "C", "F", "G", "H", "I"]  # D/E never occur in the hard task
TOKEN_COLOR = dict(zip(TOKENS, SERIES[:7]))


def sequences(paths):
    """Yield dicts: branch, ng, c_row, rows for every complete sequence."""
    for p in paths:
        rows = list(csv.DictReader(open(p)))
        cur = None
        for r in rows:
            if r["Stim"] == "F":
                if cur and cur.get("c_row") and cur.get("branch"):
                    yield cur
                cur = {"branch": None, "ng": 0, "c_row": None, "rows": []}
            if cur is None:
                continue
            cur["rows"].append(r)
            if r["Stim"] in "AB":
                cur["branch"] = r["Stim"]
            elif r["Stim"] == "G":
                cur["ng"] += 1
            elif r["StateNode"] in ("5", "6"):
                cur["c_row"] = r


def style(ax, ylabel, ylim=None):
    if GG:  # ggplot2 theme_classic
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("black")
        ax.tick_params(colors="black", labelsize=10)
        ax.set_xlabel("G count", color="black", fontsize=12)
        ax.set_ylabel(ylabel, color="black", fontsize=12)
        if ylim:
            ax.set_ylim(*ylim)
        return
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT2, labelsize=8)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("G count", color=TEXT2, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT2, fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)


def line(ax, xs, ys, color, label, **kw):
    if GG:
        kw = {"marker": "o", "markersize": 7}
    kw.setdefault("marker", "o")
    kw.setdefault("markersize", 6)
    ax.plot(xs, ys, color=color, linewidth=2, label=label, **kw)


def main():
    global GG, TAG
    args = sys.argv[1:]
    pool = None
    while args and args[0].startswith("--"):
        if args[0] == "--ggplot":
            GG = True
            args = args[1:]
        elif args[0] == "--tag":
            TAG = args[1]
            args = args[2:]
        elif args[0] == "--pool":
            pool = args[1]
            args = args[2:]
        else:
            sys.exit("unknown option %s" % args[0])
    outdir, paths = args[0], args[1:]
    os.makedirs(outdir, exist_ok=True)
    if pool:
        print("==", pool, "(%d files pooled)" % len(paths))
        figures(outdir, pool, paths)
        return
    for p in paths:
        label = os.path.basename(p).replace("_test.csv", "").replace(".csv", "")
        print("==", label)
        figures(outdir, label, [p])


def figures(outdir, label, paths):
    seqs = list(sequences(paths))
    bybin = collections.defaultdict(list)
    for s in seqs:
        bybin[gbin(s["ng"])].append(s)
    bins = [b for b in BINS if len(bybin[b]) >= 5]  # skip near-empty bins
    counts = [len(bybin[b]) for b in bins]
    ticks = bins if GG else ["%s\nn=%d" % (b, n) for b, n in zip(bins, counts)]
    suffix = "_gg" if GG else ""
    print("sequences: %d   per bin: %s" % (len(seqs), dict(zip(bins, counts))))

    # ---- Figure 1: accuracy by G count -------------------------------------
    c_acc, all_cor, all_val = [], [], []
    for b in bins:
        ss = bybin[b]
        c_acc.append(100 * sum(s["c_row"]["Predicted"] == s["c_row"]["NextStim"] for s in ss) / len(ss))
        rows = [r for s in ss for r in s["rows"]]
        all_cor.append(100 * sum(r["Predicted"] == r["NextStim"] for r in rows) / len(rows))
        all_val.append(100 * sum(int(r["Valid"]) for r in rows) / len(rows))
    fig, ax = plt.subplots(figsize=(8.2, 5.0) if GG else (10, 4.8), dpi=150)
    if GG:
        line(ax, ticks, c_acc, GG2[0], "C accuracy")
        line(ax, ticks, all_cor, GG2[1], "Overall accuracy")
        style(ax, "Accuracy", (0, 102))
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.set_title("Base (%s) \u2014 Overall vs. C-point accuracy" % TAG, loc="left",
                     color="black", fontsize=14, pad=28)
        ax.legend(frameon=False, fontsize=11, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    else:
        line(ax, ticks, all_val, SERIES[2], "Overall: prediction valid", linestyle="--", marker="s")
        line(ax, ticks, all_cor, SERIES[0], "Overall: prediction == next token")
        line(ax, ticks, c_acc, SERIES[1], "C-point accuracy (H vs I)", marker="D", markersize=7)
        style(ax, "Accuracy (%)", (0, 105))
        ax.set_title("%s: accuracy by number of G tokens" % label, loc="left", color=TEXT1, fontsize=12)
        ax.legend(frameon=False, fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "%s_accuracy%s.png" % (label, suffix)))
    plt.close(fig)
    print("C-point accuracy:", ["%s:%.0f" % (b, v) for b, v in zip(bins, c_acc)])

    # ---- Figures 2/3: PFCoutD token on the C trial, per branch -------------
    for branch in "AB":
        fig, ax = plt.subplots(figsize=(8.6, 5.0) if GG else (10, 4.8), dpi=150)
        dist = {}
        for b in bins:
            ss = [s for s in bybin[b] if s["branch"] == branch]
            cnt = collections.Counter(s["c_row"]["PFCoutDToken"] for s in ss)
            dist[b] = {t: 100 * cnt[t] / max(1, len(ss)) for t in TOKENS}
        for i, t in enumerate(TOKENS):
            ys = [dist[b][t] for b in bins]
            line(ax, ticks, ys, GG7[i] if GG else TOKEN_COLOR[t], t)
            if not GG and max(ys) >= 15:  # direct-label the prominent series at the right end
                ax.annotate(t, (ticks[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=9, color=TEXT1)
        if GG:
            # ggplot2 auto-scales y to the data (5% expansion); ticks every 20
            ymax = max(dist[b][t] for b in bins for t in TOKENS)
            style(ax, "Percentage", (-ymax * 0.03, ymax * 1.05))
            ax.set_yticks(range(0, int(ymax) + 1, 20))
            ax.set_title("PFCoutD %s branch (%s)" % (branch, TAG), loc="left",
                         color="black", fontsize=14)
            ax.legend(title="Token", frameon=False, fontsize=11, title_fontsize=12,
                      loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            style(ax, "Sequences (%)", (0, 105))
            ax.set_title("%s: PFCoutD token on the C trial, %s branch" % (label, branch),
                         loc="left", color=TEXT1, fontsize=12)
            ax.legend(title="Token", frameon=False, fontsize=9, title_fontsize=9,
                      loc="center left", bbox_to_anchor=(1.02, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "%s_pfcoutd_%s%s.png" % (label, branch, suffix)))
        plt.close(fig)
        want = branch
        print("branch %s: PFCoutD == %s per bin:" % (branch, want),
              ["%s:%.0f" % (b, dist[b][want]) for b in bins])


if __name__ == "__main__":
    main()
