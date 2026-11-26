"""Draw every chart in the report from the data in data/ (vector PDF for LaTeX, 300-dpi PNG for previews).

    python scripts/make_figures.py          # run from paper/latex; needs numpy and matplotlib

Inputs: data/results.json (compute_results.py), data/fertility.json, data/lengths.json, data/curves.json.
The architecture and pipeline diagrams are SVG files in figures/src/ (see README).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA, FIG = ROOT / "data", ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.edgecolor": "black", "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
# Okabe-Ito colour-blind-safe palette
C = {"luso-mt-tiny": "#E69F00", "luso-mt-small": "#D55E00", "luso-mt-3b": "#000000", "luso-mt-teacher": "#999999",
     "nllb-200-distilled-600M": "#0072B2", "madlad400-3b-mt": "#56B4E9", "OPUS-MT": "#009E73"}
NAME = {"luso-mt-tiny": "luso-mt-tiny (20M)", "luso-mt-small": "luso-mt-small (88M)", "luso-mt-3b": "luso-mt-3b (2.9B)",
        "luso-mt-teacher": "Teacher, public data (2.9B)", "nllb-200-distilled-600M": "NLLB-200 (615M)",
        "madlad400-3b-mt": "MADLAD-400 (2.9B)", "OPUS-MT": "OPUS-MT (75M)"}
D = ["en->ny", "ny->en", "en->tum", "tum->en"]
ARROW = {d: d.replace("->", r" $\rightarrow$ ") for d in D}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=300)
    plt.close(fig)
    print("wrote", name)


R = json.loads((DATA / "results.json").read_text())
SYS = ["luso-mt-3b", "luso-mt-small", "luso-mt-tiny", "luso-mt-teacher", "nllb-200-distilled-600M", "madlad400-3b-mt", "OPUS-MT"]
assert set(SYS) == set(R["systems"])

# ---------- chrF++ with 95% CI ----------
sig = R["significance"]["luso-mt-small"]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True, gridspec_kw={"wspace": 0.12})
for ax, d in zip(axes, D):
    for i, m in enumerate(SYS):
        r = sig[d].get(m)
        if not r:
            ax.text(1.5, i, "n/s", va="center", fontsize=7, color="#555555", style="italic")
            continue
        v, ci = r["chrF++"]["score"], r["chrF++"]["ci95"]
        ax.barh(i, v, xerr=ci, color=C[m], height=0.7, edgecolor="black", linewidth=0.4,
                error_kw={"ecolor": "black", "elinewidth": 0.6, "capsize": 1.5})
        ax.text(v + ci + 1, i, f"{v:.1f}", va="center", fontsize=6.8)
    ax.set_yticks(range(len(SYS)))
    ax.set_yticklabels([NAME[m] for m in SYS])
    ax.set_ylim(len(SYS) - 0.5, -0.5)
    ax.set_xlim(0, 70)
    ax.set_title(ARROW[d])
    ax.set_xlabel("chrF++")
    ax.tick_params(axis="y", length=0)
save(fig, "fig_ci")

# ---------- tokeniser fertility ----------
fert = json.loads((DATA / "fertility.json").read_text())
names = list(fert)
cols = ["#D55E00", "#E69F00", "#0072B2", "#56B4E9", "#009E73"]
fig, ax = plt.subplots(figsize=(4.8, 2.3))
w = 0.16
for i, n in enumerate(names):
    ax.bar(np.arange(3) + (i - 2) * w, [fert[n][l]["fertility"] for l in ("en", "ny", "tum")], w,
           color=cols[i], edgecolor="black", linewidth=0.4, label=n)
ax.set_xticks(range(3)); ax.set_xticklabels(["English", "Chichewa", "Tumbuka"])
ax.set_ylabel("Subword tokens per word")
ax.legend(frameon=False, ncol=2, loc="upper left")
save(fig, "fig_fertility")

# ---------- sentence lengths ----------
L = json.loads((DATA / "lengths.json").read_text())
b = np.array(L["bins"])
fig, ax = plt.subplots(figsize=(4.8, 2.1))
for key, lab, col in (("ny|all", "Chichewa", "#D55E00"), ("tum|all", "Tumbuka", "#0072B2")):
    cnt = np.array(L[key]["counts"], dtype=float)
    ax.step(b[:-1], cnt / cnt.sum(), where="post", color=col, lw=1.1, label=f"{lab} (n = {L[key]['n']:,})")
ax.set_xlabel("Words per sentence (Chichewa or Tumbuka side, clipped at 80)")
ax.set_ylabel("Share of pairs")
ax.legend(frameon=False)
save(fig, "fig_lengths")

# ---------- training curves ----------
cur = json.loads((DATA / "curves.json").read_text())
style = {"first run (failed)": ("#555555", ":", "v0 tiny (failed)"),
         "tiny, no distillation": ("#E69F00", "--", "tiny, no distillation"),
         "small, no distillation": ("#D55E00", "--", "small, no distillation"),
         "tiny, distilled": ("#E69F00", "-", "tiny, distilled"),
         "small, distilled": ("#D55E00", "-", "small, distilled")}
fig, ax = plt.subplots(figsize=(5.6, 2.9))
for lab, (col, ls, key) in style.items():
    ax.plot(cur[key]["epoch"], cur[key]["dev_chrf"], ls, color=col, lw=1.3, label=lab)
ax.set_xlabel("Epoch"); ax.set_ylabel("Dev chrF++ (mean, greedy)")
ax.legend(frameon=False, loc="lower right")
save(fig, "fig_curves")

# ---------- teacher and luso-mt-3b development loss ----------
tc = json.loads((DATA / "teacher_curves.json").read_text())
fig, ax = plt.subplots(figsize=(5.6, 2.7))
t, b3 = tc["teacher"], tc["luso-mt-3b"]
ax.plot(t["train_epoch"], t["train_loss"], color="#999999", lw=0.8, alpha=0.8)
ax.plot([1 + e for e in b3["train_epoch"]], b3["train_loss"], color="#888888", lw=0.8, alpha=0.8)
ax.plot(t["eval_epoch"], t["eval_loss"], "o-", color="#999999", lw=1.6, ms=4, label="teacher, public data: dev loss")
ax.plot([1 + e for e in b3["eval_epoch"]], b3["eval_loss"], "o-", color="black", lw=1.6, ms=4,
        label="luso-mt-3b, full corpus: dev loss")
ax.plot([], [], color="#999999", lw=0.8, label="training loss")
ax.axvline(1, color="black", lw=0.6, ls=":")
ax.text(1.02, ax.get_ylim()[1], "continued on full corpus", va="top", fontsize=7.5)
ax.set_xlabel("Epoch (teacher 0-1, luso-mt-3b 1-2)"); ax.set_ylabel("Loss (label-smoothed)")
ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.85))
save(fig, "fig_teacher_loss")

# ---------- difference from NLLB-200 ----------
nllb = "nllb-200-distilled-600M"
others = [m for m in SYS if m != nllb]
fig, ax = plt.subplots(figsize=(7.0, 2.8))
w = 0.13
for i, m in enumerate(others):
    xs, ys = [], []
    for k, d in enumerate(D):
        r = sig[d].get(m)
        if r:
            xs.append(k + (i - (len(others) - 1) / 2) * w)
            ys.append(r["chrF++"]["score"] - sig[d][nllb]["chrF++"]["score"])
    ax.bar(xs, ys, w, color=C[m], edgecolor="black", linewidth=0.4, label=NAME[m])
    # significance against NLLB-200 for the luso-mt models (paired bootstrap)
    base = m if m in R["significance"] else None
    if base:
        for x, y, d in zip(xs, ys, [d for d in D if sig[d].get(m)]):
            pv = R["significance"][base][d][nllb]["chrF++"]["p"]
            if pv is not None and pv < 0.05:
                ax.text(x, y + (0.35 if y >= 0 else -0.35), "*", ha="center", va="bottom" if y >= 0 else "top", fontsize=9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(4)); ax.set_xticklabels([ARROW[d] for d in D])
ax.set_ylabel(r"chrF++ minus NLLB-200")
ax.legend(frameon=False, ncol=3, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.02))
save(fig, "fig_delta")

# ---------- heatmap helper (blue scale; text black or white for contrast) ----------
def heat(ax, M, rows, cols, vmin, vmax, fmt="{:.1f}"):
    ax.imshow(np.ma.masked_invalid(M), cmap="Blues", vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=6.5, color="#777777")
            else:
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="white" if (v - vmin) / (vmax - vmin) > 0.55 else "black")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

# ---------- chrF++ by length ----------
S = R["slices"]
M = np.array([[np.nan if S[m][f"{d}|{s}"] is None else S[m][f"{d}|{s}"] for d in D for s in ("short", "medium", "long")] for m in SYS])
fig, ax = plt.subplots(figsize=(7.0, 2.4))
heat(ax, M, [NAME[m] for m in SYS], ["short", "medium", "long"] * 4, 15, 62)
for k, d in enumerate(D):
    ax.text(3 * k + 1, -0.85, ARROW[d], ha="center", va="bottom", fontsize=8.5)
    if k:
        ax.axvline(3 * k - 0.5, color="white", lw=2.5)
save(fig, "fig_slices")

# ---------- language confusion ----------
lid = R["langid"]
fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.3), gridspec_kw={"hspace": 0.75, "wspace": 0.55})
for ax, m in zip(axes.flat, SYS):
    asked = [a for a in ("en", "ny", "tum") if a in lid[m]]
    Mx = np.array([[lid[m][a][o] for o in ("en", "ny", "tum", "other")] for a in asked])
    heat(ax, Mx, [f"asked {a}" for a in asked], ["en", "ny", "tum", "oth."], 0, 100, "{:.0f}")
    ax.set_title(NAME[m], fontsize=7.5)
axes.flat[-1].axis("off")
save(fig, "fig_confusion")

# ---------- pairwise wins ----------
W = R["wins"]
Mw = np.array([[np.nan if a == b else W[a][b] for b in SYS] for a in SYS])
fig, ax = plt.subplots(figsize=(4.9, 3.4))
short = {"luso-mt-tiny": "tiny", "luso-mt-small": "small", "luso-mt-3b": "luso-mt-3b", "luso-mt-teacher": "teacher",
         "nllb-200-distilled-600M": "NLLB-200", "madlad400-3b-mt": "MADLAD-400", "OPUS-MT": "OPUS-MT"}
heat(ax, Mw, [NAME[m] for m in SYS], [short[m] for m in SYS], 0, 100, "{:.0f}")
plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
save(fig, "fig_wins")
