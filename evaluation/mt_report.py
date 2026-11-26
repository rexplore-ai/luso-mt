# /// script
# requires-python = ">=3.10"
# dependencies = ["sacrebleu>=2.4", "fasttext-wheel", "numpy<2", "matplotlib", "pandas", "huggingface_hub", "wandb"]
# ///
"""Comparison report for luso-mt: how each model performs against the others, and where it fails.

Reads prediction files (one JSON object per line: model, direction, src, ref, hyp) from local paths
or Hugging Face repos, and produces, for the same fixed test sets:

  1. Score matrix         models x directions, chrF++ and BLEU (heatmap + table)
  2. Slice matrix         models x (direction, sentence length bucket), chrF++
  3. Language confusion   per model: requested target language (rows) x language actually written
                          (columns: en / ny / tum / other, identified with GlotLID). A perfect model
                          is a diagonal; NLLB writing Chichewa when asked for Tumbuka shows up here.
  4. Win matrix           models x models: share of sentences where the row model beats the column
                          model on sentence chrF++ (pairwise, per direction pooled)

Writes report.md + PNG figures to --out, and logs them to W&B (team rexplore-labs-ltd, project
luso-mt) when WANDB_API_KEY is set.

    uv run mt_report.py --pred hf:suru8-ai/luso-mt-eval/predictions.jsonl \
        --pred hf:suru8-ai/luso-mt-small/predictions.jsonl --out report_out
"""
import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import sacrebleu

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DIRS = ["en->ny", "ny->en", "en->tum", "tum->en"]
LANG = {"eng_Latn": "en", "nya_Latn": "ny", "tum_Latn": "tum"}
COLS = ["en", "ny", "tum", "other"]
BLUE = "#008FDB"


def read_preds(spec):
    if spec.startswith("hf:"):
        from huggingface_hub import hf_hub_download
        repo, path = spec[3:].split("/", 2)[:2], spec[3:].split("/", 2)[2]
        repo = "/".join(repo)
        for rtype in ("dataset", "model"):
            try:
                spec = hf_hub_download(repo, path, repo_type=rtype)
                break
            except Exception:
                continue
    return [json.loads(line) for line in open(spec, encoding="utf-8") if line.strip()]


def length_bucket(src):
    n = len(src.split())
    return "short (<10)" if n < 10 else "long (>25)" if n > 25 else "medium"


def lid_model():
    import fasttext
    from huggingface_hub import hf_hub_download
    return fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))


def detect(lid, texts):
    out = []
    for t in texts:
        (lab,), _ = lid.predict(t.replace("\n", " ") or " ")
        out.append(LANG.get(lab.replace("__label__", ""), "other"))
    return out


def heatmap(mat, rows, cols, title, path, fmt="{:.1f}", vmin=None, vmax=None, cmap="Blues"):
    fig, ax = plt.subplots(figsize=(1.4 + 1.15 * len(cols), 1.0 + 0.55 * len(rows)))
    im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(cols)), cols, rotation=30, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    hi = np.nanmax(mat) if np.isfinite(mat).any() else 1
    for i in range(len(rows)):
        for j in range(len(cols)):
            v = mat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        color="white" if v > 0.6 * hi else "#1f2937", fontsize=9)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def md_table(df, fmt="{:.1f}"):
    head = "| " + " | ".join([df.index.name or ""] + list(map(str, df.columns))) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    rows = ["| " + " | ".join([str(i)] + [fmt.format(v) if pd.notna(v) else "—" for v in r]) + " |"
            for i, r in zip(df.index, df.values)]
    return "\n".join([head, sep] + rows)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--pred", action="append", required=True, help="path or hf:<repo>/<file>")
    a.add_argument("--out", default="report_out")
    a.add_argument("--title", default="luso-mt comparison")
    a.add_argument("--no-lid", action="store_true")
    a.add_argument("--alias", action="append", default=[], help="old=new: rename a model (merge several into one)")
    a.add_argument("--drop", nargs="*", default=[], help="model names to leave out")
    a.add_argument("--order", nargs="*", default=[], help="model display order (others follow)")
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.DataFrame([r for spec in args.pred for r in read_preds(spec)])
    df = df[df.direction.isin(DIRS) & ~df.model.isin(args.drop)]
    df["model"] = df.model.replace(dict(x.split("=", 1) for x in args.alias))
    found = list(dict.fromkeys(df.model))
    models = [m for m in args.order if m in found] + [m for m in found if m not in args.order]
    df["tgt"] = df.direction.str.split("->").str[1]
    df["bucket"] = df.src.map(length_bucket)
    df["sent_chrf"] = [sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(df.hyp, df.ref)]

    # 1. score matrix
    score = pd.DataFrame(index=pd.Index(models, name="model"), columns=DIRS, dtype=float)
    bleu = score.copy()
    for (m, d), g in df.groupby(["model", "direction"]):
        score.loc[m, d] = sacrebleu.corpus_chrf(list(g.hyp), [list(g.ref)], word_order=2).score
        bleu.loc[m, d] = sacrebleu.corpus_bleu(list(g.hyp), [list(g.ref)]).score
    heatmap(score.values.astype(float), models, DIRS, "chrF++ by direction (higher is better)",
            os.path.join(args.out, "score_matrix.png"), vmin=0, vmax=70)

    # 2. slice matrix
    slices = [(d, b) for d in DIRS for b in ("short (<10)", "medium", "long (>25)")]
    sl = pd.DataFrame(index=pd.Index(models, name="model"), columns=[f"{d} {b}" for d, b in slices], dtype=float)
    for m in models:
        for d, b in slices:
            g = df[(df.model == m) & (df.direction == d) & (df.bucket == b)]
            if len(g) >= 20:
                sl.loc[m, f"{d} {b}"] = sacrebleu.corpus_chrf(list(g.hyp), [list(g.ref)], word_order=2).score
    heatmap(sl.values.astype(float), models, list(sl.columns), "chrF++ by direction and sentence length",
            os.path.join(args.out, "slice_matrix.png"), vmin=0, vmax=70)

    # 3. language confusion
    conf_md = []
    if not args.no_lid:
        lid = lid_model()
        df["out_lang"] = detect(lid, list(df.hyp))
        n = len(models)
        ncols = min(n, 4)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.9 * nrows), squeeze=False)
        for ax in axes.flat[n:]:
            ax.axis("off")
        for k, m in enumerate(models):
            g = df[df.model == m]
            rows = [t for t in ("en", "ny", "tum") if (g.tgt == t).any()]
            mat = np.array([[((g.tgt == t) & (g.out_lang == c)).sum() / max((g.tgt == t).sum(), 1) * 100
                              for c in COLS] for t in rows])
            ax = axes[k // ncols, k % ncols]
            ax.imshow(mat, cmap="Blues", vmin=0, vmax=100)
            ax.set_xticks(range(len(COLS)), COLS)
            ax.set_yticks(range(len(rows)), [f"asked {r}" for r in rows])
            for i in range(len(rows)):
                for j in range(len(COLS)):
                    ax.text(j, i, f"{mat[i, j]:.0f}", ha="center", va="center",
                            color="white" if mat[i, j] > 60 else "#1f2937", fontsize=9)
            ax.set_title(m, fontsize=9)
            ax.set_xlabel("language written")
            t = pd.DataFrame(mat, index=pd.Index([f"asked {r}" for r in rows], name=m), columns=COLS)
            conf_md.append(md_table(t, "{:.0f}%"))
        fig.suptitle("Language confusion: % of outputs in each language", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "language_confusion.png"), dpi=160)
        plt.close(fig)

    # 4. win matrix (pairwise, sentence chrF++)
    key = df.assign(k=df.direction + "|" + df.src).pivot_table(index="k", columns="model", values="sent_chrf")
    win = pd.DataFrame(index=pd.Index(models, name="row beats column"), columns=models, dtype=float)
    for r in models:
        for c in models:
            if r != c:
                both = key[[r, c]].dropna()
                win.loc[r, c] = (both[r] > both[c]).mean() * 100 if len(both) else np.nan
    heatmap(win.values.astype(float), models, models, "Share of sentences where the row model beats the column (%)",
            os.path.join(args.out, "win_matrix.png"), fmt="{:.0f}", vmin=0, vmax=100)

    md = [f"# {args.title}", "",
          "Fixed test sets: NTREX-128 (en↔ny, 500 sentences) and SMOL en-tum (500 sentences). chrF++ unless noted.", "",
          "## Score matrix (chrF++)", "", md_table(score), "", "![score matrix](score_matrix.png)", "",
          "## BLEU", "", md_table(bleu), "",
          "## By sentence length (chrF++, slices with at least 20 sentences)", "", md_table(sl), "",
          "![slice matrix](slice_matrix.png)", ""]
    if conf_md:
        md += ["## Language confusion", "",
               "Rows: language we asked for. Columns: language the output is actually in (GlotLID). "
               "A good model is on the diagonal.", ""]
        for t in conf_md:
            md += [t, ""]
        md += ["![language confusion](language_confusion.png)", ""]
    md += ["## Win matrix", "", "Share of test sentences (all directions pooled) where the row model's sentence chrF++ "
           "is higher than the column model's.", "", md_table(win, "{:.0f}%"), "", "![win matrix](win_matrix.png)", ""]
    open(os.path.join(args.out, "report.md"), "w", encoding="utf-8").write("\n".join(md))
    score.to_csv(os.path.join(args.out, "score_matrix.csv"))
    print("\n".join(md[:12]))
    print("wrote", args.out)

    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(entity=os.environ.get("WANDB_ENTITY", "rexplore-labs-ltd"), project="luso-mt",
                         name="comparison-report", job_type="report", tags=["luso-mt", "report"])
        run.log({"score_matrix": wandb.Table(dataframe=score.reset_index()),
                 **{f.replace(".png", ""): wandb.Image(os.path.join(args.out, f))
                    for f in os.listdir(args.out) if f.endswith(".png")}})
        run.finish()


if __name__ == "__main__":
    main()
