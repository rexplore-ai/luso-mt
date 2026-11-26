# /// script
# requires-python = ">=3.10,<3.12"
# dependencies = ["pandas<2.3", "pyarrow==14.0.2", "sacrebleu>=2.4", "fasttext-wheel", "numpy<2", "huggingface_hub"]
# ///
"""Every number behind the report's results section, recomputed from system outputs on the test sets.

    hf jobs uv run --flavor cpu-upgrade --namespace suru8-ai --secrets HF_TOKEN compute_results.py

Inputs: all baseline and luso-mt outputs (suru8-ai/luso-mt-benchmark) plus luso-mt-3b outputs.
Output (written to suru8-ai/luso-mt-paper, build/results.json and build/predictions.parquet):
  significance   corpus chrF++/BLEU, 95% CI and paired-bootstrap p-value against a reference system
  wins           share of sentences where one system's sentence chrF++ beats another's
  slices         corpus chrF++ by source length (short < 10 words, long > 25 words; slices of >= 20 sentences)
  langid         GlotLID language of every output, as a confusion table per system
"""
import json
import fasttext
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from sacrebleu.metrics import BLEU, CHRF
from sacrebleu.significance import PairedTest

PAPER = "suru8-ai/luso-mt-paper"
D = ["en->ny", "ny->en", "en->tum", "tum->en"]
OPUS = {"en->ny": "opus-mt-en-ny", "ny->en": "opus-mt-ny-en", "tum->en": "opus-mt-tum-en"}

bench = pd.read_parquet(hf_hub_download("suru8-ai/luso-mt-benchmark", "data/predictions.parquet", repo_type="dataset"))
big = pd.read_json(hf_hub_download("suru8-ai/luso-mt-3b-research", "predictions.jsonl"), lines=True)
big["model"] = "luso-mt-3b"
p = pd.concat([bench[["model", "direction", "src", "ref", "hyp"]], big[["model", "direction", "src", "ref", "hyp"]]])
p["hyp"] = p.hyp.fillna("")
# one shared sentence id per direction, in the order of the test set (luso-mt-small's file)
order = {d: {s: i for i, s in enumerate(bench[(bench.model == "luso-mt-small") & (bench.direction == d)].src)} for d in D}
p["id"] = [order[d][s] for d, s in zip(p.direction, p.src)]
p = p.drop_duplicates(["model", "direction", "id"]).sort_values(["model", "direction", "id"]).reset_index(drop=True)
# OPUS-MT: one single-pair model per direction, reported as one system
opus = p[[m == OPUS.get(d) for m, d in zip(p.model, p.direction)]].assign(model="OPUS-MT")
p = pd.concat([p, opus], ignore_index=True)

chrf, bleu = CHRF(word_order=2), BLEU()
p["s_chrf"] = [chrf.sentence_score(h, [r]).score for h, r in zip(p.hyp, p.ref)]
p["src_words"] = p.src.str.split().str.len()
MAIN = ["luso-mt-small", "luso-mt-tiny", "luso-mt-3b", "luso-mt-teacher", "nllb-200-distilled-600M", "madlad400-3b-mt", "OPUS-MT"]
res = {"systems": MAIN}

# ---- significance: every system against a reference (luso-mt-small, luso-mt-3b, luso-mt-tiny) ----
res["significance"] = {}
for base in ("luso-mt-small", "luso-mt-3b", "luso-mt-tiny"):
    res["significance"][base] = {}
    for d in D:
        g = p[p.direction == d]
        names = [base] + [m for m in MAIN if m != base and m in set(g.model)]
        hyps = [g[g.model == m].sort_values("id").hyp.tolist() for m in names]
        refs = g[g.model == base].sort_values("id").ref.tolist()
        _, sc = PairedTest(list(zip(names, hyps)), {"chrF++": chrf, "BLEU": bleu}, references=[refs],
                           test_type="bs", n_samples=1000)()
        kc = next(k for k in sc if k.lower().startswith("chrf")); kb = next(k for k in sc if k.upper().startswith("BLEU"))
        res["significance"][base][d] = {n: {"chrF++": {"score": round(float(c.score), 1), "ci95": round(float(c.ci), 1),
                                                      "p": None if c.p_value is None else round(float(c.p_value), 4)},
                                            "BLEU": {"score": round(float(b.score), 1), "ci95": round(float(b.ci), 1)}}
                                        for n, c, b in zip(names, sc[kc], sc[kb])}
        print("significance", base, d, flush=True)

# ---- wins: pooled over directions both systems support ----
wins = {}
for a in MAIN:
    wins[a] = {}
    for b in MAIN:
        if a == b:
            continue
        x = p[p.model == a].set_index(["direction", "id"]).s_chrf
        y = p[p.model == b].set_index(["direction", "id"]).s_chrf
        j = x.index.intersection(y.index)
        wins[a][b] = round(float((x[j] > y[j]).mean() * 100), 1)
res["wins"] = wins

# ---- slices by source length ----
def bucket(n):
    return "short" if n < 10 else ("long" if n > 25 else "medium")
p["slice"] = p.src_words.map(bucket)
res["slices"] = {}
for m in MAIN:
    res["slices"][m] = {}
    for d in D:
        for s in ("short", "medium", "long"):
            g = p[(p.model == m) & (p.direction == d) & (p.slice == s)]
            res["slices"][m][f"{d}|{s}"] = round(chrf.corpus_score(g.hyp.tolist(), [g.ref.tolist()]).score, 1) if len(g) >= 20 else None
res["slice_sizes"] = {f"{d}|{s}": int(((p.model == "luso-mt-small") & (p.direction == d) & (p.slice == s)).sum()) for d in D for s in ("short", "medium", "long")}

# ---- language identity (GlotLID) ----
lid = fasttext.load_model(hf_hub_download("cis-lmu/glotlid", "model.bin"))
MAP = {"__label__eng_Latn": "en", "__label__nya_Latn": "ny", "__label__tum_Latn": "tum"}
labels = lid.predict([h.replace("\n", " ") if h.strip() else "." for h in p.hyp.tolist()])[0]
p["out_lang"] = [MAP.get(l[0], "other") for l in labels]
p["asked"] = p.direction.str.split("->").str[1]
res["langid"] = {m: {a: {o: round(float(((g.out_lang == o).mean()) * 100), 1) for o in ("en", "ny", "tum", "other")}
                     for a, g in p[p.model == m].groupby("asked")} for m in MAIN}

api = HfApi()
api.upload_file(path_or_fileobj=json.dumps(res, indent=1).encode(), path_in_repo="build/results.json", repo_id=PAPER,
                repo_type="dataset", commit_message="results with luso-mt-3b")
p.drop(columns=["slice"]).to_parquet("/tmp/predictions.parquet", index=False)
api.upload_file(path_or_fileobj="/tmp/predictions.parquet", path_in_repo="build/predictions.parquet", repo_id=PAPER,
                repo_type="dataset", commit_message="all system outputs with sentence chrF++ and language id")
print(json.dumps(res["langid"], indent=1))
print("done", flush=True)
