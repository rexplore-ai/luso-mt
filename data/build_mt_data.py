"""Parallel data for luso-mt (en<->ny, en<->tum), with held-out sets the model never sees.

Inputs
  --ny   luso-corpus-001 (chichewa.parquet): Chichewa text + English (human, Gemma or Gemini)
  --tum  tumbuka_translated.parquet (build_tumbuka_text.py + translate_corpus.py --src tum)
  --eval suru8-ai/luso-mt-eval candidates.json: the test sets used to score candidate models

Splits
  test  NTREX-128 nya 0-499 and SMOL en-tum (the 500 sentences in candidates.json)
  dev   NTREX-128 nya 500-999 and 300 further SMOL en-tum pairs
  train everything else, after removing every pair that shares an 8-word window with dev/test
        (either side), exact duplicates, length outliers and NTREX/FLORES-style gold data

Each pair appears in both directions. `origin` says where the English came from:
  human | mt-gemma | mt-gemini. Rows whose source side is machine-made get `bt=1`
  (tagged back-translation: the model learns not to copy translationese).
`release` keeps restricted (not redistributable) rows identifiable, so a public model can be
trained without them.

    python build_mt_data.py --ny chichewa.parquet --tum tumbuka_translated.parquet \
        --eval candidates.json --out mt_data --push suru8-ai/luso-mt-data
"""
import argparse
import json
import os
import random
import re
import unicodedata

import pandas as pd

WORD = re.compile(r"\w+", re.U)


def norm(t):
    t = unicodedata.normalize("NFKC", str(t)).lower().replace("’", "'")
    return " ".join(WORD.findall(t))


def shingles(t, n=8):
    w = norm(t).split()
    return {" ".join(w[i:i + n]) for i in range(max(1, len(w) - n + 1))} if w else set()


def origin_of(ts):
    ts = str(ts)
    if ts.startswith("human") or ts == "":
        return "human"
    return "mt-gemma" if "gemma" in ts else "mt-gemini"


def load(path, lang):
    df = pd.read_parquet(path)
    df = df[(df.en.fillna("") != "") & (df.text.fillna("") != "")]
    if "granularity" in df:
        df = df[(df.granularity == "sentence") | (df.n_words <= 120)]
    return pd.DataFrame({"xx": df.text.str.strip(), "en": df.en.str.strip(), "lang": lang,
                         "origin": df.translation_source.map(origin_of), "release": df.release,
                         "source": df.source})


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--ny", required=True)
    a.add_argument("--tum", required=True)
    a.add_argument("--eval", required=True)
    a.add_argument("--out", default="mt_data")
    a.add_argument("--push", default="")
    a.add_argument("--seed", type=int, default=13)
    args = a.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ev = json.load(open(args.eval, encoding="utf-8"))["test_sets"]
    from datasets import load_dataset
    ny_all = load_dataset("davidstap/NTREX", "nya_Latn", split="test")["text"]
    en_all = load_dataset("davidstap/NTREX", "eng_Latn", split="test")["text"]
    smol = [(r["src"], r["trg"]) for r in load_dataset("google/smol", "smolsent__en_tum", split="train")]
    test_tum = {(p["en"], p["xx"]) for p in ev["en-tum"]}
    rest_tum = [p for p in smol if p not in test_tum]
    random.Random(args.seed).shuffle(rest_tum)
    held = {
        "test": [(p["en"], p["xx"], "ny") for p in ev["en-ny"]] + [(e, x, "tum") for e, x in test_tum],
        "dev": [(e, x, "ny") for e, x in zip(en_all[500:1000], ny_all[500:1000])] + [(e, x, "tum") for e, x in rest_tum[:300]],
    }
    banned = set()
    for split in held.values():
        for e, x, _ in split:
            banned |= shingles(e) | shingles(x)

    df = pd.concat([load(args.ny, "ny"), load(args.tum, "tum")], ignore_index=True)
    n0 = len(df)
    df = df[~df.source.str.startswith(("NTREX", "FLORES"))]                      # gold news sets: eval only
    df["wx"], df["we"] = df.xx.str.split().str.len(), df.en.str.split().str.len()
    df = df[(df.wx.between(2, 150)) & (df.we.between(2, 150)) & ((df.wx / df.we).between(0.33, 3))]
    df["k"] = df.xx.map(norm) + " ||| " + df.en.map(norm)
    df = df.drop_duplicates("k")
    leak = df.apply(lambda r: bool((shingles(r.xx) | shingles(r.en)) & banned), axis=1)
    print(f"rows {n0} -> {len(df)} after filters; removed {int(leak.sum())} that overlap dev/test")
    df = df[~leak]

    rows = []
    for r in df.itertuples():
        bt_en = int(r.origin != "human")      # English is machine-made: as a *source* it is back-translation
        rows.append({"src": r.en, "tgt": r.xx, "src_lang": "en", "tgt_lang": r.lang, "bt": bt_en,
                     "origin": r.origin, "release": r.release, "split": "train"})
        rows.append({"src": r.xx, "tgt": r.en, "src_lang": r.lang, "tgt_lang": "en", "bt": 0,
                     "origin": r.origin, "release": r.release, "split": "train"})
    for split, pairs in held.items():
        for e, x, lang in pairs:
            rows.append({"src": e, "tgt": x, "src_lang": "en", "tgt_lang": lang, "bt": 0, "origin": "human",
                         "release": "public", "split": split})
            rows.append({"src": x, "tgt": e, "src_lang": lang, "tgt_lang": "en", "bt": 0, "origin": "human",
                         "release": "public", "split": split})
    out = pd.DataFrame(rows)
    out["direction"] = out.src_lang + "->" + out.tgt_lang
    stats = out.groupby(["split", "direction", "release"]).size().unstack(fill_value=0)
    print(stats.to_string())
    for s in ("train", "dev", "test"):
        out[out.split == s].drop(columns="split").to_parquet(os.path.join(args.out, f"{s}.parquet"), index=False)
    rows_by = {f"{s} {d}": {k: int(v) for k, v in r.items()} for (s, d), r in stats.iterrows()}
    with open(os.path.join(args.out, "stats.json"), "w") as f:
        json.dump({"rows": rows_by, "removed_overlap": int(leak.sum())}, f, indent=1)
    if args.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="dataset",
                          commit_message="luso-mt parallel data (train/dev/test, en<->ny, en<->tum)")
        print("pushed", args.push)


if __name__ == "__main__":
    main()
