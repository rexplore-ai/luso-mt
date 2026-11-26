import sys, pandas as pd
out = sys.argv[1]
s = pd.read_csv(f"{out}/data/scores.csv"); sp = pd.read_csv(f"{out}/data/speed.csv")
D = ["en->ny", "ny->en", "en->tum", "tum->en"]; A = {d: d.replace("->", "→") for d in D}
order = ["luso-mt-small", "luso-mt-tiny", "luso-mt-teacher", "nllb-200-distilled-600M", "madlad400-3b-mt",
         "opus-mt-en-ny", "opus-mt-ny-en", "opus-mt-tum-en", "opus-mt-en-mul", "opus-mt-mul-en"]
lic = {"luso-mt-small": "Apache-2.0", "luso-mt-tiny": "Apache-2.0", "luso-mt-teacher": "not released",
       "nllb-200-distilled-600M": "CC-BY-NC-4.0", "madlad400-3b-mt": "Apache-2.0"}
def cell(m, d):
    r = s[(s.model == m) & (s.direction == d)]
    return "—" if r.empty else (f"**{r.chrf_pp.iloc[0]}** / {r.bleu.iloc[0]}" if m.startswith("luso-mt-") and m != "luso-mt-teacher" else f"{r.chrf_pp.iloc[0]} / {r.bleu.iloc[0]}")
def link(m):
    c = s[s.model == m].checkpoint.iloc[0]
    return f"[{m}](https://huggingface.co/{c})" if " " not in c else m
rows = "\n".join(f"| {link(m)} | {lic.get(m, 'CC-BY-4.0')} | " + " | ".join(cell(m, d) for d in D) + " |" for m in order)
srows = "\n".join(f"| [{r.model}](https://huggingface.co/suru8-ai/{r.model}) | {r.params_M}M | {r.fp32_MB} MB | {r.int8_MB} MB | {r.cpu_sent_per_s_greedy} | {r.cpu_sent_per_s_beam4} | {r.cpu_sent_per_s_int8_greedy} |" for r in sp.itertuples())
threads = sp.threads.iloc[0]
card = f"""---
license: cc-by-sa-4.0
language:
- en
- ny
- tum
pretty_name: luso-mt benchmark
task_categories:
- translation
tags:
- machine-translation
- chichewa
- tumbuka
- malawi
- benchmark
- luso
size_categories:
- 10K<n<100K
configs:
- config_name: scores
  default: true
  data_files: data/scores.csv
- config_name: predictions
  data_files: data/predictions.parquet
- config_name: speed
  data_files: data/speed.csv
---

# luso-mt benchmark

How well do open translation models handle **English ↔ Chichewa** and **English ↔ Tumbuka**? This dataset holds
everything behind the comparison in the [luso-mt technical report](https://rexplore.ai/blog/luso-mt-technical-report):
the outputs of every model on the same held-out test sentences, the scores computed from them, and CPU speed and size
for the two released luso-mt models.

Released models: [suru8-ai/luso-mt-tiny](https://huggingface.co/suru8-ai/luso-mt-tiny) (19.9M) and
[suru8-ai/luso-mt-small](https://huggingface.co/suru8-ai/luso-mt-small) (87.7M), Apache-2.0.

## Test sets

| Directions | Test set | Sentences | Domain | Licence |
|---|---|---|---|---|
| en→ny, ny→en | [NTREX-128](https://github.com/MicrosoftTranslator/NTREX) (`nya`), first 500 | 500 | news | CC-BY-SA-4.0 |
| en→tum, tum→en | [SMOL](https://huggingface.co/datasets/google/smol) `smolsent` en-tum, seeded shuffle, first 500 | 500 | everyday sentences | CC-BY-4.0 |

Both sets, and every training pair sharing an 8-word window with them, were removed from luso-mt's training data.
The same 500 sentences are used in both directions of each pair.

## Results (chrF++ / BLEU)

chrF++ is sacreBLEU `CHRF(word_order=2)`; BLEU is sacreBLEU `BLEU()` with default settings, one reference, corpus level.
All systems decoded with beam 4. "—": the model does not support that direction.

| Model | Licence | en→ny | ny→en | en→tum | tum→en |
|---|---|---|---|---|---|
{rows}

luso-mt-teacher is MADLAD-400 3B fine-tuned on our data; it was used to make distillation targets and is not released.

![chrF++ by model and direction](figures/score_matrix.png)

The technical report also breaks scores down by sentence length (`figures/slice_matrix.png`), checks that each model
writes the requested language with GlotLID (`figures/language_confusion.png`), and compares models sentence by
sentence (`figures/win_matrix.png`).

## Speed and size

| Model | Parameters | fp32 file | int8 file | Sentences/s, greedy | Sentences/s, beam 4 | Sentences/s, int8 greedy |
|---|---|---|---|---|---|---|
{srows}

{threads} vCPUs (Hugging Face Jobs `cpu-upgrade`), batch size 1, 40 NTREX-128 sentences translated into Chichewa,
`max_new_tokens=128`. int8 is PyTorch dynamic quantisation of linear layers (embeddings stay fp32). Not measured on phones.

## Files

| Config | File | Rows | Columns |
|---|---|---|---|
| `scores` | `data/scores.csv` | {len(s)} | model, checkpoint, direction, test_set, n, chrf_pp, bleu |
| `predictions` | `data/predictions.parquet` | {s.n.sum():,} | model, checkpoint, direction, test_set, id, src, ref, hyp |
| `speed` | `data/speed.csv` | {len(sp)} | model, threads, params_M, fp32_MB, int8_MB, cpu_sent_per_s_* |

```python
from datasets import load_dataset
import sacrebleu

pred = load_dataset("suru8-ai/luso-mt-benchmark", "predictions", split="train").to_pandas()
g = pred[(pred.model == "luso-mt-small") & (pred.direction == "en->ny")]
print(sacrebleu.CHRF(word_order=2).corpus_score(g.hyp.tolist(), [g.ref.tolist()]).score)  # 47.6
```

To score a new system, translate the `src` of one model's rows for a direction (the sources are the same for every
model) and compare with `ref` in the same way.

## Licence and attribution

The source and reference sentences come from NTREX-128 (Federmann, Kocmi and Xin, 2022; CC-BY-SA-4.0) and SMOL
(Caswell et al., 2025; CC-BY-4.0), so this dataset is released under CC-BY-SA-4.0. System outputs are included for
evaluation and research; each model's own licence is listed in the results table.

## Citation

```bibtex
@misc{{luso-mt-2026,
  title  = {{luso-mt: small translation models for English, Chichewa and Tumbuka}},
  author = {{Byamasu, Patrick Paul}},
  year   = {{2026}},
  url    = {{https://rexplore.ai/blog/luso-mt-technical-report}}
}}
```

Maintained by Byamasu Patrick Paul, Rexplore Research Labs (Lilongwe, Malawi). Contact: research@rexplore.ai
"""
open(f"{out}/README.md", "w", encoding="utf-8", newline="\n").write(card)
print(card[card.index("## Results"):card.index("luso-mt-teacher is")])
print(card[card.index("## Speed"):card.index("{threads}") if "{threads}" in card else card.index("## Files")])
