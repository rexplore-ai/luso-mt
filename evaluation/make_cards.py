"""Write the final luso-mt model cards (tiny, small) from eval.json and benchmark.json.

    python make_cards.py final_evals.json benchmark.json model_cards/

final_evals.json maps run name (luso-mt-{tiny,small}-kd) to that run's eval.json; benchmark.json is the output of
benchmark_mt.py (suru8-ai/luso-mt-eval).
"""
import json
import sys

evals = json.load(open(sys.argv[1], encoding="utf-8"))
bench = json.load(open(sys.argv[2], encoding="utf-8"))["models"]
out_dir = sys.argv[3]
D = ["en->ny", "ny->en", "en->tum", "tum->en"]
NAME = {"en->ny": "en→ny", "ny->en": "ny→en", "en->tum": "en→tum", "tum->en": "tum→en"}
ARCH = {
    "tiny": dict(d=320, h="5 × 64", ff=1024, e=8, dd=2, voc="16,000", use="any phone, fully offline"),
    "small": dict(d=512, h="8 × 64", ff=2048, e=12, dd=4, voc="32,000", use="apps that need the best quality on a phone"),
}
REF = {
    "NLLB-200-distilled-600M (reference, non-commercial)": ["50.0 / 20.1", "55.2 / 34.2", "38.2 / 13.0", "32.7 / 12.0"],
    "MADLAD-400 3B (zero-shot)": ["37.8 / 11.5", "51.3 / 30.7", "not supported", "26.8 / 5.9"],
    "OPUS-MT en-ny / ny-en / tum-en (75M each)": ["41.0 / 12.0", "42.7 / 18.6", "not available", "24.1 / 5.4"],
}

for size in ("tiny", "small"):
    e = evals[f"luso-mt-{size}-kd"]
    b = bench[f"suru8-ai/luso-mt-{size}-kd"]
    a = ARCH[size]
    t = e["test"]
    rows = "\n".join(f"| {k} | " + " | ".join(v) + " |" for k, v in REF.items())
    ours = " | ".join(f"**{t[d]['chrF++']}** / {t[d]['BLEU']}" for d in D)
    other = "small" if size == "tiny" else "tiny"
    o = evals[f"luso-mt-{other}-kd"]["test"]
    other_row = f"| luso-mt-{other} | " + " | ".join(f"{o[d]['chrF++']} / {o[d]['BLEU']}" for d in D) + " |"
    gap = " | ".join(f"{e['gap_train_minus_dev'][d]}" for d in D)
    def metric(d):
        return f"""      - type: chrf
        name: chrF++ {NAME[d]}
        value: {t[d]['chrF++']}
      - type: bleu
        name: BLEU {NAME[d]}
        value: {t[d]['BLEU']}"""
    def result(ds_type, ds_name, dirs):
        return f"""  - task:
      type: translation
      name: Translation
    dataset:
      type: {ds_type}
      name: {ds_name}
      split: test
    metrics:
""" + chr(10).join(metric(d) for d in dirs)
    mi = [result("mt-ntrex-128", "NTREX-128 (en-ny, 500 sentences)", D[:2]),
          result("google/smol", "SMOL (en-tum, 500 sentences)", D[2:])]
    examples = "\n".join(f"| `<2{x['to']}> {x['src']}` | {x['out']} |" for x in b["examples"])
    card = f"""---
license: apache-2.0
language:
- en
- ny
- tum
library_name: transformers
pipeline_tag: translation
datasets:
- google/smol
metrics:
- chrf
- bleu
tags:
- translation
- chichewa
- tumbuka
- malawi
- on-device
- luso
model-index:
- name: luso-mt-{size}
  results:
{chr(10).join(mi)}
---

# luso-mt-{size}

**luso-mt-{size}** translates between English and Chichewa, and between English and Tumbuka, in a single
{e['params_M']}M-parameter model built for {a['use']}. It is part of the luso-mt family (`luso-mt-tiny`,
`luso-mt-small`), released by Rexplore Research Labs under Apache-2.0.

Announcement: [rexplore.ai/blog/introducing-luso-mt](https://rexplore.ai/blog/introducing-luso-mt) ·
Technical report: [rexplore.ai/blog/luso-mt-technical-report](https://rexplore.ai/blog/luso-mt-technical-report) ·
Benchmark (all test outputs and scores): [suru8-ai/luso-mt-benchmark](https://huggingface.co/datasets/suru8-ai/luso-mt-benchmark)

## How to use

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("suru8-ai/luso-mt-{size}")
model = AutoModelForSeq2SeqLM.from_pretrained("suru8-ai/luso-mt-{size}")

def translate(text, to):          # to: "en", "ny" (Chichewa) or "tum" (Tumbuka)
    batch = tok(f"<2{{to}}> {{text}}", return_tensors="pt")
    out = model.generate(**batch, num_beams=4, max_new_tokens=192)
    return tok.decode(out[0], skip_special_tokens=True)

print(translate("Where is the nearest clinic?", "ny"))
```

The target language is the tag at the start of the input (`<2en>`, `<2ny>`, `<2tum>`); the source language is not
needed. Outputs of this checkpoint for a few inputs (beam 4):

| Input | Output |
|---|---|
{examples}

## Model details

| | |
|---|---|
| Architecture | encoder-decoder Transformer, T5 v1.1 form (RMSNorm, gated-GELU, relative positions), deep encoder / shallow decoder |
| Size | d_model {a['d']}, heads {a['h']}, d_ff {a['ff']}, {a['e']} encoder layers, {a['dd']} decoder layers |
| Parameters | {e['params_M']}M (tied input and output embeddings) |
| Vocabulary | {a['voc']} SentencePiece unigram pieces shared by English, Chichewa and Tumbuka |
| Training | from scratch by sequence-level knowledge distillation from MADLAD-400 3B fine-tuned on our data |
| File size | {b['fp32_MB']} MB (fp32), {b['int8_MB']} MB with int8 dynamic quantisation |
| CPU speed | {b['cpu_sent_per_s_greedy']} sentences/s greedy, {b['cpu_sent_per_s_beam4']} with beam 4, {b['cpu_sent_per_s_int8_greedy']} int8 greedy ({{bench_threads}} vCPU, batch 1, news sentences) |
| Developed by | Byamasu Patrick Paul, Rexplore Research Labs (Lilongwe, Malawi) |
| Licence | Apache-2.0 |

## Training data

Only data we may release was used:

* Malawian Chichewa web text (news, blogs, Wikipedia): 90,441 pairs, English by machine translation (Gemini 3.8
  Flash, Gemma 3 27B);
* Tumbuka Wikipedia and Tumbuka news sites: about 50,000 pairs, English by machine translation (Gemini 3.8 Flash);
* professional translations from SMOL (Google, CC-BY-4.0);
* the teacher's translations of the same sources (285,952).

Every pair is used in both directions; machine-made source sentences are tagged (`<bt>`). Text we cannot redistribute
was excluded. Test and development sentences, and any training pair sharing an 8-word window with
them, were removed before training. The dataset is not released.

## Evaluation

Held-out test sets: NTREX-128 news (500 sentences, en ↔ ny) and SMOL everyday sentences (500 sentences, en ↔ tum).
chrF++ / BLEU (sacreBLEU), beam 4. Every system's outputs on these sentences are in
[suru8-ai/luso-mt-benchmark](https://huggingface.co/datasets/suru8-ai/luso-mt-benchmark).

| Model | en→ny | ny→en | en→tum | tum→en |
|---|---|---|---|---|
| **luso-mt-{size}** | {ours} |
{other_row}
| luso-mt teacher (MADLAD-400 3B, fine-tuned) | 49.9 / 21.9 | 53.8 / 31.4 | 37.5 / 9.7 | 34.7 / 12.7 |
{rows}

Language identity (GlotLID): the model writes the requested language in 98-100% of test outputs.

Memorisation check, chrF++ on 500 training pairs minus dev, per direction: {gap}.

## Limitations

* Trained mostly on news and web text; conversation, medicine and law are weaker.
* Most English training targets are machine translations, so English output can be literal; ny→en is
  {round(55.2 - t['ny->en']['chrF++'], 1)} chrF++ below NLLB-200.
* Tumbuka has less data and is up-sampled in training; the gap between training and development scores is larger
  for Tumbuka (see above), so Tumbuka quality is lower and less even than Chichewa.
* Names, numbers and dates should be checked.
* Very short sentences, especially instructions, are sometimes copied untranslated or translated with a repeated
  phrase (see the examples above).
* Not for legal, medical or safety-critical translation without human review.

## Citation

```bibtex
@misc{{luso-mt-2026,
  title  = {{luso-mt: small translation models for English, Chichewa and Tumbuka}},
  author = {{Byamasu, Patrick Paul}},
  year   = {{2026}},
  url    = {{https://rexplore.ai/blog/luso-mt-technical-report}}
}}
```

Contact: research@rexplore.ai
"""
    card = card.replace("{bench_threads}", str(  # left literal in the f-string above
json.load(open(sys.argv[2], encoding="utf-8"))["cpu_threads"]))
    open(f"{out_dir}/luso-mt-{size}.md", "w", encoding="utf-8", newline="\n").write(card)
    print("wrote", size)
