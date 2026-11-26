# luso-mt

Training and evaluation code for **luso-mt**, compact machine translation for English, Chichewa and Tumbuka.

| Model | Parameters | int8 size | Licence |
|---|---|---|---|
| [luso-mt-tiny](https://huggingface.co/suru8-ai/luso-mt-tiny) | 19.9M | 40.5 MB | Apache-2.0 |
| [luso-mt-small](https://huggingface.co/suru8-ai/luso-mt-small) | 87.7M | 153.5 MB | Apache-2.0 |

One model translates in all four directions (en→ny, ny→en, en→tum, tum→en); the target language is chosen with a
tag at the start of the input. The students are distilled from MADLAD-400 3B fine-tuned on Malawian data. All
system outputs on the test sets are in the
[luso-mt benchmark](https://huggingface.co/datasets/suru8-ai/luso-mt-benchmark).

## Quick start

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("suru8-ai/luso-mt-small")
model = AutoModelForSeq2SeqLM.from_pretrained("suru8-ai/luso-mt-small")

def translate(text, to):  # to: "en", "ny" or "tum"
    batch = tok(f"<2{to}> {text}", return_tensors="pt")
    out = model.generate(**batch, num_beams=4, max_new_tokens=192)
    return tok.decode(out[0], skip_special_tokens=True)

print(translate("Where is the nearest clinic?", "ny"))
```

## Repository layout

```
data/build_mt_data.py            build training pairs, fixed test/dev sets and the 8-word leak filter
training/finetune_mt_teacher.py  fine-tune MADLAD-400 3B into the teacher; translate sources for distillation
training/train_luso_mt.py        train tiny/small students from scratch (with or without distillation)
evaluation/eval_mt_candidates.py score existing open models on the test sets
evaluation/benchmark_mt.py       file size, CPU speed and example translations
evaluation/eval_int8.py          translation quality after int8 quantisation
launch.py, configs/              Hydra configs and launcher for training runs and seed sweeps
evaluation/mt_report.py          score, length, language-confusion and win matrices
evaluation/build_benchmark.py    assemble the public benchmark of system outputs
evaluation/make_cards.py         model cards from eval.json and benchmark.json
paper/compute_results.py         scores, confidence intervals, paired bootstrap, slices, wins, language ID
paper/make_tables.py             LaTeX tables from paper/data
paper/make_figures.py            report figures from paper/data (PDF and 300-dpi PNG)
paper/figures/                   architecture and pipeline diagrams (SVG)
```

Every script declares its dependencies inline ([PEP 723](https://peps.python.org/pep-0723/)) and runs with
`uv run script.py`. Training runs on Hugging Face Jobs and is configured with Hydra/OmegaConf
(`configs/config.yaml`, `configs/experiment/*.yaml`):

```
uv run launch.py experiment=tiny_kd seed=14                       # one run
uv run launch.py -m experiment=tiny_kd,tiny_base,tiny_rep2 seed=13,14,15   # seed sweep
uv run launch.py -m experiment=small_kd seed=13,14,15 hf.flavor=h200 dry_run=true
```

`tiny_rep2` is the matched-exposure control: the original pairs are repeated twice so the run sees as many rows per
epoch as a distilled run, with the same epoch limit.

## Data

The training corpus is not public, so the training runs cannot be reproduced from this repository alone. The test
sets are public: NTREX-128 (en↔ny, CC-BY-SA-4.0) and Google SMOL (en↔tum, CC-BY-4.0). Everything in `paper/` can be
regenerated from the released benchmark.

## Results (test chrF++)

| Model | en→ny | ny→en | en→tum | tum→en |
|---|---|---|---|---|
| luso-mt-small | 47.6 | 46.9 | 35.8 | 30.0 |
| luso-mt-tiny | 46.3 | 45.5 | 34.9 | 29.1 |
| NLLB-200 600M (non-commercial) | 50.0 | 55.2 | 38.2 | 32.7 |
| OPUS-MT | 41.0 | 42.7 | n/a | 24.1 |

## Citation

```bibtex
@techreport{luso-mt-2026,
  title       = {luso-mt: Compact Machine Translation for English, Chichewa and Tumbuka},
  author      = {Byamasu, Patrick Paul},
  institution = {Rexplore Research Labs},
  year        = {2026}
}
```

## Licence

Code: Apache-2.0 (see `LICENSE`). Models: Apache-2.0. Benchmark: CC-BY-SA-4.0.
