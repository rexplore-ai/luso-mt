# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.46,<5", "sentencepiece", "sacremoses", "sacrebleu>=2.4",
#                 "datasets>=3", "huggingface_hub", "protobuf", "accelerate"]
# ///
"""Score open translation models on held-out human translations, for a small on-device luso-mt.

Directions: en->ny, ny->en (NTREX-128, news, 500 sentences) and en->tum, tum->en (SMOL smolsent,
500 sentences held out with a fixed seed). Metric: chrF++ (sacrebleu, word_order=2), plus BLEU.
Also reports parameters and sentences/second, the two numbers that decide what fits on a phone.

The test sentences are written to the results repo so later training can exclude them.

    hf jobs uv run --flavor l4x1 --namespace suru8-ai --timeout 2h --secrets HF_TOKEN \
        eval_mt_candidates.py --n 500 --repo suru8-ai/luso-mt-eval
"""
import argparse
import json
import random
import time

import sacrebleu
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def test_sets(n, seed):
    ny = load_dataset("davidstap/NTREX", "nya_Latn", split="test")["text"]
    en = load_dataset("davidstap/NTREX", "eng_Latn", split="test")["text"]
    news = [(e, y) for e, y in zip(en, ny) if e.strip() and y.strip()][:n]
    smol = [(r["src"], r["trg"]) for r in load_dataset("google/smol", "smolsent__en_tum", split="train")
            if r["src"].strip() and r["trg"].strip()]
    random.Random(seed).shuffle(smol)
    return {"en-ny": news, "en-tum": smol[:n]}


class Marian:
    def __init__(self, name, tgt_prefix=""):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(DEV).eval()
        self.prefix = tgt_prefix

    def __call__(self, texts, src, tgt):
        return generate(self.model, self.tok, [self.prefix + t for t in texts])


class NLLB:
    CODE = {"en": "eng_Latn", "ny": "nya_Latn", "tum": "tum_Latn"}

    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.float16 if DEV == "cuda" else None).to(DEV).eval()

    def __call__(self, texts, src, tgt):
        self.tok.src_lang = self.CODE[src]
        bos = self.tok.convert_tokens_to_ids(self.CODE[tgt])
        return generate(self.model, self.tok, texts, forced_bos_token_id=bos)


class Madlad:
    def __init__(self, name):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name, torch_dtype=torch.bfloat16 if DEV == "cuda" else None).to(DEV).eval()

    def supports(self, tgt):
        return self.tok.convert_tokens_to_ids(f"<2{tgt}>") != self.tok.unk_token_id

    def __call__(self, texts, src, tgt):
        return generate(self.model, self.tok, [f"<2{tgt}> {t}" for t in texts])


@torch.inference_mode()
def generate(model, tok, texts, bs=32, **kw):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEV)
        ids = model.generate(**enc, num_beams=4, max_new_tokens=256, **kw)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def short(label):
    """'opus-mt-en-ny (Marian, 77M)' -> 'opus-mt-en-ny': the name used in report figures."""
    return label.split(" (")[0]


def score(hyps, refs):
    return {"chrF++": round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 1),
            "BLEU": round(sacrebleu.corpus_bleu(hyps, [refs]).score, 1)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--n", type=int, default=500)
    a.add_argument("--seed", type=int, default=13)
    a.add_argument("--repo", default="suru8-ai/luso-mt-eval")
    a.add_argument("--skip", nargs="*", default=[])
    args = a.parse_args()
    sets = test_sets(args.n, args.seed)

    # (label, loader, directions it claims to handle)
    cands = [
        ("opus-mt-en-ny (Marian, 77M)", lambda: Marian("Helsinki-NLP/opus-mt-en-ny"), [("en", "ny")]),
        ("opus-mt-ny-en (Marian, 77M)", lambda: Marian("Helsinki-NLP/opus-mt-ny-en"), [("ny", "en")]),
        ("opus-mt-tum-en (Marian, 77M)", lambda: Marian("Helsinki-NLP/opus-mt-tum-en"), [("tum", "en")]),
        ("opus-mt-en-mul (Marian, 77M)", lambda: Marian("Helsinki-NLP/opus-mt-en-mul", ">>nya<< "), [("en", "ny")]),
        ("opus-mt-mul-en (Marian, 77M)", lambda: Marian("Helsinki-NLP/opus-mt-mul-en"), [("ny", "en"), ("tum", "en")]),
        ("nllb-200-distilled-600M (CC-BY-NC)", lambda: NLLB("facebook/nllb-200-distilled-600M"),
         [("en", "ny"), ("ny", "en"), ("en", "tum"), ("tum", "en")]),
        ("madlad400-3b-mt (Apache-2.0)", lambda: Madlad("google/madlad400-3b-mt"),
         [("en", "ny"), ("ny", "en"), ("en", "tum"), ("tum", "en")]),
    ]
    results, samples, preds = [], {}, []
    for label, load, dirs in cands:
        if any(s in label for s in args.skip):
            continue
        try:
            m = load()
        except Exception as e:  # a missing checkpoint should not end the whole run
            print(f"{label}: could not load ({e.__class__.__name__}: {e})", flush=True)
            continue
        params = sum(p.numel() for p in m.model.parameters())
        for src, tgt in dirs:
            if isinstance(m, Madlad) and not m.supports(tgt):
                print(f"{label}: no <2{tgt}> token, skipping {src}->{tgt}", flush=True)
                continue
            pairs = sets["en-ny" if "ny" in (src, tgt) else "en-tum"]
            srcs = [p[0] if src == "en" else p[1] for p in pairs]
            refs = [p[1] if src == "en" else p[0] for p in pairs]
            t0 = time.time()
            hyps = m(srcs, src, tgt)
            dt = time.time() - t0
            r = {"model": label, "direction": f"{src}->{tgt}", "params_M": round(params / 1e6),
                 "sent_per_s": round(len(srcs) / dt, 1), "n": len(srcs), **score(hyps, refs)}
            results.append(r)
            samples[f"{label} | {src}->{tgt}"] = [{"src": s, "ref": f, "hyp": h} for s, f, h in list(zip(srcs, refs, hyps))[:8]]
            preds += [{"model": short(label), "direction": f"{src}->{tgt}", "src": s, "ref": f, "hyp": h}
                      for s, f, h in zip(srcs, refs, hyps)]
            print(json.dumps(r), flush=True)
        del m
        torch.cuda.empty_cache()

    print("\n== chrF++ by direction")
    for d in ["en->ny", "ny->en", "en->tum", "tum->en"]:
        for r in sorted([r for r in results if r["direction"] == d], key=lambda r: -r["chrF++"]):
            print(f"{d:8s} {r['chrF++']:5.1f} chrF++  {r['BLEU']:5.1f} BLEU  {r['params_M']:5d}M  {r['sent_per_s']:6.1f}/s  {r['model']}")

    api = HfApi()
    api.create_repo(args.repo, repo_type="dataset", private=True, exist_ok=True)
    blob = {"results": results, "samples": samples,
            "test_sets": {k: [{"en": e, "xx": x} for e, x in v] for k, v in sets.items()},
            "note": "NTREX-128 nya test (first n) and SMOL smolsent en_tum (seeded shuffle, first n). Exclude from training."}
    api.upload_file(path_or_fileobj=json.dumps(blob, ensure_ascii=False, indent=1).encode(), path_in_repo="candidates.json",
                    repo_id=args.repo, repo_type="dataset", commit_message="Candidate MT scores on held-out sets")
    # every translation, for mt_report.py (score, slice, language-confusion and win matrices)
    api.upload_file(path_or_fileobj="\n".join(json.dumps(p, ensure_ascii=False) for p in preds).encode(),
                    path_in_repo="predictions.jsonl", repo_id=args.repo, repo_type="dataset",
                    commit_message="Candidate translations of the fixed test sets")
    print(f"uploaded to {args.repo}/candidates.json and predictions.jsonl")


if __name__ == "__main__":
    main()
