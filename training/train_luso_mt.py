# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.46,<5", "sentencepiece", "sacrebleu>=2.4", "datasets>=3",
#                 "huggingface_hub", "accelerate", "protobuf", "pandas", "pyarrow", "wandb"]
# ///
"""luso-mt: small on-device translator for English <-> Chichewa and English <-> Tumbuka.

One model, four directions; the target language is a tag on the input ("<2ny> ...").

Architecture (encoder-decoder Transformer, T5 v1.1 style, trained from scratch):
  * RMSNorm pre-norm, gated-GELU feed-forward, relative position buckets (no length limit learnt)
  * deep encoder / shallow decoder: decoding cost sits in the decoder, so 2-4 decoder layers keep
    phone latency low while a deep encoder keeps quality (Kasai et al. 2021)
  * one SentencePiece unigram vocabulary for en, ny, tum, tied input/output embeddings

  size   d_model heads d_ff  enc dec vocab  params
  tiny     320    5   1024   8   2   16k   19.9M   (int8 ~20 MB)
  small    512    8   2048  12   4   32k   87.7M   (int8 ~90 MB)
  base     768   12   3072  12   6   32k  208.6M   (server / recent phones)

Two-step training ("transfer, then compress"):
  1. a large pretrained model is fine-tuned on our data (finetune_mt_teacher.py)
  2. this script trains the student on the original pairs plus the teacher's translations of the
     same sources (sequence-level knowledge distillation, Kim & Rush 2016), passed with --kd.

Against memorisation: dev/test pairs and anything sharing an 8-word window with them are removed
upstream (build_mt_data.py); here we add dropout, label smoothing, weight decay, subword sampling
(SentencePiece alpha=0.1), tagged back-translation (<bt>), early stopping on dev chrF++ and a final
train-vs-dev gap report.

    hf jobs uv run --flavor h200 --namespace suru8-ai --timeout 8h --secrets HF_TOKEN --secrets WANDB_API_KEY \
        train_luso_mt.py --size small --data suru8-ai/luso-mt-data --repo suru8-ai/luso-mt-small

Logs to Weights & Biases (team rexplore-labs-ltd, project luso-mt) when WANDB_API_KEY is set.
"""
import argparse
import json
import math
import os
import random

import pandas as pd
import sacrebleu
import sentencepiece as spm
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import Dataset
from transformers import (DataCollatorForSeq2Seq, EarlyStoppingCallback, Seq2SeqTrainer,
                          Seq2SeqTrainingArguments, T5Config, T5ForConditionalGeneration, T5Tokenizer)

SIZES = {
    "tiny": dict(d_model=320, num_heads=5, d_kv=64, d_ff=1024, num_layers=8, num_decoder_layers=2, vocab=16000, dropout=0.1),
    "small": dict(d_model=512, num_heads=8, d_kv=64, d_ff=2048, num_layers=12, num_decoder_layers=4, vocab=32000, dropout=0.1),
    "base": dict(d_model=768, num_heads=12, d_kv=64, d_ff=3072, num_layers=12, num_decoder_layers=6, vocab=32000, dropout=0.1),
}
TAGS = ["<2en>", "<2ny>", "<2tum>", "<bt>"]


def load_split(repo, split):
    return pd.read_parquet(hf_hub_download(repo, f"{split}.parquet", repo_type="dataset"))


def train_tokenizer(df, vocab, out_dir, seed):
    text = pd.concat([df.src, df.tgt]).drop_duplicates()
    text = text.sample(min(len(text), 3_000_000), random_state=seed)
    path = os.path.join(out_dir, "spm_corpus.txt")
    with open(path, "w", encoding="utf-8") as f:
        for t in text:
            f.write(t.replace("\n", " ") + "\n")
    prefix = os.path.join(out_dir, "spiece")
    spm.SentencePieceTrainer.train(
        input=path, model_prefix=prefix, vocab_size=vocab, model_type="unigram", character_coverage=0.9999,
        pad_id=0, eos_id=1, unk_id=2, bos_id=-1, user_defined_symbols=TAGS, input_sentence_size=3_000_000,
        shuffle_input_sentence=True, normalization_rule_name="nmt_nfkc", num_threads=os.cpu_count())
    os.remove(path)
    return prefix + ".model"


def wandb_init(name, config):
    """W&B under the rexplore-labs-ltd team; silently off when no key is available."""
    if not os.environ.get("WANDB_API_KEY"):
        return "none"
    os.environ.setdefault("WANDB_ENTITY", "rexplore-labs-ltd")
    os.environ.setdefault("WANDB_PROJECT", "luso-mt")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    import wandb
    wandb.init(entity=os.environ["WANDB_ENTITY"], project=os.environ["WANDB_PROJECT"], name=name, config=config,
               tags=["luso-mt", name])
    return "wandb"


def wandb_log_final(report):
    try:
        import wandb
        if wandb.run is None:
            return
        flat = {}
        for split in ("test", "dev", "train_probe"):
            for d, m in report.get(split, {}).items():
                for k, v in m.items():
                    if k != "n":
                        flat[f"final/{split}/{d}/{k}"] = v
        for d, v in report.get("gap_train_minus_dev", {}).items():
            flat[f"final/gap_train_minus_dev/{d}"] = v
        wandb.log(flat)
        wandb.summary.update(flat)
        wandb.finish()
    except Exception as e:
        print("wandb final log skipped:", e)


def tag(src, tgt_lang, bt):
    return f"<2{tgt_lang}> " + ("<bt> " if bt else "") + src


class Pairs(Dataset):
    def __init__(self, df, tok, max_len=192, sample=False):
        self.src = [tag(s, t, b) for s, t, b in zip(df.src, df.tgt_lang, df.bt)]
        self.tgt = list(df.tgt)
        self.tok, self.max_len, self.sample = tok, max_len, sample

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        x = self.tok(self.src[i], max_length=self.max_len, truncation=True)
        y = self.tok(text_target=self.tgt[i], max_length=self.max_len, truncation=True)
        return {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"], "labels": y["input_ids"]}


def balance(df, seed, tum_up=3.0, cap_internal=None):
    """Up-sample Tumbuka directions (little data) and optionally cap restricted rows."""
    parts = []
    for (_, tl), g in df.groupby(["src_lang", "tgt_lang"]):
        if cap_internal is not None:
            pub, intl = g[g.release == "public"], g[g.release != "public"]
            g = pd.concat([pub, intl.sample(min(len(intl), int(cap_internal * max(len(pub), 1))), random_state=seed)])
        if "tum" in (g.src_lang.iloc[0], tl):
            g = pd.concat([g] * int(tum_up) + [g.sample(frac=tum_up % 1, random_state=seed)])
        parts.append(g)
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


@torch.inference_mode()
def translate(model, tok, srcs, tgt_lang, bs=64, beams=4):
    model.eval()
    out = []
    for i in range(0, len(srcs), bs):
        enc = tok([tag(s, tgt_lang, 0) for s in srcs[i:i + bs]], return_tensors="pt", padding=True,
                  truncation=True, max_length=192).to(model.device)
        ids = model.generate(**enc, num_beams=beams, max_new_tokens=192)
        out += tok.batch_decode(ids, skip_special_tokens=True)
    return out


def chrf(hyps, refs):
    return round(sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score, 1)


def score_split(model, tok, df, n=None, beams=4, preds=None, name=None):
    """chrF++/BLEU per direction; when `preds` is a list, every translation is appended to it
    (model, direction, src, ref, hyp) for mt_report.py."""
    res = {}
    for d, g in df.groupby("direction"):
        g = g.head(n) if n else g
        hyps = translate(model, tok, list(g.src), g.tgt_lang.iloc[0], beams=beams)
        res[d] = {"chrF++": chrf(hyps, list(g.tgt)),
                  "BLEU": round(sacrebleu.corpus_bleu(hyps, [list(g.tgt)]).score, 1), "n": len(g)}
        if preds is not None:
            preds += [{"model": name, "direction": d, "src": s, "ref": r, "hyp": h}
                      for s, r, h in zip(g.src, g.tgt, hyps)]
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--size", choices=SIZES, default="small")
    a.add_argument("--data", default="suru8-ai/luso-mt-data")
    a.add_argument("--kd", default="", help="dataset repo with kd.parquet: teacher translations of train sources")
    a.add_argument("--public-only", action="store_true", help="drop restricted rows")
    a.add_argument("--cap-internal", type=float, default=1.0, help="internal rows per public row, per direction")
    a.add_argument("--repo", default="")
    a.add_argument("--epochs", type=float, default=20, help="upper bound; early stopping on dev chrF++ decides")
    a.add_argument("--lr", type=float, default=5e-4)
    a.add_argument("--subword-alpha", type=float, default=0.5,
                   help="SentencePiece sampling smoothing over the 64 best segmentations; 0 disables. "
                        "Do not use alpha 0.1 with nbest -1: it shreds words to near-characters and the model "
                        "learns to ignore its input (luso-mt v0, 2026-11-25).")
    a.add_argument("--batch", type=int, default=256)
    a.add_argument("--eval-steps", type=int, default=2000)
    a.add_argument("--seed", type=int, default=13)
    a.add_argument("--max-train", type=int, default=0, help="debug: limit training rows")
    a.add_argument("--smoke", action="store_true", help="10-minute end-to-end check on a small GPU")
    args = a.parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    cfg = dict(SIZES[args.size])
    if args.smoke:
        args.max_train, args.epochs, args.eval_steps, args.batch = args.max_train or 4000, 1, 30, 32
        cfg["vocab"] = 3000
    out_dir = f"luso-mt-{args.size}"
    os.makedirs(out_dir, exist_ok=True)

    train, dev, test = (load_split(args.data, s) for s in ("train", "dev", "test"))
    if args.smoke:
        dev, test = dev.groupby("direction").head(16), test.groupby("direction").head(16)
    if args.public_only:
        train = train[train.release == "public"]
    if args.kd:
        kd = pd.read_parquet(hf_hub_download(args.kd, "kd.parquet", repo_type="dataset"))
        kd = kd[kd.tgt_kd.fillna("") != ""]
        ratio = kd.tgt_kd.str.split().str.len() / kd.tgt.str.split().str.len().clip(lower=1)
        kd = kd[ratio.between(0.5, 2.0)]  # drop repeated or truncated teacher outputs
        kd = kd.assign(tgt=lambda d: d.tgt_kd, origin="kd").drop(columns="tgt_kd")
        if args.public_only:
            kd = kd[kd.release == "public"]
        train = pd.concat([train, kd[train.columns]], ignore_index=True)
    train = balance(train, args.seed, cap_internal=None if args.public_only else args.cap_internal)
    if args.max_train:
        train = train.head(args.max_train)
    print("training rows by direction:", train.groupby("direction").size().to_dict(), flush=True)

    sp_path = train_tokenizer(train, cfg["vocab"], out_dir, args.seed)
    tok = T5Tokenizer(sp_path, extra_ids=0, additional_special_tokens=TAGS, legacy=False)
    sampling = {"enable_sampling": True, "alpha": args.subword_alpha, "nbest_size": 64} if args.subword_alpha > 0 else {}
    tok_train = T5Tokenizer(sp_path, extra_ids=0, additional_special_tokens=TAGS, legacy=False, sp_model_kwargs=sampling)

    conf = T5Config(vocab_size=len(tok), d_model=cfg["d_model"], d_kv=cfg["d_kv"], d_ff=cfg["d_ff"],
                    num_layers=cfg["num_layers"], num_decoder_layers=cfg["num_decoder_layers"],
                    num_heads=cfg["num_heads"], feed_forward_proj="gated-gelu", dropout_rate=cfg["dropout"],
                    tie_word_embeddings=True, decoder_start_token_id=tok.pad_token_id,
                    pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    model = T5ForConditionalGeneration(conf)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"luso-mt-{args.size}: {n_params / 1e6:.1f}M parameters", flush=True)

    dev_small = dev.groupby("direction").head(200)
    steps_per_epoch = math.ceil(len(train) / args.batch)
    targs = Seq2SeqTrainingArguments(
        output_dir=os.path.join(out_dir, "ckpt"), per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=128, learning_rate=args.lr, lr_scheduler_type="inverse_sqrt",
        warmup_steps=min(4000, steps_per_epoch), num_train_epochs=args.epochs, weight_decay=0.01,
        label_smoothing_factor=0.1, max_grad_norm=1.0, bf16=torch.cuda.is_available(),
        eval_strategy="steps", eval_steps=args.eval_steps, save_strategy="steps", save_steps=args.eval_steps,
        save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="chrf", greater_is_better=True,
        predict_with_generate=True, generation_num_beams=1, generation_max_length=192, logging_steps=200,
        dataloader_num_workers=8, report_to=wandb_init(f"luso-mt-{args.size}", vars(args) | cfg | {"params_M": round(n_params / 1e6, 1)}),
        run_name=f"luso-mt-{args.size}" + ("-kd" if args.kd else "") + ("-public" if args.public_only else ""), seed=args.seed)

    def compute(pred):
        preds, labels = pred
        preds = [[t for t in p if t >= 0] for p in preds]
        hyps = tok.batch_decode(preds, skip_special_tokens=True)
        labels = [[t if t >= 0 else tok.pad_token_id for t in l] for l in labels]
        refs = tok.batch_decode(labels, skip_special_tokens=True)
        return {"chrf": chrf(hyps, refs)}

    trainer = Seq2SeqTrainer(
        model=model, args=targs, train_dataset=Pairs(train, tok_train), eval_dataset=Pairs(dev_small, tok),
        data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True), compute_metrics=compute,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)])
    trainer.train()

    model = trainer.model
    model.tie_weights()  # the best checkpoint stores only the shared embedding; re-tie encoder, decoder and output
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    api = HfApi()
    if args.repo:  # upload before evaluating, so a failing evaluation never loses the model
        api.create_repo(args.repo, private=True, exist_ok=True)
        api.upload_folder(folder_path=out_dir, repo_id=args.repo, ignore_patterns=["ckpt/*"],
                          commit_message=f"luso-mt-{args.size}: {n_params / 1e6:.0f}M parameters")
        print("pushed model", args.repo, flush=True)
    preds = []
    report = {"size": args.size, "params_M": round(n_params / 1e6, 1), "config": cfg,
              "train_rows": len(train), "public_only": args.public_only, "kd": bool(args.kd),
              "test": score_split(model, tok, test, preds=preds, name=f"luso-mt-{args.size}"),
              "dev": score_split(model, tok, dev)}
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(p, ensure_ascii=False) for p in preds))
    # memorisation check: the same metric on 500 training pairs per direction; a large gap to dev = memorising
    train_probe = train[train.origin == "human"].groupby("direction").head(16 if args.smoke else 500)
    report["train_probe"] = score_split(model, tok, train_probe)
    report["gap_train_minus_dev"] = {d: round(report["train_probe"][d]["chrF++"] - report["dev"][d]["chrF++"], 1)
                                     for d in report["dev"] if d in report["train_probe"]}
    json.dump(report, open(os.path.join(out_dir, "eval.json"), "w"), indent=1)
    print(json.dumps(report, indent=1), flush=True)
    wandb_log_final(report)

    if args.repo:
        for fn in ("eval.json", "predictions.jsonl"):
            api.upload_file(path_or_fileobj=os.path.join(out_dir, fn), path_in_repo=fn, repo_id=args.repo,
                            commit_message=f"luso-mt-{args.size}: test scores and translations")
        print("pushed eval.json and predictions.jsonl to", args.repo)


if __name__ == "__main__":
    main()
