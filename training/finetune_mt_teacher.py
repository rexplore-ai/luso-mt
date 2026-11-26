# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.46,<5", "sentencepiece", "sacrebleu>=2.4", "datasets>=3",
#                 "huggingface_hub", "accelerate", "protobuf", "pandas", "pyarrow", "wandb"]
# ///
"""Step 1 of luso-mt: transfer learning. Fine-tune a large pretrained translator on our pairs,
score it on the held-out sets, then translate the training sources for distillation (kd.parquet).

  --base google/madlad400-3b-mt       Apache-2.0, T5, target tag "<2xx>"  (safe to distil from)
  --base facebook/nllb-200-distilled-600M   CC-BY-NC: research teacher only, do not ship students from it

    hf jobs uv run --flavor h200 --namespace suru8-ai --timeout 10h --secrets HF_TOKEN --secrets WANDB_API_KEY \
        finetune_mt_teacher.py --base google/madlad400-3b-mt --data suru8-ai/luso-mt-data \
        --repo suru8-ai/luso-mt-teacher --kd-repo suru8-ai/luso-mt-data

Research run (all data incl. internal, continue from the teacher, no distillation, stop training after 8.3 h):

    hf jobs uv run --flavor h200 --namespace suru8-ai --timeout 10h --secrets HF_TOKEN --secrets WANDB_API_KEY         finetune_mt_teacher.py --base suru8-ai/luso-mt-teacher --repo suru8-ai/luso-mt-3b-research         --run-name luso-mt-3b-research --epochs 1 --batch 32 --grad-accum 4 --tum-copies 2 --time-budget-h 8.3

Logs to Weights & Biases (team rexplore-labs-ltd, project luso-mt) when WANDB_API_KEY is set.
"""
import argparse
import json
import os
import time

import pandas as pd
import sacrebleu
import torch
from huggingface_hub import HfApi, hf_hub_download
from torch.utils.data import Dataset
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, EarlyStoppingCallback,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback)

T0 = time.time()


class TimeBudget(TrainerCallback):
    """Stop training after a wall-clock budget, so upload and scoring still finish inside the job timeout."""

    def __init__(self, hours):
        self.limit = hours * 3600

    def on_step_end(self, args, state, control, **kw):
        if self.limit and time.time() - T0 > self.limit:
            print(f"time budget reached at step {state.global_step}: stopping training", flush=True)
            control.should_training_stop = True
        return control

NLLB = {"en": "eng_Latn", "ny": "nya_Latn", "tum": "tum_Latn"}


class Codec:
    """Hides the difference between MADLAD (<2xx> prefix) and NLLB (src_lang + forced BOS)."""

    def __init__(self, base, tok):
        self.nllb, self.tok = "nllb" in base, tok
        if not self.nllb:
            tum = tok.convert_tokens_to_ids("<2tum>")
            if tum == tok.unk_token_id:  # MADLAD has no Tumbuka tag: add one and learn it
                tok.add_tokens(["<2tum>"], special_tokens=True)

    def encode(self, srcs, src_langs, tgt_langs, max_len=192, targets=None):
        if self.nllb:
            rows = []
            for s, sl, tl, t in zip(srcs, src_langs, tgt_langs, targets or [None] * len(srcs)):
                self.tok.src_lang, self.tok.tgt_lang = NLLB[sl], NLLB[tl]
                rows.append(self.tok(s, text_target=t, max_length=max_len, truncation=True))
            return rows
        return [self.tok(f"<2{tl}> {s}", text_target=t, max_length=max_len, truncation=True)
                for s, tl, t in zip(srcs, tgt_langs, targets or [None] * len(srcs))]

    @torch.inference_mode()
    def translate(self, model, srcs, src_lang, tgt_lang, bs=64, beams=4):
        model.eval()
        out = []
        for i in range(0, len(srcs), bs):
            chunk = srcs[i:i + bs]
            if self.nllb:
                self.tok.src_lang = NLLB[src_lang]
                enc = self.tok(chunk, return_tensors="pt", padding=True, truncation=True, max_length=192).to(model.device)
                kw = {"forced_bos_token_id": self.tok.convert_tokens_to_ids(NLLB[tgt_lang])}
            else:
                enc = self.tok([f"<2{tgt_lang}> {s}" for s in chunk], return_tensors="pt", padding=True,
                               truncation=True, max_length=192).to(model.device)
                kw = {}
            ids = model.generate(**enc, num_beams=beams, max_new_tokens=192, **kw)
            out += self.tok.batch_decode(ids, skip_special_tokens=True)
        return out


class Pairs(Dataset):
    def __init__(self, df, codec):
        self.rows = codec.encode(list(df.src), list(df.src_lang), list(df.tgt_lang), targets=list(df.tgt))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return {"input_ids": r["input_ids"], "attention_mask": r["attention_mask"], "labels": r["labels"]}


def wandb_init(name, config):
    """W&B under the rexplore-labs-ltd team; off when no key is available."""
    if not os.environ.get("WANDB_API_KEY"):
        return "none"
    os.environ.setdefault("WANDB_ENTITY", "rexplore-labs-ltd")
    os.environ.setdefault("WANDB_PROJECT", "luso-mt")
    os.environ.setdefault("WANDB_LOG_MODEL", "false")
    import wandb
    wandb.init(entity=os.environ["WANDB_ENTITY"], project=os.environ["WANDB_PROJECT"], name=name,
               config=json.loads(json.dumps(config, default=str)), tags=["luso-mt", "teacher"])
    return "wandb"


def wandb_log(prefix, res):
    try:
        import wandb
        if wandb.run is not None:
            flat = {f"{prefix}/{d}/{k}": v for d, m in res.items() for k, v in m.items() if k != "n"}
            wandb.log(flat)
            wandb.summary.update(flat)
    except Exception as e:
        print("wandb log skipped:", e)


def score(codec, model, df, preds=None, name=None):
    res = {}
    for d, g in df.groupby("direction"):
        hyps = codec.translate(model, list(g.src), g.src_lang.iloc[0], g.tgt_lang.iloc[0])
        res[d] = {"chrF++": round(sacrebleu.corpus_chrf(hyps, [list(g.tgt)], word_order=2).score, 1),
                  "BLEU": round(sacrebleu.corpus_bleu(hyps, [list(g.tgt)]).score, 1), "n": len(g)}
        if preds is not None:
            preds += [{"model": name, "direction": d, "src": s, "ref": r, "hyp": h} for s, r, h in zip(g.src, g.tgt, hyps)]
        print(d, res[d], flush=True)
    return res


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--base", default="google/madlad400-3b-mt")
    a.add_argument("--data", default="suru8-ai/luso-mt-data")
    a.add_argument("--repo", default="suru8-ai/luso-mt-teacher")
    a.add_argument("--kd-repo", default="", help="push kd.parquet (teacher translations of train sources) here")
    a.add_argument("--kd-max", type=int, default=400_000, help="sources to distil (public first, all Tumbuka)")
    a.add_argument("--public-only", action="store_true")
    a.add_argument("--epochs", type=float, default=2)
    a.add_argument("--lr", type=float, default=1e-4)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--grad-accum", type=int, default=2)
    a.add_argument("--max-train", type=int, default=0)
    a.add_argument("--tum-copies", type=int, default=3, help="times each Tumbuka pair appears in training")
    a.add_argument("--eval-steps", type=int, default=0, help="eval/save interval (default: about 10 per run)")
    a.add_argument("--time-budget-h", type=float, default=0, help="stop training after this many hours since start")
    a.add_argument("--run-name", default="")
    a.add_argument("--smoke", action="store_true", help="quick end-to-end check: tiny data, short scoring, no KD push")
    args = a.parse_args()

    split = {s: pd.read_parquet(hf_hub_download(args.data, f"{s}.parquet", repo_type="dataset")) for s in ("train", "dev", "test")}
    if args.smoke:
        args.max_train, args.epochs, args.kd_max = args.max_train or 256, 1, 64
        split["dev"] = split["dev"].groupby("direction").head(16)
        split["test"] = split["test"].groupby("direction").head(16)
    train = split["train"]
    if args.public_only:
        train = train[train.release == "public"]
    # Tumbuka is ~5x smaller: repeat it so the teacher learns it, and keep human pairs twice
    tum = train[(train.src_lang == "tum") | (train.tgt_lang == "tum")]
    train = pd.concat([train] + [tum] * (args.tum_copies - 1) + [train[train.origin == "human"]]).sample(frac=1, random_state=13)
    if args.max_train:
        train = train.head(args.max_train)
    print("teacher training rows:", train.groupby("direction").size().to_dict(), flush=True)

    tok = AutoTokenizer.from_pretrained(args.base)
    codec = Codec(args.base, tok)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model.resize_token_embeddings(len(tok))
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    before = score(codec, model.cuda(), split["test"].groupby("direction").head(200))
    run_name = args.run_name or f"luso-mt-teacher-{args.base.split('/')[-1]}"
    steps = int(len(train) * args.epochs / (args.batch * args.grad_accum))
    every = args.eval_steps or max(50, steps // 10 // 50 * 50)
    print(f"optimizer steps: {steps}, eval/save every {every}", flush=True)
    targs = Seq2SeqTrainingArguments(
        output_dir="teacher_ckpt", per_device_train_batch_size=args.batch, gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03, num_train_epochs=args.epochs,
        weight_decay=0.01, label_smoothing_factor=0.1, bf16=True, optim="adafactor",
        eval_strategy="steps", eval_steps=every, save_strategy="steps", save_steps=every, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", logging_steps=100,
        per_device_eval_batch_size=64, dataloader_num_workers=4,
        report_to=wandb_init(run_name, vars(args) | {"zero_shot_test_200": before, "train_rows": len(train)}),
        run_name=run_name)
    trainer = Seq2SeqTrainer(model=model, args=targs, train_dataset=Pairs(train, codec),
                             eval_dataset=Pairs(split["dev"], codec),
                             data_collator=DataCollatorForSeq2Seq(tok, model=model, padding=True),
                             callbacks=[EarlyStoppingCallback(early_stopping_patience=3), TimeBudget(args.time_budget_h)])
    trainer.train()
    model = trainer.model
    model.config.use_cache = True
    # upload before the (long) evaluation, so a failing evaluation never loses the trained model
    os.makedirs("teacher", exist_ok=True)
    model.save_pretrained("teacher")
    tok.save_pretrained("teacher")
    api = HfApi()
    api.create_repo(args.repo, private=True, exist_ok=True)
    api.upload_folder(folder_path="teacher", repo_id=args.repo, commit_message=f"luso-mt teacher from {args.base}")
    print("pushed", args.repo, f"({(time.time() - T0) / 3600:.2f} h since start)", flush=True)

    preds = []
    probe = split["train"].groupby("direction").sample(n=16 if args.smoke else 300, random_state=7)
    report = {"base": args.base, "zero_shot_test_200": before,
              "test": score(codec, model, split["test"], preds=preds, name=run_name),
              "dev": score(codec, model, split["dev"]),
              "train_probe": score(codec, model, probe),
              "steps": trainer.state.global_step, "hours": round((time.time() - T0) / 3600, 2)}
    report["gap_train_minus_dev"] = {d: round(report["train_probe"][d]["chrF++"] - report["dev"][d]["chrF++"], 1)
                                     for d in report["dev"] if d in report["train_probe"]}
    wandb_log("final/test", report["test"])
    wandb_log("final/dev", report["dev"])
    json.dump(report, open("teacher/eval.json", "w"), indent=1)
    with open("teacher/predictions.jsonl", "w", encoding="utf-8") as f:
        f.write("\n".join(json.dumps(p, ensure_ascii=False) for p in preds))
    for fn in ("eval.json", "predictions.jsonl"):
        api.upload_file(path_or_fileobj=f"teacher/{fn}", path_in_repo=fn, repo_id=args.repo,
                        commit_message="Teacher scores and test translations")
    print(json.dumps(report, indent=1), flush=True)

    if args.kd_repo:
        src = split["train"]
        if args.public_only:  # a public student must not learn from restricted sources
            src = src[src.release == "public"]
        src = src.assign(_p=((src.release != "public") & (src.src_lang != "tum") & (src.tgt_lang != "tum")).astype(int))
        src = src.sort_values("_p").head(args.kd_max).drop(columns="_p")
        outs = []
        for d, g in src.groupby("direction"):
            print(f"distilling {d}: {len(g)}", flush=True)
            outs.append(g.assign(tgt_kd=codec.translate(model, list(g.src), g.src_lang.iloc[0], g.tgt_lang.iloc[0], bs=128)))
        kd = pd.concat(outs)
        kd.to_parquet("kd.parquet", index=False)
        api.create_repo(args.kd_repo, repo_type="dataset", private=True, exist_ok=True)
        api.upload_file(path_or_fileobj="kd.parquet", path_in_repo="kd.parquet", repo_id=args.kd_repo,
                        repo_type="dataset", commit_message=f"teacher translations for distillation ({args.base})")
        print("pushed kd.parquet", len(kd), flush=True)


if __name__ == "__main__":
    main()
