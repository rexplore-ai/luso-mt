# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.46,<5", "sentencepiece", "protobuf", "pandas", "pyarrow", "sacrebleu>=2.4", "huggingface_hub"]
# ///
"""Translation quality of the released students after int8 dynamic quantisation, on the test sets, on CPU.

    hf jobs uv run --flavor cpu-upgrade --namespace suru8-ai --secrets HF_TOKEN eval_int8.py

Compares fp32 and int8 (PyTorch dynamic quantisation of linear layers) chrF++/BLEU per direction, beam 4, and
writes build/int8.json to suru8-ai/luso-mt-paper.
"""
import json, os, time
import pandas as pd
import sacrebleu
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def cpu_quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            return max(1, int(int(q) / int(p)))
    except (OSError, ValueError):
        pass
    return len(os.sched_getaffinity(0))


torch.set_num_threads(cpu_quota())
test = pd.read_parquet(hf_hub_download("suru8-ai/luso-mt-data", "test.parquet", repo_type="dataset"))


@torch.inference_mode()
def translate(model, tok, srcs, tgt, bs=16):
    out = []
    for i in range(0, len(srcs), bs):
        enc = tok([f"<2{tgt}> {s}" for s in srcs[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=192)
        out += tok.batch_decode(model.generate(**enc, num_beams=4, max_new_tokens=192), skip_special_tokens=True)
    return out


res = {"threads": torch.get_num_threads()}
for name in ("suru8-ai/luso-mt-tiny", "suru8-ai/luso-mt-small"):
    tok = AutoTokenizer.from_pretrained(name)
    fp32 = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
    int8 = torch.quantization.quantize_dynamic(fp32, {torch.nn.Linear}, dtype=torch.qint8)
    res[name] = {}
    for d, g in test.groupby("direction"):
        row = {}
        for label, m in (("fp32", fp32), ("int8", int8)):
            t0 = time.time()
            hyps = translate(m, tok, list(g.src), g.tgt_lang.iloc[0])
            row[label] = {"chrF++": round(sacrebleu.corpus_chrf(hyps, [list(g.tgt)], word_order=2).score, 1),
                          "BLEU": round(sacrebleu.corpus_bleu(hyps, [list(g.tgt)]).score, 1),
                          "seconds": round(time.time() - t0, 1)}
        row["delta_chrF++"] = round(row["int8"]["chrF++"] - row["fp32"]["chrF++"], 1)
        res[name][d] = row
        print(name, d, row, flush=True)
HfApi().upload_file(path_or_fileobj=json.dumps(res, indent=1).encode(), path_in_repo="build/int8.json",
                    repo_id="suru8-ai/luso-mt-paper", repo_type="dataset", commit_message="int8 quality")
print("uploaded int8.json", flush=True)
