# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "transformers>=4.46,<5", "sentencepiece", "protobuf", "pandas", "pyarrow", "huggingface_hub"]
# ///
"""Size, CPU speed and example translations for luso-mt checkpoints (numbers for model cards and posts).

For each model: parameter count, fp32 and int8 (dynamic quantisation of Linear layers) file size,
CPU throughput (sentences/second, beam 4 and greedy, batch 1) on test sentences, and translations of
fixed example inputs. Runs on a CPU flavour so the speed reflects devices without a GPU.

    hf jobs uv run --flavor cpu-upgrade --namespace suru8-ai --timeout 1h --secrets HF_TOKEN \
        benchmark_mt.py --models suru8-ai/luso-mt-tiny suru8-ai/luso-mt-small --out suru8-ai/luso-mt-eval
"""
import argparse
import io
import json
import os
import time

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

EXAMPLES = [
    ("ny", "Where is the nearest clinic?"),
    ("en", "Kodi chipatala chapafupi chili kuti?"),
    ("tum", "The rains came early this year."),
    ("ny", "Please wash your hands before eating."),
    ("tum", "Where is the nearest clinic?"),
]


def cpu_quota():
    """vCPUs this container may use; os.cpu_count() reports the host and oversubscribes threads."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def size_mb(state_dict):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    return round(buf.tell() / 1e6, 1)


@torch.inference_mode()
def throughput(model, tok, texts, tgt, beams):
    t0 = time.time()
    for t in texts:
        enc = tok(f"<2{tgt}> {t}", return_tensors="pt")
        model.generate(**enc, num_beams=beams, max_new_tokens=128)
    return round(len(texts) / (time.time() - t0), 2)


@torch.inference_mode()
def translate(model, tok, text, tgt):
    enc = tok(f"<2{tgt}> {text}", return_tensors="pt")
    return tok.decode(model.generate(**enc, num_beams=4, max_new_tokens=128)[0], skip_special_tokens=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--models", nargs="+", required=True)
    a.add_argument("--data", default="suru8-ai/luso-mt-data")
    a.add_argument("--n", type=int, default=60, help="test sentences per speed measurement")
    a.add_argument("--out", default="suru8-ai/luso-mt-eval")
    args = a.parse_args()
    threads = cpu_quota()
    torch.set_num_threads(threads)
    print(f"threads={threads} (os.cpu_count()={os.cpu_count()})", flush=True)
    test = pd.read_parquet(hf_hub_download(args.data, "test.parquet", repo_type="dataset"))
    tum_example = test[(test.direction == "tum->en")].src.iloc[3]
    examples = EXAMPLES + [("en", tum_example)]
    srcs = list(test[test.direction == "en->ny"].src.head(args.n))
    results = {"cpu_threads": threads, "models": {}}
    for name in args.models:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSeq2SeqLM.from_pretrained(name).eval()
        q = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(name, "loaded", flush=True)
        r = {
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
            "fp32_MB": size_mb(model.state_dict()),
            "int8_MB": size_mb(q.state_dict()),
            "cpu_sent_per_s_beam4": throughput(model, tok, srcs, "ny", 4),
            "cpu_sent_per_s_greedy": throughput(model, tok, srcs, "ny", 1),
            "cpu_sent_per_s_int8_greedy": throughput(q, tok, srcs, "ny", 1),
            "examples": [{"to": tgt, "src": s, "out": translate(model, tok, s, tgt)} for tgt, s in examples],
        }
        results["models"][name] = r
        print(json.dumps({name: r}, ensure_ascii=False, indent=1), flush=True)
    api = HfApi()
    api.upload_file(path_or_fileobj=json.dumps(results, ensure_ascii=False, indent=1).encode(), path_in_repo="benchmark.json",
                    repo_id=args.out, repo_type="dataset", commit_message="CPU speed, size and example translations")
    print("uploaded benchmark.json")


if __name__ == "__main__":
    main()
