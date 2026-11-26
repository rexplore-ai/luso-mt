"""Build the public luso-mt benchmark files from test-set predictions (scores recomputed with sacreBLEU)."""
import json, os, shutil, sys
import pandas as pd, sacrebleu
from huggingface_hub import hf_hub_download

root = sys.argv[1]; raw, out = f"{root}/raw", f"{root}/out"
os.makedirs(f"{out}/data", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True)
shutil.copy(hf_hub_download("suru8-ai/luso-mt-teacher", "predictions.jsonl"), f"{raw}/teacher.predictions.jsonl")
parts = [pd.read_json(f"{raw}/{f}", lines=True) for f in
         ["luso-mt-small.predictions.jsonl", "luso-mt-tiny.predictions.jsonl", "teacher.predictions.jsonl", "predictions.jsonl"]]
p = pd.concat(parts, ignore_index=True)
print(p.model.value_counts().to_dict())
HF = {"luso-mt-small": "suru8-ai/luso-mt-small", "luso-mt-tiny": "suru8-ai/luso-mt-tiny",
      "nllb-200-distilled-600M": "facebook/nllb-200-distilled-600M", "madlad400-3b-mt": "google/madlad400-3b-mt",
      "opus-mt-en-ny": "Helsinki-NLP/opus-mt-en-ny", "opus-mt-ny-en": "Helsinki-NLP/opus-mt-ny-en",
      "opus-mt-tum-en": "Helsinki-NLP/opus-mt-tum-en", "opus-mt-mul-en": "Helsinki-NLP/opus-mt-mul-en",
      "opus-mt-en-mul": "Helsinki-NLP/opus-mt-en-mul"}
teacher = [m for m in p.model.unique() if m not in HF]; assert len(teacher) == 1, teacher
p["model"] = p.model.replace({teacher[0]: "luso-mt-teacher"})
HF["luso-mt-teacher"] = "suru8-ai/luso-mt-teacher (fine-tuned google/madlad400-3b-mt, not released)"
p["checkpoint"] = p.model.map(HF)
p["test_set"] = p.direction.map(lambda d: "NTREX-128" if "ny" in d else "SMOL")
p["id"] = p.groupby(["model", "direction"]).cumcount()
p = p[["model", "checkpoint", "direction", "test_set", "id", "src", "ref", "hyp"]]
p.to_parquet(f"{out}/data/predictions.parquet", index=False)
chrf, bleu = sacrebleu.CHRF(word_order=2), sacrebleu.BLEU()
rows = []
for (m, d), g in p.groupby(["model", "direction"], sort=False):
    rows.append(dict(model=m, checkpoint=HF[m], direction=d, test_set=g.test_set.iloc[0], n=len(g),
                     chrf_pp=round(chrf.corpus_score(g.hyp.fillna("").tolist(), [g.ref.tolist()]).score, 1),
                     bleu=round(bleu.corpus_score(g.hyp.fillna("").tolist(), [g.ref.tolist()]).score, 1)))
s = pd.DataFrame(rows); s.to_csv(f"{out}/data/scores.csv", index=False)
print(s.pivot(index="model", columns="direction", values="chrf_pp").to_string())
b = json.load(open(sys.argv[2], encoding="utf-8"))
speed = [dict(model=k.split("/")[-1].replace("-kd", ""), threads=b["cpu_threads"], params_M=v["params_M"], fp32_MB=v["fp32_MB"],
              int8_MB=v["int8_MB"], cpu_sent_per_s_greedy=v["cpu_sent_per_s_greedy"], cpu_sent_per_s_beam4=v["cpu_sent_per_s_beam4"],
              cpu_sent_per_s_int8_greedy=v["cpu_sent_per_s_int8_greedy"]) for k, v in b["models"].items()]
pd.DataFrame(speed).to_csv(f"{out}/data/speed.csv", index=False)
for f in ["score_matrix.png", "slice_matrix.png", "language_confusion.png", "win_matrix.png"]:
    shutil.copy(f"{sys.argv[3]}/{f}", f"{out}/figures/{f}")
print("built", os.listdir(f"{out}/data"))
