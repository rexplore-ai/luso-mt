"""Generate the data-driven LaTeX tables from data/ (run from paper/latex): python scripts/make_tables.py

data/results.json (compute_results.py): scores, 95% CIs and paired-bootstrap p-values.
data/ablation_significance.json: distilled vs non-distilled students.  data/fertility.json: tokeniser statistics.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA, TAB = ROOT / "data", ROOT / "tables"
TAB.mkdir(exist_ok=True)
D = ["en->ny", "ny->en", "en->tum", "tum->en"]
HEAD = r"\enny & \nyen & \entum & \tumen"
R = json.loads((DATA / "results.json").read_text(encoding="utf-8"))
sig = R["significance"]["luso-mt-small"]
abl = json.loads((DATA / "ablation_significance.json").read_text(encoding="utf-8"))
fert = json.loads((DATA / "fertility.json").read_text(encoding="utf-8"))


def cell(m, d, best):
    r = sig[d].get(m)
    if not r:
        return r"\textit{n/s}"
    c, b = r["chrF++"], r["BLEU"]
    v = f"{c['score']:.1f}"
    v = rf"\textbf{{{v}}}" if m in best else v
    return rf"{v}\,{{\scriptsize$\pm${c['ci95']:.1f}}} / {b['score']:.1f}"


def best_of(group, d):
    vals = {m: sig[d][m]["chrF++"]["score"] for m in group if m in sig[d]}
    top = max(vals.values())
    return {m for m, v in vals.items() if v == top}


small = [("luso-mt-small", "88M", "Apache-2.0"), ("luso-mt-tiny", "20M", "Apache-2.0"), ("OPUS-MT", "75M", "CC-BY-4.0")]
large = [("luso-mt-3b", "2.9B", "to be released"), ("luso-mt-teacher", "2.9B", "not released"),
         ("nllb-200-distilled-600M", "615M", "CC-BY-NC-4.0"), ("madlad400-3b-mt", "2.9B", "Apache-2.0")]
label = {"luso-mt-small": "luso-mt-small", "luso-mt-tiny": "luso-mt-tiny", "OPUS-MT": "OPUS-MT (per pair)",
         "luso-mt-3b": "luso-mt-3b", "luso-mt-teacher": "Teacher (public data)",
         "nllb-200-distilled-600M": "NLLB-200 distilled", "madlad400-3b-mt": "MADLAD-400 3B"}


def block(rows):
    best = {d: best_of([m for m, _, _ in rows], d) for d in D}
    return "\n".join(f"{label[m]} & {p} & " + " & ".join(cell(m, d, best[d]) for d in D) + f" & {l} \\\\" for m, p, l in rows)


(TAB / "main-results.tex").write_text(r"""\begin{table*}[t]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\caption{Test results: \chrf{} with 95\% bootstrap confidence interval / BLEU, 500 sentences per direction, beam size 4.
Bold: best in each group. Every difference between luso-mt-small and any other system is significant (paired
bootstrap, 1{,}000 resamples, $p \le 0.007$). luso-mt-3b is significantly better than NLLB-200 in \enny, \entum{} and
\tumen{} ($p \le 0.002$); the difference in \nyen{} is not significant ($p = 0.19$). \textit{n/s}: not supported.}
\label{tab:main}
\begin{tabular}{@{}llccccl@{}}
\toprule
Model & Params & """ + HEAD + r""" & Licence \\
\midrule
\multicolumn{7}{@{}l}{\textit{Phone-sized (under 100M parameters)}} \\
""" + block(small) + r"""
\midrule
\multicolumn{7}{@{}l}{\textit{Large}} \\
""" + block(large) + r"""
\bottomrule
\end{tabular}
\end{table*}
""", encoding="utf-8")

arows = []
for sz in ("tiny", "small"):
    a = abl[sz]
    arows.append(f"luso-mt-{sz}, no distillation & " + " & ".join(f"{a[d]['baseline']:.1f}" for d in D) + r" \\")
    arows.append(f"luso-mt-{sz}, distilled & " + " & ".join(f"\\textbf{{{a[d]['distilled']:.1f}}}" for d in D) + r" \\")
    arows.append(r"\quad gain & " + " & ".join(f"+{a[d]['delta']:.1f}$^{{\\dagger}}$" for d in D) + r" \\" + (r" \midrule" if sz == "tiny" else ""))
(TAB / "ablation.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Distillation ablation: identical students trained with and without the teacher's translations (test \chrf).
$^{\dagger}$\,$p = 0.001$, the smallest value a 1{,}000-sample paired bootstrap can report.}
\label{tab:ablation}
\begin{tabular}{@{}lcccc@{}}
\toprule
Student & """ + HEAD + r""" \\
\midrule
""" + "\n".join(arows) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")

frows = []
for name, v in fert.items():
    n = rf"\textbf{{{name}}}" if name.startswith("luso-mt-small") else name
    frows.append(f"{n} & {v['en']['fertility']:.2f} & {v['ny']['fertility']:.2f} & {v['tum']['fertility']:.2f} & "
                 f"{round(v['ny']['split_word_share']*100)}\\% & {round(v['tum']['split_word_share']*100)}\\% \\\\")
(TAB / "fertility.tex").write_text(r"""\begin{table}[t]
\centering\small
\caption{Tokeniser fertility (subword tokens per word) on the test sets (17{,}295 English, 9{,}409 Chichewa and 7{,}115
Tumbuka words) and share of words split into two or more pieces. Lower is better.}
\label{tab:fertility}
\begin{tabular}{@{}lccccc@{}}
\toprule
Tokeniser (vocabulary) & English & Chichewa & Tumbuka & Split ny & Split tum \\
\midrule
""" + "\n".join(frows) + r"""
\bottomrule
\end{tabular}
\end{table}
""", encoding="utf-8")
print("wrote", sorted(p.name for p in TAB.glob("*.tex")))
