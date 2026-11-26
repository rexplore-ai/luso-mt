# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub", "requests"]
# ///
"""Compile the LaTeX report on a Hugging Face Jobs CPU machine (no local TeX needed).

    hf jobs uv run --flavor cpu-upgrade --namespace suru8-ai --secrets HF_TOKEN compile_on_hf.py

Expects build/latex.tar.gz in the private dataset suru8-ai/luso-mt-paper; writes build/main.pdf and
build/compile.log back to it. Uses Tectonic (a self-contained TeX engine that fetches packages on demand).
"""
import io, os, subprocess, tarfile
import requests
from huggingface_hub import HfApi, hf_hub_download

REPO = "suru8-ai/luso-mt-paper"
api = HfApi()
work = "/tmp/paper"
os.makedirs(work, exist_ok=True)
tarfile.open(hf_hub_download(REPO, "build/latex.tar.gz", repo_type="dataset")).extractall(work)

# pinned Tectonic release (the version this report was verified with); the GitHub API is rate-limited anonymously
PINNED = ("https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.17.0/"
          "tectonic-0.17.0-x86_64-unknown-linux-musl.tar.gz")
resp = requests.get(PINNED, timeout=300)
if resp.status_code != 200:
    rel = requests.get("https://api.github.com/repos/tectonic-typesetting/tectonic/releases/latest", timeout=60).json()
    url = next(a["browser_download_url"] for a in rel["assets"] if a["name"].endswith("x86_64-unknown-linux-musl.tar.gz"))
    resp = requests.get(url, timeout=300)
print("tectonic:", resp.url.rsplit("/", 1)[-1], resp.status_code, flush=True)
tarfile.open(fileobj=io.BytesIO(resp.content)).extractall("/tmp/tt")
tectonic = next(os.path.join(r, f) for r, _, fs in os.walk("/tmp/tt") for f in fs if f == "tectonic")
os.chmod(tectonic, 0o755)

p = subprocess.run([tectonic, "-X", "compile", "--keep-logs", "main.tex"], cwd=work, capture_output=True, text=True)
log = p.stdout + "\n" + p.stderr
for extra in ("main.log", "main.blg"):
    fp = os.path.join(work, extra)
    if os.path.exists(fp):
        log += f"\n===== {extra} =====\n" + open(fp, encoding="utf-8", errors="replace").read()
print(log[-6000:], flush=True)
api.upload_file(path_or_fileobj=log.encode(), path_in_repo="build/compile.log", repo_id=REPO, repo_type="dataset",
                commit_message="compile log")
pdf = os.path.join(work, "main.pdf")
if os.path.exists(pdf):
    api.upload_file(path_or_fileobj=pdf, path_in_repo="build/main.pdf", repo_id=REPO, repo_type="dataset",
                    commit_message="compiled report")
    print("uploaded main.pdf", os.path.getsize(pdf), flush=True)
else:
    print("NO PDF; exit code", p.returncode, flush=True)
