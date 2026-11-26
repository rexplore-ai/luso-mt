# /// script
# requires-python = ">=3.10"
# dependencies = ["hydra-core>=1.3", "omegaconf>=2.3"]
# ///
"""Submit luso-mt training runs to Hugging Face Jobs from Hydra/OmegaConf configs.

    uv run launch.py experiment=tiny_kd seed=14
    uv run launch.py -m experiment=tiny_kd seed=13,14,15
    uv run launch.py -m experiment=tiny_kd,tiny_base,tiny_rep2 seed=13,14,15 dry_run=true

Each run becomes one detached job: `hf jobs uv run ... <script> --seed <seed> --repo <repo_prefix>-s<seed> ...`.
The training script keeps plain command-line arguments, so it also runs on its own.
"""
import shlex
import subprocess
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

ROOT = Path(__file__).resolve().parent


def to_cli(args: DictConfig) -> list[str]:
    out = []
    for key, value in args.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                out.append(flag)
        elif value is not None:
            out += [flag, str(value)]
    return out


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    exp = cfg.experiment
    script = ROOT / exp.script
    cmd = ["hf", "jobs", "uv", "run", "--flavor", cfg.hf.flavor, "--namespace", cfg.hf.namespace,
           "--timeout", str(cfg.hf.timeout), "--detach"]
    for secret in cfg.hf.secrets:
        cmd += ["--secrets", secret]
    cmd += [str(script), *to_cli(exp.args), "--seed", str(cfg.seed), "--repo", f"{exp.repo_prefix}-s{cfg.seed}"]
    print(OmegaConf.to_yaml(cfg.experiment), flush=True)
    print(" ".join(shlex.quote(c) for c in cmd), flush=True)
    if not cfg.dry_run:
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
