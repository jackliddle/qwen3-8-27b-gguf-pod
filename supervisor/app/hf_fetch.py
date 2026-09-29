"""Child-process entry point for model downloads (see engines.base.hf_download)."""

import argparse

from huggingface_hub import snapshot_download


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--revision")
    p.add_argument("--pattern", action="append", default=[])
    a = p.parse_args()
    path = snapshot_download(
        a.repo,
        revision=a.revision,
        local_dir=a.dir,
        allow_patterns=a.pattern or None,
    )
    print(f"Fetched into {path}", flush=True)


if __name__ == "__main__":
    main()
