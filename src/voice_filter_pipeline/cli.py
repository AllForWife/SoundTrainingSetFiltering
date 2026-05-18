from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import Pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Filter female training audio by content only.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, default=Path.cwd(), help="Project root containing clips/.")
    common.add_argument("--limit", type=int, default=None, help="Process at most N files/rows.")
    common.add_argument("--seed", type=int, default=7, help="Random seed for pilot sampling.")
    common.add_argument("--random", action="store_true", help="Randomly sample when --limit is used.")
    common.add_argument("--workers", type=int, default=None, help="Parallel workers for non-UVR stages. Default: auto.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("scan", "classify", "score", "verify", "filter-only"):
        sub.add_parser(name, parents=[common])
    run_all = sub.add_parser("run-all", parents=[common])
    run_all.add_argument("--backend", default="auto", help="auto, audio-separator, or demucs.")
    separate = sub.add_parser("separate", parents=[common])
    separate.add_argument("--backend", default="auto", help="auto, audio-separator, or demucs.")
    uvr_files = sub.add_parser("uvr-files", parents=[common])
    uvr_files.add_argument("files", nargs="+", type=Path, help="Specific files to clean with UVR.")
    uvr_files.add_argument("--backend", default="auto", help="auto, audio-separator, or demucs.")
    uvr_folder = sub.add_parser("uvr-folder", parents=[common])
    uvr_folder.add_argument("folder", type=Path, help="Folder whose audio files should be cleaned with UVR.")
    uvr_folder.add_argument("--backend", default="auto", help="auto, audio-separator, or demucs.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = Pipeline(args.root, max_workers=args.workers)
    common = {"limit": args.limit, "seed": args.seed, "randomize": args.random}
    if args.command == "scan":
        result = pipeline.scan(**common)
    elif args.command == "classify":
        result = pipeline.classify(**common)
    elif args.command == "score":
        result = pipeline.score(**common)
    elif args.command == "separate":
        result = pipeline.separate(**common, backend=args.backend)
    elif args.command == "verify":
        result = pipeline.verify(**common)
    elif args.command == "filter-only":
        result = pipeline.filter_only(**common)
    elif args.command == "run-all":
        result = pipeline.run_all(**common, backend=args.backend)
    elif args.command == "uvr-files":
        result = pipeline.uvr_selected_files(args.files, backend=args.backend)
    elif args.command == "uvr-folder":
        result = pipeline.uvr_folder(args.folder, **common, backend=args.backend)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
