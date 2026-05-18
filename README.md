# Voice Filter Pipeline

This project filters `clips/` by audio content only. File names and external
metadata are ignored.

## Commands

```powershell
python .\voice_filter.py scan --limit 1000 --random
python .\voice_filter.py classify --limit 1000 --random
python .\voice_filter.py score --limit 1000 --random
python .\voice_filter.py separate --limit 1000 --random
python .\voice_filter.py verify --limit 1000 --random
python .\voice_filter.py run-all --limit 1000 --random
```

Add `--workers N` to speed up non-UVR stages. The default is automatic and
bounded to at most 8 workers:

```powershell
python .\voice_filter.py filter-only --workers 8
```

UVR remains controlled separately because running many UVR model jobs at once
can exhaust GPU memory.

Outputs are written under `work/` and `out/`. The source `clips/` folder is
never modified.

## UVR backend

`separate` prefers an installed CLI backend:

- `audio-separator`
- `demucs`

If neither is available, matching files are marked with `reject_reason:
uvr_backend_missing` rather than pretending background removal succeeded.

## Final training set

Final accepted clips are written to:

```text
out/female_clean_trainable/
```

The complete decision trail is:

```text
work/manifests/master.jsonl
```

## Practical first run

Start with a filtering-only pilot. This does not run UVR:

```powershell
python .\voice_filter.py filter-only --limit 1000 --random
```

Then inspect:

```powershell
Get-Content .\out\report.md
```

For full filtering, omit `--limit` only after the pilot looks reasonable:

```powershell
python .\voice_filter.py filter-only --random
```

The command is resumable because each stage reads and rewrites
`work/manifests/master.jsonl`. Progress is checkpointed every 500 processed
items, and existing files under `out/` are used to skip work that already
finished.

## Install UVR backend

The classifier and scorer are intentionally separate from UVR. To clean every
audio file inside a chosen folder, select the folder in the GUI or run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_uvr_env.ps1
$env:VOICE_FILTER_UVR_PYTHON = (Resolve-Path .\.venv-uvr\Scripts\python.exe).Path
$env:VOICE_FILTER_UVR_MODEL = "MDX23C-8KFFT-InstVoc_HQ.ckpt"
python .\voice_filter.py uvr-folder .\out\female_raw_pass --backend audio-separator
```

The Python 3.11 `.venv-uvr` path is intentional: the default Python 3.14 can run
the pipeline, while the UVR package stack runs in a compatibility environment.

The verified pilot used `MDX23C-8KFFT-InstVoc_HQ.ckpt`, downloaded under
`D:\tmp\audio-separator-models\`.
