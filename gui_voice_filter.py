from __future__ import annotations

import argparse
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, END, StringVar, Tk, filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText


def app_root() -> Path:
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        if exe_dir.name.lower() == "dist" and (exe_dir.parent / ".venv-uvr").exists():
            return exe_dir.parent
        return exe_dir
    return Path(__file__).resolve().parent


ROOT = app_root()
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from voice_filter_pipeline.pipeline import Pipeline, resolve_uvr_batch_size, resolve_worker_count


STAGE_LABELS = {
    "掃描音檔": "scan",
    "分類性別": "classify",
    "女聲音質評分": "score",
    "只做篩選": "filter-only",
    "驗證既有 UVR 輸出": "verify",
    "對資料夾做 UVR": "uvr-folder",
}

STAGE_BY_VALUE = {value: label for label, value in STAGE_LABELS.items()}


PROGRESS_STAGE_LABELS = {
    "scan": "掃描音檔",
    "classify": "分類性別",
    "score": "女聲音質評分",
    "separate": "UVR 去背",
    "verify": "驗證 UVR 輸出",
    "manual_uvr": "資料夾 UVR",
}


def parse_limit(value: str) -> int | None:
    value = value.strip()
    if not value:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("處理數量必須是正整數")
    return parsed


def resolve_limit(value: str, process_all: bool) -> int | None:
    if process_all:
        return None
    return parse_limit(value)


def parse_workers(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("平行 workers 必須是正整數，或輸入 auto")
    return parsed


def parse_uvr_batch_size(value: str) -> int | None:
    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("UVR 批次大小必須是正整數，或輸入 auto")
    return parsed


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "估算中"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours} 小時 {minutes} 分"
    if minutes:
        return f"{minutes} 分 {secs} 秒"
    return f"{secs} 秒"


def format_progress(progress: dict) -> str:
    stage = PROGRESS_STAGE_LABELS.get(str(progress.get("stage")), str(progress.get("stage")))
    processed = int(progress.get("processed", 0))
    total = int(progress.get("total", 0))
    percent = float(progress.get("percent", 0.0))
    rate = float(progress.get("rate_per_min", 0.0))
    eta = format_duration(progress.get("eta_sec"))
    elapsed = format_duration(progress.get("elapsed_sec"))
    current = progress.get("current_path")
    prefix = "已保存 checkpoint" if progress.get("checkpoint") else "進度"
    if progress.get("done"):
        prefix = "階段完成"
    base = f"{prefix}｜{stage}｜{processed}/{total}｜{percent:.1f}%｜速度 {rate:.1f} 筆/分｜已跑 {elapsed}｜預計剩餘 {eta}"
    if current:
        return f"{base}｜目前：{Path(str(current)).name}"
    return base


class VoiceFilterGui:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("聲音訓練集過濾工具")
        self.root.geometry("1040x740")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_requested = False

        self.project_root = StringVar(value=str(ROOT))
        self.input_folder = StringVar(value=str(ROOT / "clips"))
        self.uvr_folder = StringVar(value=str(ROOT / "out" / "female_raw_pass"))
        self.limit = StringVar(value="100")
        self.workers = StringVar(value="auto")
        self.uvr_batch_size = StringVar(value="16")
        self.process_all = BooleanVar(value=False)
        self.randomize = BooleanVar(value=True)
        self.notify_on_finish = BooleanVar(value=False)
        self.status = StringVar(value="待命")
        self.stage = StringVar(value=STAGE_BY_VALUE["filter-only"])
        self.backend = StringVar(value="audio-separator")
        self.uvr_python = StringVar(value=str(ROOT / ".venv-uvr" / "Scripts" / "python.exe"))
        self.uvr_model = StringVar(value="MDX23C-8KFFT-InstVoc_HQ.ckpt")

        self._build_ui()
        self.refresh_report()
        self._poll_logs()

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)

        source = ttk.LabelFrame(outer, text="輸入來源")
        source.pack(fill="x")
        ttk.Label(source, text="專案/輸出位置").grid(row=0, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.project_root).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(source, text="瀏覽", command=self.choose_project_root).grid(row=0, column=4)
        ttk.Label(source, text="篩選來源資料夾").grid(row=1, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.input_folder).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(source, text="瀏覽", command=self.choose_input_folder).grid(row=1, column=4)
        ttk.Label(source, text="UVR 來源資料夾").grid(row=2, column=0, sticky="w")
        ttk.Entry(source, textvariable=self.uvr_folder).grid(row=2, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(source, text="選擇資料夾", command=self.choose_uvr_folder).grid(row=2, column=4)
        source.columnconfigure(3, weight=1)

        controls = ttk.LabelFrame(outer, text="執行設定")
        controls.pack(fill="x", pady=(10, 0))
        ttk.Label(controls, text="執行模式").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.stage,
            values=list(STAGE_LABELS.keys()),
            state="readonly",
            width=20,
        ).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(controls, text="處理數量").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.limit, width=10).grid(row=0, column=3, sticky="w", padx=6)
        ttk.Checkbutton(controls, text="整個資料夾全做", variable=self.process_all).grid(row=0, column=4, sticky="w")
        ttk.Checkbutton(controls, text="隨機抽樣", variable=self.randomize).grid(row=0, column=5, sticky="w", padx=(8, 0))
        ttk.Label(controls, text="平行 workers").grid(row=1, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.workers, width=10).grid(row=1, column=3, sticky="w", padx=6)
        ttk.Label(controls, text="UVR 批次大小").grid(row=1, column=4, sticky="w")
        ttk.Entry(controls, textvariable=self.uvr_batch_size, width=10).grid(row=1, column=5, sticky="w", padx=6)
        ttk.Label(controls, text="UVR 後端").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.backend,
            values=["auto", "audio-separator", "demucs"],
            state="readonly",
            width=18,
        ).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(controls, text="UVR Python 路徑").grid(row=2, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.uvr_python).grid(row=2, column=1, columnspan=3, sticky="ew", padx=6)
        ttk.Button(controls, text="瀏覽", command=self.choose_uvr_python).grid(row=2, column=4)
        ttk.Label(controls, text="UVR 模型").grid(row=3, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.uvr_model).grid(row=3, column=1, columnspan=3, sticky="ew", padx=6)
        controls.columnconfigure(3, weight=1)

        status_row = ttk.Frame(outer)
        status_row.pack(fill="x", pady=(8, 0))
        ttk.Label(status_row, text="狀態").pack(side="left")
        ttk.Label(status_row, textvariable=self.status).pack(side="left", padx=(6, 16))
        ttk.Checkbutton(status_row, text="完成後跳提示", variable=self.notify_on_finish).pack(side="left")

        hint = ttk.Label(
            outer,
            text="建議流程：先用「只做篩選」處理 clips，確認 out/female_raw_pass 後，再選該資料夾執行「對資料夾做 UVR」。",
        )
        hint.pack(fill="x", pady=(8, 0))

        buttons = ttk.Frame(outer)
        buttons.pack(fill="x", pady=10)
        ttk.Button(buttons, text="開始執行", command=self.start).pack(side="left")
        ttk.Button(buttons, text="本階段後停止", command=self.stop).pack(side="left", padx=6)
        ttk.Button(buttons, text="開啟輸出資料夾", command=lambda: self.open_path(Path(self.project_root.get()) / "out")).pack(side="left", padx=6)
        ttk.Button(buttons, text="開啟篩選後女聲", command=lambda: self.open_path(Path(self.project_root.get()) / "out" / "female_raw_pass")).pack(side="left")
        ttk.Button(buttons, text="開啟 UVR 結果", command=lambda: self.open_path(Path(self.project_root.get()) / "out" / "uvr_cleaned_selected")).pack(side="left", padx=6)
        ttk.Button(buttons, text="開啟紀錄檔", command=lambda: self.open_path(Path(self.project_root.get()) / "work" / "manifests" / "master.jsonl")).pack(side="left")
        ttk.Button(buttons, text="重新載入報告", command=self.refresh_report).pack(side="left", padx=6)

        panes = ttk.PanedWindow(outer, orient="vertical")
        panes.pack(fill="both", expand=True)
        log_frame = ttk.LabelFrame(panes, text="執行紀錄")
        self.log_text = ScrolledText(log_frame, height=16)
        self.log_text.pack(fill="both", expand=True)
        report_frame = ttk.LabelFrame(panes, text="處理報告")
        self.report_text = ScrolledText(report_frame, height=10)
        self.report_text.pack(fill="both", expand=True)
        panes.add(log_frame, weight=3)
        panes.add(report_frame, weight=2)

    def choose_project_root(self) -> None:
        path = filedialog.askdirectory(initialdir=self.project_root.get())
        if path:
            self.project_root.set(path)

    def choose_input_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self.input_folder.get())
        if path:
            self.input_folder.set(path)

    def choose_uvr_folder(self) -> None:
        path = filedialog.askdirectory(initialdir=self.uvr_folder.get())
        if path:
            self.uvr_folder.set(path)
            self.stage.set(STAGE_BY_VALUE["uvr-folder"])

    def choose_uvr_python(self) -> None:
        name = filedialog.askopenfilename(filetypes=[("Python 執行檔", "python.exe"), ("所有檔案", "*.*")])
        if name:
            self.uvr_python.set(name)

    def start(self) -> None:
        if self.worker and self.worker.is_alive():
            self.set_status("已有任務正在背景執行")
            self.log("目前已有任務正在執行，未啟動新任務。")
            return
        self.cancel_requested = False
        self.set_status("背景執行中")
        self.worker = threading.Thread(target=self._run_worker, daemon=True)
        self.worker.start()

    def stop(self) -> None:
        self.cancel_requested = True
        self.log("已要求停止。程式會在目前階段完成後停止。")

    def _run_worker(self) -> None:
        try:
            limit = resolve_limit(self.limit.get(), self.process_all.get())
            root = Path(self.project_root.get()).resolve()
            input_folder = Path(self.input_folder.get()).resolve()
            uvr_folder = Path(self.uvr_folder.get()).resolve()
            workers = parse_workers(self.workers.get())
            uvr_batch_size = parse_uvr_batch_size(self.uvr_batch_size.get())
            os.environ["VOICE_FILTER_UVR_PYTHON"] = self.uvr_python.get()
            os.environ["VOICE_FILTER_UVR_MODEL"] = self.uvr_model.get()
            pipeline = Pipeline(
                root=root,
                clips_dir=input_folder,
                progress_callback=self.log_progress,
                max_workers=workers,
                uvr_batch_size=uvr_batch_size,
            )
            stage = STAGE_LABELS.get(self.stage.get(), self.stage.get())
            kwargs = {"limit": limit, "randomize": self.randomize.get()}
            self.log(f"開始執行：{STAGE_BY_VALUE.get(stage, stage)}，輸出位置：{root}")
            self.log("處理範圍：整個資料夾" if limit is None else f"處理範圍：{limit} 筆")
            self.log(f"平行 workers：{pipeline.max_workers}（UVR 本身仍以單工作避免 GPU 記憶體互搶）")
            self.log(f"UVR 批次大小：{pipeline.uvr_batch_size}（一次啟動 backend 處理多個檔案，避免重複載入模型）")
            self.log("進度保存：每 500 筆自動保存一次，可中斷後續跑；已存在於 out 的輸出會自動跳過。")
            self.log(f"篩選來源資料夾：{input_folder}")
            self.log(f"UVR 來源資料夾：{uvr_folder}")
            result = self._run_stage(pipeline, stage, kwargs, uvr_folder)
            self.log(f"完成：{result}")
            self.root.after(0, lambda: self.set_status("完成"))
            self.root.after(0, self.refresh_report)
            if self.notify_on_finish.get():
                self.root.after(0, lambda: messagebox.showinfo("處理完成", f"任務已完成。\n\n結果：{result}"))
        except Exception as exc:
            self.log(f"錯誤：{exc}")
            self.root.after(0, lambda: self.set_status("錯誤，請查看執行紀錄"))
            if self.notify_on_finish.get():
                self.root.after(0, lambda exc=exc: messagebox.showerror("聲音訓練集過濾工具", str(exc)))

    def _run_stage(self, pipeline: Pipeline, stage: str, kwargs: dict, uvr_folder: Path):
        if stage == "scan":
            return pipeline.scan(**kwargs)
        if stage == "classify":
            return pipeline.classify(**kwargs)
        if stage == "score":
            return pipeline.score(**kwargs)
        if stage == "filter-only":
            return pipeline.filter_only(**kwargs)
        if stage == "verify":
            return pipeline.verify(**kwargs)
        if stage == "uvr-folder":
            return pipeline.uvr_folder(uvr_folder, **kwargs, backend=self.backend.get())
        raise ValueError(f"未知的執行階段：{stage}")

    def log(self, message: str) -> None:
        self.log_queue.put(message)

    def set_status(self, message: str) -> None:
        self.status.set(message)

    def log_progress(self, progress: dict) -> None:
        text = format_progress(progress)
        self.log(text)
        self.root.after(0, lambda: self.set_status(text))

    def _poll_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log_text.insert(END, message + "\n")
            self.log_text.see(END)
        self.root.after(150, self._poll_logs)

    def refresh_report(self) -> None:
        report = Path(self.project_root.get()) / "out" / "report.md"
        text = report.read_text(encoding="utf-8") if report.exists() else "尚未產生報告。"
        self.report_text.delete("1.0", END)
        self.report_text.insert(END, text)

    def open_path(self, path: Path) -> None:
        if not path.exists():
            messagebox.showwarning("找不到路徑", f"路徑不存在：\n{path}")
            return
        subprocess.Popen(["explorer", str(path)])


def main(argv: list[str] | None = None) -> int:
    if os.environ.get("VOICE_FILTER_GUI_SMOKE") == "1":
        parse_limit("10")
        parse_limit("")
        resolve_limit("10", True)
        parse_workers("auto")
        parse_uvr_batch_size("16")
        resolve_worker_count(None)
        resolve_uvr_batch_size(None)
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Import-only smoke test.")
    args = parser.parse_args(argv)
    if args.smoke:
        parse_limit("10")
        parse_limit("")
        resolve_limit("10", True)
        parse_workers("auto")
        parse_uvr_batch_size("16")
        resolve_worker_count(None)
        resolve_uvr_batch_size(None)
        return 0
    root = Tk()
    VoiceFilterGui(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
