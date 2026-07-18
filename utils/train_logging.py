"""
Dual-channel training logging (paper2-style): tqdm TTY + ANSI-free ``train_valid.log``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_ANSI_RE = re.compile(r"\x1B\[[0-9;]*m")

PREFIX_CONFIG = "[Config] "
PREFIX_SYSTEM = "[System] "

RESET = "\033[0m"
CYAN = "\033[36m"
BLUE = "\033[34m"
GREEN = "\033[32m"
DIM = "\033[2m"


def strip_ansi(s: str) -> str:
    return _ANSI_RE.sub("", s)


def _c(msg: str, color: str) -> str:
    return f"{color}{msg}{RESET}"


class _AnsiFreeFileFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__(fmt="[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        return strip_ansi(super().format(record))


class TrainingDualLogger:
    """Routes plain timestamped lines to ``train_valid.log`` and TTY via ``tqdm.write``."""

    _LOGGER_NAME = "paper3.train.dual"

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = Path(run_dir)
        self._log_path = self._run_dir / "train_valid.log"
        self._logger: Optional[logging.Logger] = None

    def setup(self, *, append: bool = False) -> None:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        lg = logging.getLogger(self._LOGGER_NAME)
        lg.handlers.clear()
        lg.setLevel(logging.INFO)
        lg.propagate = False
        fh = logging.FileHandler(self._log_path, mode="a" if append else "w", encoding="utf-8")
        fh.setFormatter(_AnsiFreeFileFormatter())
        lg.addHandler(fh)
        self._logger = lg

    def _file(self, plain: str) -> None:
        if self._logger is not None:
            self._logger.info(plain)

    def tty(self, msg: str) -> None:
        tqdm.write(msg)

    def log_plain_file(self, plain: str) -> None:
        self._file(plain)

    def log_system_file_only(self, plain_detail: str) -> None:
        self._file(f"{PREFIX_SYSTEM}{plain_detail}")

    def log_both_plain(self, plain: str) -> None:
        self._file(plain)
        tqdm.write(plain)

    def log_epoch_summary_line(self, line: str) -> None:
        plain = strip_ansi(line)
        self._file(plain)
        tqdm.write(plain)

    def emit_config_box_top(self, *, width: int = 60) -> None:
        sep = "=" * width
        self.tty("")
        self.tty(_c(sep, DIM))
        self.tty(_c(sep, DIM))

    def emit_config_line(self, key: str, value: str) -> None:
        plain = f"{key}={value}"
        self._file(f"{PREFIX_CONFIG}{plain}")
        self.tty(_c(f"{PREFIX_CONFIG}{plain}", CYAN))

    def emit_config_box_bottom(self, *, width: int = 60) -> None:
        self.log_system_file_only("phase=training_main  boundary=after_config")
        sep = "=" * width
        self.tty(_c(sep, DIM))
        self.tty(_c(sep, DIM))
        self.tty("")

    def log_best_update(self, epoch: int, updated_names: list[str], saved_files: list[str]) -> None:
        names = ", ".join(updated_names)
        files = " ".join(saved_files)
        best_plain = f"[Best] epoch={epoch:03d} updated: {names}"
        save_plain = f"  -> saved: {files}"
        self._file(best_plain)
        self._file(save_plain)
        self.tty(_c(best_plain, GREEN))
        self.tty(save_plain)
        self.log_system_file_only(
            f"epoch={epoch:03d}  event=best_checkpoint  updated={names.replace(' ', '')}  files={files}"
        )
