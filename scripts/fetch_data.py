"""获取本项目所需的数据集与权重。

    python scripts/fetch_data.py              # 全部
    python scripts/fetch_data.py lta
    python scripts/fetch_data.py libcity
    python scripts/fetch_data.py weights
"""

from __future__ import annotations

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

LTA_DIR = ROOT / "datasets" / "lta"
LIBCITY_DATA = ROOT / "datasets" / "libcity"
LIBCITY_DIR = ROOT / "third_party" / "LibCity"
LIBCITY_LINK = LIBCITY_DIR / "raw_data"

LTA_REPO = "LOVEIII486/isy5002-lta-traffic-images"
LIBCITY_NAMES = ("PEMSD8", "METR_LA")
LIBCITY_SOURCES = (
    "https://pan.baidu.com/s/1qEfcXBO-QwZfiT0G3IYMpQ （提取码 1231）\n"
    "    https://drive.google.com/drive/folders/1g5v2Gq1tkOq8XO0HDCZ9nOTtRpB6-gPe"
)


def _is_link(path: pathlib.Path) -> bool:
    if not path.exists() and not path.is_symlink():
        return False
    checker = getattr(pathlib.Path, "is_junction", None)  # Python 3.12+
    return path.is_symlink() or bool(checker and checker(path))


def _make_link(link: pathlib.Path, target: pathlib.Path) -> None:
    if sys.platform == "win32":
        # mklink /J 建目录联接，不需要管理员权限。
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(f"mklink 失败：{(result.stdout + result.stderr).strip()}")
    else:
        os.symlink(target, link, target_is_directory=True)


def fetch_lta() -> None:
    from huggingface_hub import snapshot_download

    print(f"LTA 图像 → {LTA_DIR}")
    snapshot_download(repo_id=LTA_REPO, repo_type="dataset", local_dir=LTA_DIR)
    shutil.rmtree(LTA_DIR / ".cache", ignore_errors=True)  # snapshot_download 的记账目录


def fetch_libcity() -> None:
    if not (LIBCITY_DIR / "libcity").is_dir():
        raise SystemExit(
            "LibCity submodule 未就位。先执行：\n"
            "    git submodule update --init third_party/LibCity"
        )

    missing = [name for name in LIBCITY_NAMES if not (LIBCITY_DATA / name).is_dir()]
    if missing:
        raise SystemExit(
            f"{LIBCITY_DATA} 下缺少 {', '.join(missing)}。\n"
            f"从以下任一来源下载后放入该目录：\n    {LIBCITY_SOURCES}"
        )

    print(f"数据目录 {LIBCITY_DATA}")
    print(f"  已就位 {', '.join(LIBCITY_NAMES)}")

    # LibCity 用相对路径 ./raw_data/<数据集>/ 读数据，基准是它的工作目录，
    # 所以数据放在 submodule 外面、在 submodule 根留一个入口即可。
    if _is_link(LIBCITY_LINK):
        target = pathlib.Path(os.path.realpath(LIBCITY_LINK))
        if target == LIBCITY_DATA.resolve():
            print(f"联接已就位 -> {target}")
            return
        raise SystemExit(f"{LIBCITY_LINK} 指向 {target}，应为 {LIBCITY_DATA}。确认后删除它再重跑。")
    if LIBCITY_LINK.is_dir():
        if any(LIBCITY_LINK.iterdir()):
            raise SystemExit(f"{LIBCITY_LINK} 是含数据的真实目录，先把数据移进 {LIBCITY_DATA} 再重跑。")
        LIBCITY_LINK.rmdir()

    _make_link(LIBCITY_LINK, LIBCITY_DATA)
    print(f"已建联接 {LIBCITY_LINK} -> {LIBCITY_DATA}")


def fetch_weights() -> None:
    sys.path.insert(0, str(ROOT))
    from src.detection.run_detect import ensure_weights

    ensure_weights()


TARGETS = {"lta": fetch_lta, "libcity": fetch_libcity, "weights": fetch_weights}


def main() -> int:
    parser = argparse.ArgumentParser(description="获取数据集与权重")
    parser.add_argument("targets", nargs="*", choices=[*TARGETS, "all"], default=["all"])
    args = parser.parse_args()

    names = list(TARGETS) if not args.targets or "all" in args.targets else args.targets
    for name in names:
        TARGETS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
