"""LibCity 兼容层：官方入口在本机导入即失败，这里逐项绕开，只作用于本进程。"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
import types
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIBCITY_ROOT = PROJECT_ROOT / "third_party" / "LibCity"
MODEL_DIR = LIBCITY_ROOT / "libcity" / "model" / "traffic_speed_prediction"
EXECUTOR_DIR = LIBCITY_ROOT / "libcity" / "executor"

# LibCity 只用到这两个别名：float 在三个 _load_*_3d 加载器里，int 在 GWNET 与 GMAN 里。
_NUMPY_ALIASES = (("float", float), ("int", int))

_active_exp_id: int | str | None = None


def _patch_numpy_aliases() -> list[str]:
    """补回 numpy 1.x 的标量别名，返回本次补上的名字。"""
    patched = []
    for name, value in _NUMPY_ALIASES:
        if not hasattr(np, name):
            setattr(np, name, value)
            patched.append(name)
    return patched


def _ensure_tensorboard() -> str:
    """有 tensorboard 就用真的，没有就注入最小替身。"""
    try:
        import torch.utils.tensorboard  # noqa: F401

        return "real"
    except Exception:
        pass

    class _SummaryWriter:
        """替身：任何方法调用都返回 None。"""

        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *args, **kwargs: None

    # 不能只看 sys.modules：没被导入过的模块不在其中，那会在已安装时也错误地注入替身。
    for module_name in ("tensorboard", "torch.utils.tensorboard"):
        module = ModuleType(module_name)
        module.SummaryWriter = _SummaryWriter  # type: ignore[attr-defined]
        sys.modules[module_name] = module
    return "shim"


def _ensure_ray() -> str:
    """训练执行器只用到 ray.tune 的 checkpoint_dir 与 report。"""
    try:
        import ray  # noqa: F401

        return "real"
    except Exception:
        pass

    @contextlib.contextmanager
    def checkpoint_dir(step: int = 0, **kwargs):
        yield str(_cache_dir("model_cache"))

    tune = types.SimpleNamespace(checkpoint_dir=checkpoint_dir, report=lambda **kwargs: None)
    ray_module = ModuleType("ray")
    ray_module.tune = tune  # type: ignore[attr-defined]
    sys.modules["ray"] = ray_module
    sys.modules["ray.tune"] = tune  # type: ignore[assignment]
    return "shim"


def _patch_loss_torch_returns() -> bool:
    """让 R² 与 EVAR 返回张量，避免 LibCity 内部的设备不匹配。"""
    import torch

    from libcity.model import loss as libcity_loss

    if getattr(libcity_loss, "_isy5002_patched", False):
        return True

    original_r2 = libcity_loss.r2_score_torch
    original_evar = libcity_loss.explained_variance_score_torch
    libcity_loss.r2_score_torch = lambda preds, labels: torch.as_tensor(original_r2(preds, labels))
    libcity_loss.explained_variance_score_torch = lambda preds, labels: torch.as_tensor(
        original_evar(preds, labels)
    )
    libcity_loss._isy5002_patched = True
    return True


def _cache_dir(*parts: str) -> Path:
    """当前实验的缓存目录。"""
    return cache_dir("shared" if _active_exp_id is None else _active_exp_id, *parts)


def cache_dir(exp_id: int | str, *parts: str) -> Path:
    """LibCity 在子模块内按 exp_id 组织的缓存目录。"""
    return LIBCITY_ROOT / "libcity" / "cache" / str(exp_id) / Path(*parts)


def _load_module_by_path(module_name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法装载 {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_BASE_EXECUTORS = (
    ("libcity.executor.abstract_executor", "abstract_executor.py"),
    ("libcity.executor.traffic_state_executor", "traffic_state_executor.py"),
)

# 类名到文件名的对应不是机械可推的（TrafficStateExecutor 落不到 traffic_state_executor
# 这个名字上），因此显式登记。
_EXECUTOR_MODULES = {
    "TrafficStateExecutor": "libcity.executor.traffic_state_executor",
    "MTGNNExecutor": "libcity.executor.mtgnn_executor",
}


def _install_executor_package() -> None:
    """用只指向同一目录的空包顶替 libcity.executor，并登记两个基础模块。

    这样按路径装载 dcrnn_executor 之类的子类时，它们对基类的导入才能解析。
    """
    if "libcity.executor" not in sys.modules:
        package = ModuleType("libcity.executor")
        package.__path__ = [str(EXECUTOR_DIR)]  # type: ignore[attr-defined]
        sys.modules["libcity.executor"] = package

    for module_name, filename in _BASE_EXECUTORS:
        if module_name not in sys.modules:
            _load_module_by_path(module_name, EXECUTOR_DIR / filename)


def load_model_module(model_name: str) -> ModuleType:
    """按文件路径装载模型模块，绕开会导入 dgl 的模型包 __init__。"""
    path = MODEL_DIR / f"{model_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{path}")
    return _load_module_by_path(f"libcity_model_{model_name.lower()}", path)


def load_model_class(model_name: str):
    """按名字取模型类。"""
    module = load_model_module(model_name)
    return getattr(module, model_name)


def load_executor_class(name: str = "TrafficStateExecutor"):
    """按名字装载执行器，绕开会导入 23 个执行器的 libcity.executor 包。"""
    _install_executor_package()
    module_name = _EXECUTOR_MODULES.get(name)
    if module_name is None:
        raise KeyError(f"未登记的执行器：{name}；已登记：{sorted(_EXECUTOR_MODULES)}")
    if module_name not in sys.modules:
        _load_module_by_path(module_name, EXECUTOR_DIR / f"{module_name.rsplit('.', 1)[-1]}.py")
    return getattr(sys.modules[module_name], name)


def activate(exp_id: int | str | None = None) -> dict[str, object]:
    """施加全部兼容处理并把 LibCity 根目录加入 sys.path，返回各项处理的结果。

    exp_id 决定 LibCity 的缓存路径，跨数据集必须用不同的值，否则会共用评估缓存。
    """
    global _active_exp_id
    _active_exp_id = exp_id

    if str(LIBCITY_ROOT) not in sys.path:
        sys.path.insert(0, str(LIBCITY_ROOT))

    report: dict[str, object] = {
        "libcity_root": str(LIBCITY_ROOT),
        "exp_id": exp_id,
        "numpy_aliases": _patch_numpy_aliases(),
        "tensorboard": _ensure_tensorboard(),
        "ray": _ensure_ray(),
    }
    report["loss_torch_returns"] = _patch_loss_torch_returns()
    return report


@contextlib.contextmanager
def libcity_cwd():
    """LibCity 一律用相对路径读 config 与 raw_data，须在其根目录下执行。"""
    original = Path.cwd()
    os.chdir(LIBCITY_ROOT)
    try:
        yield
    finally:
        os.chdir(original)


def collect_artifacts(exp_id: int | str, destination: Path) -> list[Path]:
    """把子模块缓存里的产物复制出来；缓存被 gitignore，重新检出子模块即丢失。"""
    destination.mkdir(parents=True, exist_ok=True)
    source = LIBCITY_ROOT / "libcity" / "cache" / str(exp_id)
    copied: list[Path] = []
    if not source.is_dir():
        return copied
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        copied.append(target)

    log_dir = LIBCITY_ROOT / "libcity" / "log"
    if log_dir.is_dir():
        for path in sorted(log_dir.glob(f"{exp_id}-*")):
            target = destination / "log" / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            copied.append(target)
    return copied
