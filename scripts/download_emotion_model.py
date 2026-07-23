"""
下载中文情绪分类模型并导出为 ONNX 格式。

默认从 ModelScope 下载（国内可直连），也可通过 HF_ENDPOINT 镜像访问 HuggingFace。

使用方式：
    # 默认：ModelScope 中文情绪分类模型
    python scripts/download_emotion_model.py

    # 指定模型 ID
    python scripts/download_emotion_model.py --model-id iic/nlp_structbert_emotion-classification_chinese-base

    # 从 HuggingFace 镜像下载
    set HF_ENDPOINT=https://hf-mirror.com && python scripts/download_emotion_model.py --model-id bert-base-chinese --source hf

依赖：
    pip install optimum[onnxruntime] onnx onnxruntime transformers modelscope

ONNX 导出后，在 .env 中设置：
    EMOTION_ENGINE=onnx
    EMOTION_ONNX_MODEL_PATH=models/emotion_classifier/model.onnx
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = str(REPO_ROOT / "models" / "emotion_classifier")
# ModelScope 上的中文情绪分类模型（StructBERT，7 标签，已微调）
DEFAULT_MODEL_ID = "iic/nlp_structbert_emotion-classification_chinese-base"


def _download_from_modelscope(model_id: str, download_dir: Path) -> Path:
    """从 ModelScope 下载模型到指定目录，返回模型目录路径。"""
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[INFO] 从 ModelScope 下载: {model_id}")
    model_dir = snapshot_download(
        model_id=model_id,
        cache_dir=str(download_dir),
        revision="master",
    )
    # snapshot_download 可能返回 cache 中的符号链接路径，直接复制到目标
    local_dir = download_dir / "pytorch_model"
    if not (local_dir / "config.json").exists():
        print("[INFO] 复制模型文件到工作目录...")
        if local_dir.exists():
            shutil.rmtree(str(local_dir), ignore_errors=True)
        shutil.copytree(str(model_dir), str(local_dir))
    return local_dir


def _download_from_huggingface(model_id: str, download_dir: Path) -> Path:
    """从 HuggingFace 下载模型，返回模型目录路径。"""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    print(f"[INFO] 从 HuggingFace 下载: {model_id}")
    local_dir = download_dir / "pytorch_model"
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model.save_pretrained(str(local_dir))
    tokenizer.save_pretrained(str(local_dir))
    return local_dir


def download_and_export(
    model_id: str, output_dir: str, force: bool = False, source: str = "auto"
) -> None:
    """下载模型并导出为 ONNX。"""
    out = Path(output_dir)
    model_file = out / "model.onnx"
    tokenizer_file = out / "tokenizer_config.json"

    if model_file.exists() and tokenizer_file.exists() and not force:
        print(f"[SKIP] {model_file} 已存在，使用 --force 覆盖")
        return

    out.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 模型 ID: {model_id}")

    # 检查依赖
    try:
        import optimum.onnxruntime  # noqa: F401
        from transformers import AutoTokenizer
    except ImportError as e:
        print(f"[ERROR] 缺少依赖: {e}", file=sys.stderr)
        print(
            "请运行: pip install optimum[onnxruntime] onnx onnxruntime transformers modelscope",
            file=sys.stderr,
        )
        sys.exit(1)

    # 临时下载目录
    work_dir = out / "_download"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 决定下载源
        if source == "auto":
            # ModelScope 格式（iic/...）走 ms，否则走 hf
            if model_id.startswith("iic/") or model_id.startswith("damo/"):
                source = "ms"
            else:
                source = "hf"

        # 下载 PyTorch 模型
        if source == "ms":
            model_dir = _download_from_modelscope(model_id, work_dir)
        else:
            model_dir = _download_from_huggingface(model_id, work_dir)

        print(f"[INFO] 模型文件路径: {model_dir}")

        # optimum-cli 导出 ONNX
        print("[INFO] 导出 ONNX...")
        ret = os.system(
            f'python -m optimum.exporters.onnx '
            f'--model "{model_dir}" '
            f'--task text-classification '
            f'"{out}"'
        )
        if ret != 0:
            raise RuntimeError(f"optimum-cli 导出失败，返回码 {ret}")

        # 重命名为统一名称 model.onnx
        onnx_files = list(out.glob("*.onnx"))
        if not onnx_files:
            raise RuntimeError("未找到导出的 .onnx 文件")
        if onnx_files[0].name != "model.onnx":
            onnx_files[0].rename(model_file)

        print(f"[OK] 模型已保存到 {out}")
        files = sorted(f.name for f in out.iterdir() if f.name != "_download")
        print(f"     文件: {', '.join(files)}")
    except Exception as e:
        print(f"[ERROR] 导出失败: {e}", file=sys.stderr)
        # 不删 out（_download 可能有用），但要提示
        print(f"[HINT] 可手动清理: {out}", file=sys.stderr)
        sys.exit(1)
    finally:
        # 清理临时下载目录
        if work_dir.exists():
            shutil.rmtree(str(work_dir), ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="下载中文情绪分类模型并导出 ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认：ModelScope structbert 情绪分类模型（国内推荐）
  python scripts/download_emotion_model.py

  # 指定模型
  python scripts/download_emotion_model.py --model-id iic/nlp_structbert_emotion-classification_chinese-base

  # 从 HuggingFace 镜像下载
  set HF_ENDPOINT=https://hf-mirror.com
  python scripts/download_emotion_model.py --model-id bert-base-chinese --source hf
        """.strip(),
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help=f"模型 ID（默认: {DEFAULT_MODEL_ID}）",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT,
        help=f"输出目录（默认: {DEFAULT_OUTPUT}）",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "ms", "hf"],
        default="auto",
        help="下载源：auto=自动判断, ms=ModelScope, hf=HuggingFace",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制覆盖已有模型文件",
    )
    args = parser.parse_args()
    download_and_export(args.model_id, args.output_dir, args.force, args.source)


if __name__ == "__main__":
    main()
