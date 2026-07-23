"""
Whisper模型下载脚本
==================

用于预先下载Whisper模型到本地，避免使用时需要联网。

使用方式：
    python download_whisper_model.py --model base

模型大小参考：
    tiny:   ~75MB
    base:   ~150MB
    small:  ~500MB
    medium: ~1.5GB
    large:  ~3GB
"""

import os
import argparse
import sys


def download_model(model_size: str, output_dir: str = None):
    """
    下载Whisper模型
    
    Args:
        model_size: 模型大小 (tiny/base/small/medium/large)
        output_dir: 输出目录，默认为项目目录下的models文件夹
    """
    try:
        from faster_whisper import download_model
    except ImportError:
        print("请先安装faster-whisper: pip install faster-whisper")
        sys.exit(1)
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "models")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"正在下载 Whisper {model_size} 模型...")
    print(f"保存目录: {output_dir}")
    print("请耐心等待，首次下载可能需要几分钟...")
    
    try:
        model_path = download_model(model_size, output_dir)
        print(f"\n✓ 模型下载成功!")
        print(f"模型路径: {model_dir}")
        print(f"\n使用方式:")
        print(f"  在 .env 文件中添加:")
        print(f"  WHISPER_MODEL_PATH={model_path}")
        return model_path
    except Exception as e:
        print(f"\n✗ 模型下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 使用代理:")
        print("   set HTTP_PROXY=http://your-proxy:port")
        print("   set HTTPS_PROXY=http://your-proxy:port")
        print("3. 手动下载模型:")
        print("   访问 https://huggingface.co/Systran")
        print("   下载对应的模型文件")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="下载Whisper模型")
    parser.add_argument(
        "--model", 
        type=str, 
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="模型大小 (默认: base)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录"
    )
    
    args = parser.parse_args()
    download_model(args.model, args.output)
