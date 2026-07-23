"""
手动下载Whisper模型指南
======================

由于网络限制，需要手动下载Whisper模型。

方法1：使用镜像站下载
--------------------
1. 访问 Hugging Face 镜像站（如 hf-mirror.com）
2. 搜索 "Systran/faster-whisper-base"（或其他大小）
3. 下载所有文件到本地目录

方法2：使用代理下载
------------------
设置环境变量后运行下载脚本：

Windows CMD:
    set HTTP_PROXY=http://127.0.0.1:7890
    set HTTPS_PROXY=http://127.0.0.1:7890
    python download_whisper_model.py --model base

Windows PowerShell:
    $env:HTTP_PROXY="http://127.0.0.1:7890"
    $env:HTTPS_PROXY="http://127.0.0.1:7890"
    python download_whisper_model.py --model base

方法3：使用Hugging Face镜像
--------------------------
设置环境变量：

Windows CMD:
    set HF_ENDPOINT=https://hf-mirror.com
    python download_whisper_model.py --model base

Windows PowerShell:
    $env:HF_ENDPOINT="https://hf-mirror.com"
    python download_whisper_model.py --model base

方法4：直接下载模型文件
---------------------
1. 访问 https://hf-mirror.com/Systran/faster-whisper-base
2. 下载所有文件到 models/faster-whisper-base 目录
3. 在 .env 文件中设置:
   WHISPER_MODEL_PATH=models/faster-whisper-base

模型文件列表（base版本）:
- config.json
- model.bin
- tokenizer.json
- vocabulary.txt
- vocabulary.json
"""

import os
import sys


def download_with_mirror(model_size: str = "base"):
    """使用镜像站下载模型"""
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    print(f"使用镜像站下载 Whisper {model_size} 模型...")
    print("镜像地址: https://hf-mirror.com")
    
    try:
        from faster_whisper import WhisperModel
        
        output_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"保存目录: {output_dir}")
        print("正在下载，请耐心等待...")
        
        model = WhisperModel(
            model_size,
            device="cpu",
            compute_type="int8",
            download_root=output_dir
        )
        
        print(f"\n✓ 模型下载成功!")
        print(f"模型已保存到缓存目录")
        return True
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        return False


if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else "base"
    download_with_mirror(model)
