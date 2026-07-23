"""将已下载的 PyTorch 模型导出为 ONNX（一次性脚本）"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

work_dir = "models/emotion_classifier/pytorch"
out_dir = "models/emotion_classifier"

model = AutoModelForSequenceClassification.from_pretrained(work_dir)
tokenizer = AutoTokenizer.from_pretrained(work_dir)
tokenizer.save_pretrained(out_dir)

dummy = tokenizer("test", return_tensors="pt")
torch.onnx.export(
    model,
    (dummy["input_ids"], dummy["attention_mask"]),
    out_dir + "/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch"},
    },
    opset_version=14,
)
size_mb = os.path.getsize(out_dir + "/model.onnx") / 1024 / 1024
print(f"ONNX 模型已导出: {out_dir}/model.onnx ({size_mb:.1f} MB)")
