# coding: UTF-8
import torch
from importlib import import_module

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    model_name = "bert"  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    # train
    model = x.Model(config)
    state_dict = torch.load("THUCNews/saved_dict/bert.ckpt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    seq_length = 32
    input_ids = torch.zeros(1, seq_length, dtype=torch.long)
    attention_mask = torch.zeros(1, seq_length, dtype=torch.long)
    torch.onnx.export(
        model, (input_ids, attention_mask), "classifier.onnx", 
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        opset_version=11,
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "logits": {0: "batch"}
        }
    )