
# pip install transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
import os
import torch
import torch.onnx

def load_model():
    cache_dir = "model"
    backbone  = "dslim/bert-base-NER"

    tokenizer_file = os.path.join(cache_dir, "tokenizer.json")
    if os.path.exists(tokenizer_file):
        tokenizer = AutoTokenizer.from_pretrained(cache_dir, add_prefix_space=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(backbone, add_prefix_space=True)
        tokenizer.save_pretrained(cache_dir)

    config_file = os.path.join(cache_dir, "config.json") 
    if os.path.exists(config_file):
        config_model = AutoConfig.from_pretrained(config_file)
    else:
        config_model = AutoConfig.from_pretrained(backbone) 
        config_model.save_pretrained(cache_dir)

    backbone_file = os.path.join(cache_dir, "pytorch_model.bin") 
    if os.path.exists(backbone_file):
        backbone = AutoModelForTokenClassification.from_pretrained(backbone_file, config=config_model)
    else:
        backbone = AutoModelForTokenClassification.from_pretrained(backbone, config=config_model)
        backbone.save_pretrained(cache_dir)
    return backbone, tokenizer


class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()

        self.model = model
    
    def forward(self, input_ids, attention_mask):
        
        logits = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            return_dict = False
        )[0]
        return torch.softmax(logits, dim=-1)

model, tokenizer = load_model()
seq_length     = 512
input_ids      = torch.zeros(1, seq_length, dtype=torch.long)
attention_mask = torch.ones(1, seq_length, dtype=torch.long)

model = Model(model)
torch.onnx.export(model, (input_ids, attention_mask), 
    "workspace/ner.onnx", 
    opset_version=11, 
    input_names=["input_ids", "attention_mask"], 
    output_names=["logits"], 
    dynamic_axes={
        "input_ids": {0: "batch", 1: "seqlength"},
        "attention_mask": {0: "batch", 1: "seqlength"},
        "logits": {0: "batch", 1: "seqlength"}
    }
)