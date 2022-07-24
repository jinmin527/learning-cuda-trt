
# pip install transformers
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline
import os

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


model, tokenizer = load_model()
input_text = "My name is Clara and I live in Berkeley, California."
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

for item in nlp(input_text):
    print(item)

# model.eval()
# pred = model(**tokenizer(input_text, return_tensors="pt"))
# probs = pred.logits.softmax(-1)[0]
# labels = probs.argmax(-1)
# print(labels)

# vocab = open("model/vocab.txt", "r").read().split()
# tok = tokenizer(input_text)
# input_ids = tok["input_ids"]

# # for i in input_ids:
# #     print(vocab[i], i)
# print(probs.shape, labels.shape)
# for r, i in enumerate(labels):
#     print(probs[r, i])