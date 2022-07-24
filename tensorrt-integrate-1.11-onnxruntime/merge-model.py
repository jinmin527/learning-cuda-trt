
##################################### export onnx

import torch
from torchvision import models

model1 = models.resnet18(True).eval()
model2 = models.resnet50(True).eval()

input_tensor = torch.zeros(1, 3, 224, 224)
torch.onnx.export(
    model1, (input_tensor,), "model1.onnx", opset_version=11, input_names=["image"], output_names=["prob"]
)

torch.onnx.export(
    model2, (input_tensor,), "model2.onnx", opset_version=11, input_names=["image"], output_names=["prob"]
)

with torch.no_grad():
    print("Torch Out1 = ", model1(input_tensor)[0, :5].data.numpy())
    print("Torch Out2 = ", model2(input_tensor)[0, :5].data.numpy())




############################################## merge onnx

import onnx

model1 = onnx.load("model1.onnx")
model2 = onnx.load("model2.onnx")

def rename_model(model, newname):

    for n in model.graph.node:
        n.name = newname(n.name)
        for i in range(len(n.input)):
            n.input[i] = newname(n.input[i])

        for i in range(len(n.output)):
            n.output[i] = newname(n.output[i])

    for n in model.graph.initializer:
        n.name = newname(n.name)

    for n in model.graph.input:
        n.name = newname(n.name)

    for n in model.graph.output:
        n.name = newname(n.name)


def newname_func(prefix):
    def impl(name):
        return f"{prefix}-{name}"
    return impl

rename_model(model1, newname_func("model1"))
rename_model(model2, newname_func("model2"))

model1.graph.node.extend(model2.graph.node)
model1.graph.initializer.extend(model2.graph.initializer)
# input不合并
#model1.graph.input.extend(model2.graph.input)
for n in model1.graph.node:
    for i in range(len(n.input)):
        if n.input[i] == model2.graph.input[0].name:
            n.input[i] = model1.graph.input[0].name

model1.graph.output.extend(model2.graph.output)

onnx.save(model1, "merge.onnx")





#############################################runtime inference

import onnxruntime
import numpy as np

image = np.zeros((1, 3, 224, 224), dtype=np.float32)
session = onnxruntime.InferenceSession("merge.onnx", providers=["CPUExecutionProvider"])
pred = session.run(["model1-prob", "model2-prob"], {"model1-image": image})

print("Onnx Out1 = ", pred[0][0, :5])
print("Onnx Out2 = ", pred[1][0, :5])