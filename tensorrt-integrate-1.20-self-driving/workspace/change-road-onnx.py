import onnx
import onnx.helper

model = onnx.load("road-segmentation-adas-0001.onnx")

rms = []
for i, n in enumerate(model.graph.node):
    if n.name == "Conv__618":
        n.input[0] = "data"

    if n.name == "StatefulPartitionedCall/model/conv2d/Conv2D__6":
        #model.graph.node.remove(n)
        rms.append(i)

    # if n.name == "StatefulPartitionedCall/model/lambda_4/resize/ResizeBilinear":
    #     #model.graph.node.remove(n)
    #     rms.append(i)
    #     tinput = n.input[0]

    # if n.name == "StatefulPartitionedCall/model/tf.nn.softmax/Softmax":
    #     n.input[0] = tinput
    #     n.attribute[0].CopyFrom(onnx.helper.make_attribute("axis", 1))
    #     print(n)

for n in rms[::-1]:
    del model.graph.node[n]

model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("data", 1, [1, 3, 512, 896]))
model.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("tf.identity", 1, [1, 512, 896, 4]))
model.producer_name = "pytorch"
model.producer_version = "1.9"
model.graph.name = "model"
onnx.save(model, "new-road.onnx")