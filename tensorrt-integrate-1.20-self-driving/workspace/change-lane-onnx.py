import onnx
import onnx.helper

model = onnx.load("ultra_fast_lane_detection_culane_288x800.onnx")
postprocess = onnx.load("postprocess.onnx")

for n in postprocess.graph.node:
    n.name = "post/" + n.name

    for i, v in enumerate(n.input):
        if v == "0":
            n.input[i] = "200"
        else:
            n.input[i] = "post/" + v

    for i, v in enumerate(n.output):
        if v == "18":
            n.output[i] = "points"
        else:
            n.output[i] = "post/" + v

model.graph.node.extend(postprocess.graph.node)

while len(model.graph.output) > 0:
    model.graph.output.pop()

model.graph.output.extend(postprocess.graph.output)
model.graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info("input.1", 1, ["batch", 3, 288, 800]))
model.graph.output[0].CopyFrom(onnx.helper.make_tensor_value_info("points", 1, ["batch", 18, 4]))

while len(model.graph.value_info) > 0:
    model.graph.value_info.pop()
    
onnx.save(model, "new-lane.onnx")