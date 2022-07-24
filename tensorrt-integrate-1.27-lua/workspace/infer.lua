
local yolo = require "yolo"
local logger = require "logger"

if not yolo.exists("yolov5s.trtmodel") then
    ok = yolo.compile({
        max_batch_size=1,
        source="yolov5s.onnx",
        output="yolov5s.trtmodel",
        fp16=false,
        device_id=0,
        max_workspace_size=1024*1024*256
    })
    logger.info("compile " .. (ok and "Finish" or "Failed"))
end

infer = yolo.create_infer({
    file="yolov5s.trtmodel",
    gpuid=0,
    threshold=0.25
})

ok = infer:forward({
    input="rq.jpg",
    output="rq-output.jpg"
})
logger.info("forward " .. (ok and "Finish" or "Failed"))

