import torch
from mmdet.apis import init_detector, inference_detector

config_file = 'configs/yolox/yolox_tiny_8x8_300e_coco.py'
# 从 model zoo 下载 checkpoint 并放在 `checkpoints/` 文件下
# 网址为: http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth
checkpoint_file = 'checkpoints/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth'
device = 'cuda:0'
#初始化检测器
# model = init_detector(config_file, checkpoint_file, device=device)
# # 推理演示图像
# print(inference_detector(model, 'demo/demo.jpg'))

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = init_detector(config_file, checkpoint_file, device=device)

    def forward(self, x):
        ib, ic, ih, iw = map(int, x.shape)
        x = self.model.backbone(x)
        x = self.model.neck(x)
        clas, bbox, objness = self.model.bbox_head(x)

        output_x = []
        for class_item, bbox_item, objness_item in zip(clas, bbox, objness):
            hm_b, hm_c, hm_h, hm_w = map(int, class_item.shape)
            stride_h, stride_w = ih / hm_h, iw / hm_w
            strides = torch.tensor([stride_w, stride_h], device=device).view(-1, 1, 2)

            prior_y, prior_x = torch.meshgrid(torch.arange(hm_h), torch.arange(hm_w))
            prior_x = prior_x.reshape(hm_h * hm_w, 1).to(device)
            prior_y = prior_y.reshape(hm_h * hm_w, 1).to(device)
            prior_xy = torch.cat([prior_x, prior_y], dim=-1)
            class_item = class_item.permute(0, 2, 3, 1).reshape(-1, hm_h * hm_w, hm_c)
            bbox_item  = bbox_item.permute(0, 2, 3, 1).reshape(-1, hm_h * hm_w, 4)
            objness_item = objness_item.reshape(-1, hm_h * hm_w, 1)
            pred_xy = (bbox_item[..., :2] + prior_xy) * strides
            pred_wh = bbox_item[..., 2:4].exp() * strides
            pred_class = torch.cat([objness_item, class_item], dim=-1).sigmoid()
            output_x.append(torch.cat([pred_xy, pred_wh, pred_class], dim=-1))

        return torch.cat(output_x, dim=1)

m = Model().eval()
image = torch.zeros(1, 3, 416, 416, device=device)
torch.onnx.export(
    m, (image,), "yolox.onnx",
    opset_version=11, 
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},
        "output": {0: "batch"}
    }
)
print("Done.!")