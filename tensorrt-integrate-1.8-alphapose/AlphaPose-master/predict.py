
from turtle import up
import yaml
from easydict import EasyDict as edict
from alphapose.models import builder
import torch
import numpy as np
import cv2

def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config

class MySPPE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        checkpoint = "pretrained_models/multi_domain_fast50_regression_256x192.pth"
        cfg = update_config("configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml")
        self.pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(checkpoint, map_location="cpu"))

    def forward(self, x):
        hm = self.pose_model(x)
        stride = int(256 / hm.size(2))
        b, c, h, w = map(int, hm.size())
        prob = hm.sigmoid()
        confidence, _ = prob.view(-1, c, h * w).max(dim=2, keepdim=True)
        prob = prob / prob.sum(dim=[2, 3], keepdim=True)
        coordx = torch.arange(w, device=prob.device, dtype=torch.float32)
        coordy = torch.arange(h, device=prob.device, dtype=torch.float32)
        hmx = (prob.sum(dim=2) * coordx).sum(dim=2, keepdim=True) * stride
        hmy = (prob.sum(dim=3) * coordy).sum(dim=2, keepdim=True) * stride
        keypoint = torch.cat([hmx, hmy, confidence], dim=2)
        return keypoint

model = MySPPE().eval()

x, y, w, h = 158, 104, 176, 693
image = cv2.imread("gril.jpg")[y:y+h, x:x+w]
image = image[..., ::-1]
image = cv2.resize(image, (256, 192))
image = ((image / 255.0) - [0.406, 0.457, 0.480]).astype(np.float32)
image = image.transpose(2, 0, 1)[None]
image = torch.from_numpy(image)

with torch.no_grad():
    keypoint = model(image)

print(keypoint.shape)
#return torch.cat([hmx, hmy, confidence], dim=2)

dummy = torch.zeros(1, 3, 256, 192)
torch.onnx.export(
    model, (dummy,), "fastpose.onnx", input_names=["image"], output_names=["predict"], opset_version=11, 
    dynamic_axes={
        "image": {0:"batch"}, "predict": {0:"batch"}
    }
)
print("Done")