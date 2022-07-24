
import torch
import yaml
from easydict import EasyDict as edict

from alphapose.models import builder

class Alphapose(torch.nn.Module):
    def __init__(self):
        super().__init__()
        config_file = "configs/halpe_136/resnet/256x192_res50_lr1e-3_2x-regression.yaml"
        check_point = "pretrained_models/multi_domain_fast50_regression_256x192.pth"
        with open(config_file, "r") as f:
            config = edict(yaml.load(f, Loader=yaml.FullLoader))

        self.pose_model = builder.build_sppe(config.MODEL, preset_cfg=config.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(check_point, map_location="cpu"))

    def forward(self, x):
        hm = self.pose_model(x)
        
        stride = int(x.size(2) / hm.size(2))
        b, c, h, w = map(int, hm.size())
        prob = hm.sigmoid()
        confidence, _ = prob.view(-1, c, h * w).max(dim=2, keepdim=True)
        prob = prob / prob.sum(dim=[2, 3], keepdim=True)
        coordx = torch.arange(w, device=prob.device, dtype=torch.float32)
        coordy = torch.arange(h, device=prob.device, dtype=torch.float32)
        hmx = (prob.sum(dim=2) * coordx).sum(dim=2, keepdim=True) * stride
        hmy = (prob.sum(dim=3) * coordy).sum(dim=2, keepdim=True) * stride
        return torch.cat([hmx, hmy, confidence], dim=2)

model = Alphapose().eval()
dummy = torch.zeros(1, 3, 256, 192)
torch.onnx.export(
    model, (dummy,), "alpha-pose-136.onnx", input_names=["images"], output_names=["keypoints"], 
    opset_version=11,
    dynamic_axes={
        "images": {0: "batch"},
        "keypoints": {0: "batch"}
    }
)