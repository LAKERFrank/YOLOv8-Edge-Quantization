
import argparse
import torch

from ultralytics.nn.autobackend import AutoBackend
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.utils.torch_utils import select_device

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    default=r'/Users/bartek/git/BartekTao/ultralytics/runs/detect/train178/weights/last.pt',
                    help='Path to .pt model file')
args_cli = parser.parse_args()

cfg=DEFAULT_CFG
verbose = False

overrides = {}
overrides['model'] = args_cli.model_path
overrides['mode'] = 'predict_v2'
overrides['data'] = 'tracknet.yaml'
overrides['batch'] = 1
args = get_cfg(cfg, overrides)

model = AutoBackend(args.model,
                         device=select_device(args.device, verbose=verbose),
                         dnn=args.dnn,
                         data=args.data,
                         fp16=args.half,
                         fuse=True,
                         verbose=verbose)

model.eval()

# 假設輸入是 1x3x640x640 的圖像
dummy_input = torch.randn(1, 10, 640, 640).to(model.device)

# 將模型轉為 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
