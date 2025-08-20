"""
Head-only QAT micro-finetune for sensitive blocks (kpt/DFL/track head).
Freeze backbone/neck; insert fake-quant on selected heads; brief fine-tune; then convert.
NOTE: You must wire your own dataloader + loss for real training.
"""
import argparse, torch
import torch.ao.quantization as tq
from ultralytics import YOLO

def select_qat_modules(model, include_keys):
    for n, m in model.named_modules():
        m.qconfig = tq.get_default_qat_qconfig('fbgemm') if any(k in n for k in include_keys) else None

def freeze_except(model, include_keys):
    for n, p in model.named_parameters():
        p.requires_grad = any(k in n for k in include_keys)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True)
    ap.add_argument("--include", nargs="+", default=["kpt", "dfl", "head", "heatmap"])
    ap.add_argument("--epochs", type=int, default=5)
    args = ap.parse_args()

    y = YOLO(args.pt)
    net = y.model
    net.eval()
    freeze_except(net, args.include)
    select_qat_modules(net, args.include)

    prepared = tq.prepare_qat(net, inplace=False)
    prepared.train()

    # TODO: plug in your dataloader + loss, e.g. MultiLoss for pose/track
    opt = torch.optim.AdamW([p for p in prepared.parameters() if p.requires_grad], lr=1e-4)
    for epoch in range(args.epochs):
        # for imgs, targets in loader:
        #     out = prepared(imgs)
        #     loss = multi_loss(out, targets)
        #     loss.backward(); opt.step(); opt.zero_grad()
        pass

    prepared.eval()
    quantized = tq.convert(prepared)
    torch.save(quantized.state_dict(), "out/qat_heads_int8_state.pth")
    print("[OK] QAT head model saved. Re-export ONNX from this checkpoint if needed.")
