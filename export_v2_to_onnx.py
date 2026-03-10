import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


BOARD_SIZE = 15
DEFAULT_PTH = "Gobang_AlphaZero_V2_model.pth"
DEFAULT_ONNX = "Gobang_AlphaZero_V2.onnx"


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = BOARD_SIZE):
        super().__init__()
        self.board_size = board_size

        self.conv1 = nn.Conv2d(4, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(5)])

        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)

        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 128)
        self.value_fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        policy_log_probs = F.log_softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_log_probs, value


def export_onnx(pth_path: str, onnx_path: str, opset: int = 17):
    if not os.path.exists(pth_path):
        raise FileNotFoundError(f"Model file not found: {pth_path}")

    model = PolicyValueNet()
    state_dict = torch.load(pth_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 4, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["policy_log_probs", "value"],
        dynamic_axes={
            "input": {0: "batch"},
            "policy_log_probs": {0: "batch"},
            "value": {0: "batch"},
        },
    )

    print(f"Exported ONNX model to: {onnx_path}")


def main():
    parser = argparse.ArgumentParser(description="Export Gobang AlphaZero V2 PyTorch model to ONNX")
    parser.add_argument("--pth", default=DEFAULT_PTH, help="Path to .pth model file")
    parser.add_argument("--onnx", default=DEFAULT_ONNX, help="Path to output .onnx file")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    export_onnx(args.pth, args.onnx, args.opset)


if __name__ == "__main__":
    main()
