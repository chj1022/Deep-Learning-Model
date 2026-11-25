import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import nibabel as nib
import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base * 4, base * 8)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)

        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


def normalize_ct_fast(x: np.ndarray, clip_min: float, clip_max: float) -> np.ndarray:
    x = np.clip(x, clip_min, clip_max)
    x = (x - clip_min) / (clip_max - clip_min + 1e-6)
    x = x * 2.0 - 1.0
    return x.astype(np.float32, copy=False)


def pad_to_multiple_of_8(img2d: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = img2d.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8

    pad_h = new_h - h
    pad_w = new_w - w

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return img2d, (0, 0, 0, 0)

    fill = float(img2d.min())
    img_pad = np.pad(img2d, ((top, bottom), (left, right)), mode="constant", constant_values=fill)
    return img_pad, (top, bottom, left, right)


def unpad_2d(arr2d: np.ndarray, pad: Tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = pad
    if top == bottom == left == right == 0:
        return arr2d
    h, w = arr2d.shape
    return arr2d[top:h - bottom, left:w - right]



def load_model(ckpt_path: Path, device: str, num_classes: int) -> nn.Module:
      try:
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
    except TypeError:
              ckpt = torch.load(str(ckpt_path), map_location=device)

    model = UNetSmall(in_channels=1, num_classes=num_classes, base=32).to(device)

      if isinstance(ckpt, dict) and "model_state" in ckpt and isinstance(ckpt["model_state"], dict):
        state = ckpt["model_state"]
    elif isinstance(ckpt, dict):
        state = ckpt
    else:
        raise ValueError("Unsupported checkpoint format. Expect a dict containing model weights.")

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def infer_volume(
    model: nn.Module,
    vol: np.ndarray,
    device: str,
    clip_min: float,
    clip_max: float,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    if batch_size <= 0 or clip_min >= clip_max:
        raise ValueError("Invalid batch size or CT clipping range.")
    if vol.ndim != 3:
        raise ValueError(f"Only 3D NIfTI supported. Got shape={vol.shape}")

    h, w, z_slices = vol.shape
    out = np.zeros((h, w, z_slices), dtype=np.int16)

    amp_enabled = use_amp and str(device).startswith("cuda")

    z = 0
    while z < z_slices:
        z_end = min(z + batch_size, z_slices)

        batch_imgs = []
        pads = []
        for zi in range(z, z_end):
            img2d = vol[..., zi].astype(np.float32, copy=False)
            img2d = normalize_ct_fast(img2d, clip_min, clip_max)
            img2d_pad, pad = pad_to_multiple_of_8(img2d)
            batch_imgs.append(img2d_pad)
            pads.append(pad)

        batch_np = np.stack(batch_imgs, axis=0)  # [N,H,W]
        batch_t = torch.from_numpy(batch_np).unsqueeze(1).to(device=device, dtype=torch.float32, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(batch_t)             # [N,C,H,W]
            pred = torch.argmax(logits, dim=1)  # [N,H,W]

        pred_np = pred.detach().cpu().numpy().astype(np.int16, copy=False)

        for i, zi in enumerate(range(z, z_end)):
            pred2d = unpad_2d(pred_np[i], pads[i])
            if pred2d.shape != (h, w):
                raise ValueError(f"Unexpected prediction shape: {pred2d.shape}; expected {(h, w)}")
            out[..., zi] = pred2d

        z = z_end

    return out


def save_nifti_like(ref_nii: nib.Nifti1Image, data: np.ndarray, out_path: Path) -> None:
    out_nii = nib.Nifti1Image(data, affine=ref_nii.affine, header=ref_nii.header.copy())
    out_nii.set_data_dtype(np.int16)
    out_nii.header.set_slope_inter(1.0, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_nii, str(out_path))


def collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        return sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))],
            key=lambda x: x.name
        )
    raise FileNotFoundError(f"Input not found: {input_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="")
    parser.add_argument("--output", type=str, default="", help="")
    parser.add_argument("--output-dir", type=str, default="", help="）")

    parser.add_argument("")
    parser.add_argument("--num-classes", type=int, default=9)
    parser.add_argument("--clip-min", type=float, default=-200.0)
    parser.add_argument("--clip-max", type=float, default=300.0)
    parser.add_argument("--batch-size", type=int, default=12, help="")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--no-amp", action="store_true", help="")
    args = parser.parse_args()

    in_path = Path(args.input)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = args.device
    model = load_model(ckpt_path, device=device, num_classes=int(args.num_classes))

    inputs = collect_inputs(in_path)
    if len(inputs) == 0:
        raise RuntimeError("No input nifti files found.")

    if in_path.is_file():
        out_paths = [Path(args.output)] if args.output else [in_path.parent / "Segmentation.nii"]
    else:
        out_dir = Path(args.output_dir) if args.output_dir else (in_path / "pred")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_paths = [out_dir / f"{p.stem}_Segmentation.nii" for p in inputs]

    if any(dst.resolve() in {src.resolve() for src in inputs} for dst in out_paths):
        raise ValueError("Output must not overwrite an input CT file.")

    for src, dst in zip(inputs, out_paths):
        print(f"[INFO] Inference: {src}")
        nii = nib.load(str(src))
        vol = np.asanyarray(nii.dataobj)

        seg = infer_volume(
            model=model,
            vol=vol,
            device=device,
            clip_min=float(args.clip_min),
            clip_max=float(args.clip_max),
            batch_size=int(args.batch_size),
            use_amp=(not args.no_amp),
        )
        save_nifti_like(nii, seg, dst)
        print(f"[OK] Saved: {dst}")

    print("[DONE] All inference finished.")


if __name__ == "__main__":
    main()
