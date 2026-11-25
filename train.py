import time
import random
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import nibabel as nib
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

    data_root: str = "data"
    out_dir: str = "outputs"
    log_file: str = "outputs/log.txt"
    best_model_path: str = "outputs/best_model.pt"

    num_classes: int = 9  
    include_bg_in_mean_dice: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    torch_threads: int = 8
    epochs: int = 60
    batch_size: int = 10
    lr: float = 3e-4
    weight_decay: float = 1e-4
    crop_h: int = 256
    crop_w: int = 256
    bg_slice_ratio: float = 0.10
    clip_min: float = -200.0
    clip_max: float = 300.0
    train_samples_per_epoch: int = 12000
    eval_samples: int = 3000
    cache_size: int = 16
    seed: int = 42
    tqdm_mininterval: float = 0.5


class TqdmConsoleHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)


def setup_logger(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = TqdmConsoleHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


LOGGER = logging.getLogger("train_logger")

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(cfg: Config) -> None:
    set_seed(cfg.seed)
    torch.set_num_threads(cfg.torch_threads)
    torch.backends.cudnn.benchmark = True  

class NiftiSliceDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        label_dir: Path,
        crop_hw: Tuple[int, int],
        is_train: bool,
        bg_slice_ratio: float,
        clip_min: float,
        clip_max: float,
        cache_size: int,
        max_len: int,
    ):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.crop_h, self.crop_w = crop_hw
        self.is_train = is_train
        self.bg_slice_ratio = float(bg_slice_ratio)
        self.clip_min = float(clip_min)
        self.clip_max = float(clip_max)
        self.cache_size = int(cache_size)
        self.virtual_len = int(max_len)

        self.image_files = sorted(
            [p for p in image_dir.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))],
            key=lambda x: x.name
        )
        self.label_files = sorted(
            [p for p in label_dir.iterdir() if p.is_file() and p.name.lower().endswith((".nii", ".nii.gz"))],
            key=lambda x: x.name
        )
        if len(self.image_files) == 0 or len(self.label_files) == 0:
            raise RuntimeError(f"Empty dataset: {image_dir} or {label_dir}")

        img_map = {self._case_id(p.name): p for p in self.image_files}
        lab_map = {self._case_id(p.name): p for p in self.label_files}
        common = sorted(set(img_map.keys()) & set(lab_map.keys()), key=lambda x: (len(x), x))
        if len(img_map) != len(self.image_files) or len(lab_map) != len(self.label_files):
            raise ValueError("Duplicate case IDs detected.")
        if set(img_map) != set(lab_map):
            raise ValueError("Unmatched image/label case IDs.")
        if len(common) == 0:
            raise RuntimeError("No matched image/label pairs found in given directories.")

        self.cases: List[Tuple[Path, Path, str]] = [(img_map[c], lab_map[c], c) for c in common]

        self.fg_index: List[Tuple[int, int]] = []
        self.bg_index: List[Tuple[int, int]] = []

        LOGGER.info(f"Indexing slices (train={is_train}) ...")
        for case_i, (img_p, lab_p, _) in enumerate(self.cases):
            lab_nii = nib.load(str(lab_p))
            img_nii = nib.load(str(img_p))
            if img_nii.shape != lab_nii.shape or not np.allclose(img_nii.affine, lab_nii.affine, atol=1e-4):
                raise ValueError(f"Image/label geometry mismatch: {lab_p}")
            lab = np.asanyarray(lab_nii.dataobj)
            if not np.all(np.isfinite(lab)) or np.any(lab != np.floor(lab)) or np.any(lab < 0) or np.any(lab >= CFG.num_classes):
                raise ValueError(f"Invalid label values: {lab_p}")
            lab = lab.astype(np.int16, copy=False)
            if lab.ndim != 3:
                raise ValueError(f"Label is not 3D: {lab_p}")

            z_slices = lab.shape[-1]
            for z in range(z_slices):
                sl = lab[..., z]
                if np.any(sl > 0):
                    self.fg_index.append((case_i, z))
                else:
                    self.bg_index.append((case_i, z))

        if len(self.fg_index) == 0:
            raise RuntimeError("No foreground slices found (label>0). Please verify labels.")

        LOGGER.info(f"Slice index built: fg={len(self.fg_index)}, bg={len(self.bg_index)}, cases={len(self.cases)}")

        self._cache: Dict[int, Dict[str, object]] = {}
        self._cache_lru: List[int] = []


    def _case_id(filename: str) -> str:
        stem = filename
        low = stem.lower()
        if low.endswith(".nii.gz"):
            stem = stem[:-7]
        elif low.endswith(".nii"):
            stem = stem[:-4]
        stem_low = stem.lower()

        m = re.search(r"-(\d+)$", stem_low)
        if m:
            return m.group(1).lstrip("0") or "0"
        nums = re.findall(r"(\d+)", stem_low)
        if nums:
            return nums[-1].lstrip("0") or "0"
        return stem_low

    def __len__(self) -> int:
        return self.virtual_len

    def _get_case_niis(self, case_i: int) -> Tuple[nib.Nifti1Image, nib.Nifti1Image]:
        if case_i in self._cache:
            if case_i in self._cache_lru:
                self._cache_lru.remove(case_i)
            self._cache_lru.append(case_i)
            return self._cache[case_i]["img_nii"], self._cache[case_i]["lab_nii"]

        img_p, lab_p, _ = self.cases[case_i]
        img_nii = nib.load(str(img_p))
        lab_nii = nib.load(str(lab_p))

        self._cache[case_i] = {"img_nii": img_nii, "lab_nii": lab_nii}
        self._cache_lru.append(case_i)

        if len(self._cache_lru) > self.cache_size:
            evict = self._cache_lru.pop(0)
            if evict in self._cache:
                del self._cache[evict]

        return img_nii, lab_nii

    def _normalize_ct_fast(self, x: np.ndarray) -> np.ndarray:
        x = np.clip(x, self.clip_min, self.clip_max)
        x = (x - self.clip_min) / (self.clip_max - self.clip_min + 1e-6)
        x = x * 2.0 - 1.0
        return x.astype(np.float32, copy=False)

    def _crop_or_pad(self, img2d: np.ndarray, lab2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        h, w = img2d.shape
        th, tw = self.crop_h, self.crop_w

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            fill = float(img2d.min())
            img2d = np.pad(img2d, ((top, bottom), (left, right)), mode="constant", constant_values=fill)
            lab2d = np.pad(lab2d, ((top, bottom), (left, right)), mode="constant", constant_values=0)
            h, w = img2d.shape

        if self.is_train:
            y0 = random.randint(0, h - th) if h > th else 0
            x0 = random.randint(0, w - tw) if w > tw else 0
        else:
            y0 = (h - th) // 2 if h > th else 0
            x0 = (w - tw) // 2 if w > tw else 0

        img2d = img2d[y0:y0 + th, x0:x0 + tw]
        lab2d = lab2d[y0:y0 + th, x0:x0 + tw]
        return img2d, lab2d

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.is_train:
            use_bg = (random.random() < self.bg_slice_ratio) and (len(self.bg_index) > 0)
            case_i, z = random.choice(self.bg_index if use_bg else self.fg_index)
        else:
            combined = self.fg_index + self.bg_index[: min(len(self.bg_index), len(self.fg_index))]
            case_i, z = combined[idx % len(combined)]

        img_nii, lab_nii = self._get_case_niis(case_i)

        img2d = np.asanyarray(img_nii.dataobj[..., z]).astype(np.float32, copy=False)
        lab2d = np.asanyarray(lab_nii.dataobj[..., z]).astype(np.int64, copy=False)

        img2d = self._normalize_ct_fast(img2d)
        img2d, lab2d = self._crop_or_pad(img2d, lab2d)

        img_t = torch.from_numpy(img2d).unsqueeze(0).float()  [1,H,W]
        lab_t = torch.from_numpy(lab2d).long()                 [H,W]
        return {"image": img_t, "label": lab_t}

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

def dice_per_class(pred: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-6) -> torch.Tensor:
    dices = []
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        denom = p.sum().float() + t.sum().float()
        dices.append((2.0 * inter + eps) / (denom + eps))
    return torch.stack(dices, dim=0)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, cfg: Config) -> Dict[str, float]:
    model.eval()
    total_correct = 0
    total_pixels = 0

    dice_sum = torch.zeros(cfg.num_classes, dtype=torch.float64)
    dice_count = 0

    pbar = tqdm(
        loader,
        desc="Eval",
        leave=False,
        dynamic_ncols=True,
        mininterval=cfg.tqdm_mininterval,
    )
    for batch in pbar:
        img = batch["image"].to(cfg.device, non_blocking=True)
        lab = batch["label"].to(cfg.device, non_blocking=True)

        logits = model(img)
        pred = torch.argmax(logits, dim=1)

        total_correct += (pred == lab).sum().item()
        total_pixels += lab.numel()

        d = dice_per_class(pred, lab, cfg.num_classes).cpu().double()
        dice_sum += d
        dice_count += 1

    dice_mean = (dice_sum / max(dice_count, 1)).numpy().tolist()
    acc = float(total_correct) / float(max(total_pixels, 1))

    start_c = 0 if cfg.include_bg_in_mean_dice else 1
    mean_dice = float(np.mean(dice_mean[start_c:]))

    metrics: Dict[str, float] = {"mean_dice": mean_dice, "pixel_acc": acc}
    for c in range(cfg.num_classes):
        metrics[f"dice_class_{c}"] = float(dice_mean[c])
    return metrics


def train_one_epoch(model, loader, optimizer, scaler, cfg: Config) -> float:
    model.train()
    ce_loss = nn.CrossEntropyLoss()
    use_amp = cfg.device.startswith("cuda")

    running = 0.0
    n_batches = 0

    pbar = tqdm(
        loader,
        desc="Train",
        leave=False,
        dynamic_ncols=True,
        mininterval=cfg.tqdm_mininterval,
    )
    for batch in pbar:
        img = batch["image"].to(cfg.device, non_blocking=True)
        lab = batch["label"].to(cfg.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(img)
            loss = ce_loss(logits, lab)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running += float(loss.item())
        n_batches += 1


        pbar.set_postfix_str(f"loss={loss.item():.4f}")

    return running / max(n_batches, 1)


def format_metrics(metrics: Dict[str, float], cfg: Config) -> str:
    parts = [f"mean_dice={metrics['mean_dice']:.4f}", f"pixel_acc={metrics['pixel_acc']:.4f}"]
    for c in range(1, cfg.num_classes):
        parts.append(f"dice[{c}]={metrics[f'dice_class_{c}']:.4f}")
    return " | ".join(parts)


def make_loader(ds: Dataset, batch_size: int, cfg: Config, drop_last: bool) -> DataLoader:
    kwargs = dict(
        batch_size=batch_size,
        shuffle=False,   
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=drop_last,
        persistent_workers=(cfg.num_workers > 0),
    ) 
    if cfg.num_workers > 0:
        kwargs["prefetch_factor"] = 4
    return DataLoader(ds, **kwargs)


def main():
    global LOGGER
    LOGGER = setup_logger(CFG.log_file)
    configure_runtime(CFG)
    Path(CFG.out_dir).mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Device: {CFG.device}")
    LOGGER.info(f"num_classes={CFG.num_classes} (labels expected: 0..{CFG.num_classes - 1})")
    LOGGER.info(f"epochs={CFG.epochs}, batch_size={CFG.batch_size}, lr={CFG.lr}, crop={CFG.crop_h}x{CFG.crop_w}")
    LOGGER.info(f"bg_slice_ratio={CFG.bg_slice_ratio}, clip=[{CFG.clip_min},{CFG.clip_max}]")
    LOGGER.info(f"threads={CFG.torch_threads}, workers={CFG.num_workers}")
    LOGGER.info(f"train_samples_per_epoch={CFG.train_samples_per_epoch}, eval_samples={CFG.eval_samples}")
    LOGGER.info(f"cache_size={CFG.cache_size}")
    LOGGER.info(f"log_file={CFG.log_file}")
    LOGGER.info(f"best_model_path={CFG.best_model_path}")

    data_root = Path(CFG.data_root)
    train_img_dir = data_root / "train" / "image"
    train_lab_dir = data_root / "train" / "label"
    val_img_dir = data_root / "val" / "image"
    val_lab_dir = data_root / "val" / "label"

    for p in [train_img_dir, train_lab_dir, val_img_dir, val_lab_dir]:
        if not p.exists():
            raise FileNotFoundError(f"Missing directory: {p}")

    train_ds = NiftiSliceDataset(
        image_dir=train_img_dir,
        label_dir=train_lab_dir,
        crop_hw=(CFG.crop_h, CFG.crop_w),
        is_train=True,
        bg_slice_ratio=CFG.bg_slice_ratio,
        clip_min=CFG.clip_min,
        clip_max=CFG.clip_max,
        cache_size=CFG.cache_size,
        max_len=CFG.train_samples_per_epoch,
    )
    val_ds = NiftiSliceDataset(
        image_dir=val_img_dir,
        label_dir=val_lab_dir,
        crop_hw=(CFG.crop_h, CFG.crop_w),
        is_train=False,
        bg_slice_ratio=0.0,
        clip_min=CFG.clip_min,
        clip_max=CFG.clip_max,
        cache_size=max(4, CFG.cache_size // 2),
        max_len=CFG.eval_samples,
    )

    overlap = {c[2] for c in train_ds.cases} & {c[2] for c in val_ds.cases}
    if overlap:
        raise ValueError(f"Overlapping case IDs in train and val: {sorted(overlap)}")

    train_loader = make_loader(train_ds, CFG.batch_size, CFG, drop_last=True)
    val_loader = make_loader(val_ds, CFG.batch_size, CFG, drop_last=False)

    model = UNetSmall(in_channels=1, num_classes=CFG.num_classes, base=32).to(CFG.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)

    use_amp = CFG.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda" if use_amp else "cpu", enabled=use_amp)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    best_mean_dice = -1.0
    best_epoch = -1
    best_metrics: Optional[Dict[str, float]] = None

    LOGGER.info("Start training ...")
    t0 = time.time()

    for epoch in range(1, CFG.epochs + 1):
        epoch_t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, CFG)
        scheduler.step()

        metrics = evaluate(model, val_loader, CFG)

        lr_now = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - epoch_t0

        
        LOGGER.info(
            f"Epoch {epoch:03d}/{CFG.epochs:03d} | "
            f"lr={lr_now:.6f} | train_loss={train_loss:.4f} | "
            f"Validation: {format_metrics(metrics, CFG)} | time={elapsed:.1f}s"
        )

      
        if metrics["mean_dice"] > best_mean_dice:
            best_mean_dice = metrics["mean_dice"]
            best_epoch = epoch
            best_metrics = metrics

            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "cfg": CFG.__dict__,
                "best_mean_dice": best_mean_dice,
                "best_metrics": best_metrics,
            }
            torch.save(ckpt, CFG.best_model_path)
            LOGGER.info(f"[SAVE] New best model: {CFG.best_model_path} (mean_dice={best_mean_dice:.4f})")

    total = time.time() - t0
    LOGGER.info(f"Training finished. Total time: {total/60.0:.1f} min")

    if best_metrics is None:
        LOGGER.info("No best model was saved (unexpected).")
        return

    LOGGER.info(f"Best epoch: {best_epoch} | best_mean_dice={best_mean_dice:.4f}")
    LOGGER.info("Best validation metrics:")
    LOGGER.info(format_metrics(best_metrics, CFG))

    dices = [best_metrics.get(f"dice_class_{c}", float("nan")) for c in range(CFG.num_classes)]
    mean_muscle_dice = float(np.mean(dices[1:]))
    LOGGER.info(f"Summary | MeanDice(foreground)={mean_muscle_dice:.4f} | PixelAcc={best_metrics['pixel_acc']:.4f}")


if __name__ == "__main__":
    main()
