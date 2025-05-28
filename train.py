import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda
from PIL import Image
from tqdm.auto import tqdm

from segment_anything import sam_model_registry
from model import TinyViT

# ===============================
# Constants for SAM pre-processing
# ===============================
MEAN = [123.675, 116.28, 103.53]
STD = [ 58.395,  57.12,  57.375]
TARGET = 1024

def resize_and_pad(img: Image.Image) -> Image.Image:
    """Resize longest side to TARGET and pad to a square."""
    w, h = img.size
    scale = TARGET / max(w, h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    img = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (TARGET, TARGET), (0, 0, 0))
    padded.paste(img, (0, 0))

    return padded

def get_transform():
    return Compose([
        Lambda(resize_and_pad), # on-the-fly resize+pad
        ToTensor(),             # uint8 [0–255] -> float [0.0–1.0]
        Lambda(lambda x: x * 255.0),
        Normalize(mean=MEAN, std=STD),
    ])

# ===============================
# Dataset Class
# ===============================
class ImageDataset(Dataset):
    """Loads original images & applies on-the-fly preprocessing."""
    def __init__(self, img_dir: Path, transform):
        exts = ("*.jpg", "*.jpeg", "*.png")
        self.paths = sorted(p for ext in exts for p in img_dir.glob(ext))
        if not self.paths:
            raise RuntimeError(f"No images found in {img_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")

        return self.transform(img)

# ===============================
# Utility Functions
# ===============================
def save_checkpoint(state, out_dir: Path, epoch: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"checkpoint_epoch{epoch:03d}.pth"
    torch.save(state, fname)
    logging.info(f"Saved checkpoint: {fname}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img-dir",    type=Path, required=True)
    p.add_argument("--out-dir",    type=Path, required=True)
    p.add_argument("--sam-check",  type=Path, required=True)
    p.add_argument("--batch-size", type=int,   default=16)
    p.add_argument("--epochs",     type=int,   default=50)
    p.add_argument("--lr",         type=float, default=1e-4)
    p.add_argument("--wd",         type=float, default=1e-5)
    p.add_argument("--val-split",  type=float, default=0.1)
    p.add_argument("--save-freq",  type=int,   default=5)
    p.add_argument("--use-amp",    action="store_true")
    return p.parse_args()

# ===============================
# Main Training Function
# ===============================
def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ─── Dataset & DataLoaders ────────────────────────────────────────────────
    full_ds = ImageDataset(args.img_dir, get_transform())
    val_size = int(len(full_ds) * args.val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    logging.info(f"Dataset split -> train {len(train_ds)}, val {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,     
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,      
        pin_memory=True,
    )

    # ─── Teacher (SAM) Setup ─────────────────────────────────────────────────
    sam = sam_model_registry["vit_h"](checkpoint=str(args.sam_check))
    sam.to(device).eval()

    # ─── Student + Optimizer + Scheduler ────────────────────────────────────
    student = TinyViT(
        img_size=1024, in_chans=3,
        embed_dims=[64,128,160,320],
        depths=[2,2,6,2],
        num_heads=[2,4,5,10],
        window_sizes=[7,7,14,7],
        drop_path_rate=0.0
    ).to(device)

    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    ckpt_files = sorted(
        args.out_dir.glob("checkpoint_epoch*.pth"),
        key=lambda p: int(p.stem.split("epoch")[1])
    )

    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        state       = torch.load(latest_ckpt, map_location=device)
        student.load_state_dict(state["student_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        scaler.load_state_dict(state["scaler_state"])
        start_epoch = state["epoch"] + 1
        logging.info(f"Resuming from checkpoint {latest_ckpt.name}, starting at epoch {start_epoch}")
    else:
        start_epoch = 1

    val_log_path = args.out_dir / "val_log.csv"
    if not val_log_path.exists():
        with open(val_log_path, "w") as f:
            f.write("epoch,avg_train_loss,avg_val_loss\n")

    # ─── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        student.train()
        total_train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False)

        for batch_idx, imgs in enumerate(pbar, start=1):
            imgs = imgs.to(device, non_blocking=True)

            # 1) teacher forward
            with torch.no_grad():
                teacher_latents = sam.image_encoder(imgs)

            # 2) student forward + loss
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                preds = student(imgs)
                loss  = criterion(preds, teacher_latents)

            # 3) backward + step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_train_loss += loss.item()
            pbar.set_postfix(train_loss=total_train_loss / batch_idx)

        avg_train = total_train_loss / len(train_loader)
        logging.info(f"Epoch {epoch}: avg train loss = {avg_train:.6f}")

        # ─── Validation ───────────────────────────────────────────────────────
        student.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for imgs in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False):
                imgs = imgs.to(device, non_blocking=True)
                teacher_latents = sam.image_encoder(imgs)
                preds = student(imgs)
                total_val_loss += criterion(preds, teacher_latents).item()

        avg_val = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch}: avg val   loss = {avg_val:.6f}")

        with open(val_log_path, "a") as f:
            f.write(f"{epoch},{avg_train:.6f},{avg_val:.6f}\n")

        scheduler.step()

        # ─── Checkpointing ────────────────────────────────────────────────────
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt = {
                "epoch": epoch,
                "student_state":   student.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state":    scaler.state_dict(),
            }
            save_checkpoint(ckpt, args.out_dir, epoch)

    logging.info("Training complete!")

if __name__ == "__main__":
    main()