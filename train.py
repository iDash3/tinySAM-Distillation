import argparse
import logging
import os
from pathlib import Path
from typing import Optional

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

def scale_to_255(tensor):
    """Scale tensor from [0,1] to [0,255] range."""
    return tensor * 255.0

def get_transform():
    return Compose([
        resize_and_pad,  # Direct function reference instead of Lambda
        ToTensor(),      # uint8 [0–255] -> float [0.0–1.0]
        scale_to_255,    # Named function instead of lambda
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
    p = argparse.ArgumentParser(description="Train miniSAM model via knowledge distillation")
    p.add_argument("--img-dir",    type=Path, required=True, help="Directory containing training images")
    p.add_argument("--out-dir",    type=Path, required=True, help="Output directory for checkpoints and logs")
    p.add_argument("--sam-check",  type=Path, required=True, help="Path to SAM teacher model checkpoint")
    p.add_argument("--batch-size", type=int,   default=16, help="Training batch size")
    p.add_argument("--epochs",     type=int,   default=50, help="Number of training epochs")
    p.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    p.add_argument("--wd",         type=float, default=1e-5, help="Weight decay")
    p.add_argument("--val-split",  type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--save-freq",  type=int,   default=5, help="Checkpoint save frequency (epochs)")
    p.add_argument("--num-workers", type=int,  default=4, help="Number of dataloader workers")
    p.add_argument("--use-amp",    action="store_true", help="Use automatic mixed precision training")
    return p.parse_args()

# ===============================
# Pipeline Training Function
# ===============================
def run_training(
    img_dir: Path,
    sam_checkpoint: Path,
    out_dir: Optional[Path] = None,
    batch_size: int = 16,
    epochs: int = 50,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    val_split: float = 0.1,
    save_freq: int = 5,
    num_workers: int = 4,
    use_amp: bool = True
) -> bool:
    """
    Run training from the pipeline. Returns True if successful, False otherwise.
    
    Args:
        img_dir: Directory containing training images
        sam_checkpoint: Path to SAM teacher model checkpoint
        out_dir: Output directory for checkpoints (default: ./models/tiny_sam)
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        val_split: Validation split ratio
        save_freq: Checkpoint save frequency
        num_workers: Number of dataloader workers
        use_amp: Use automatic mixed precision training
    """
    try:
        # Set default output directory
        if out_dir is None:
            out_dir = Path("./models/tiny_sam")
        
        # Setup logging
        out_dir.mkdir(parents=True, exist_ok=True)
        log_file = out_dir / "training.log"
        
        # Configure logging to write to both file and console
        # Only configure if not already configured to avoid conflicts
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s %(levelname)s: %(message)s",
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )
        else:
            # Add file handler to existing logger if needed
            logger = logging.getLogger()
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
            logger.addHandler(file_handler)
        
        logging.info("="*50)
        logging.info("Starting miniSAM training pipeline")
        logging.info("="*50)
        logging.info(f"Image directory: {img_dir}")
        logging.info(f"SAM checkpoint: {sam_checkpoint}")
        logging.info(f"Output directory: {out_dir}")
        logging.info(f"Batch size: {batch_size}")
        logging.info(f"Epochs: {epochs}")
        logging.info(f"Learning rate: {lr}")
        logging.info(f"Workers: {num_workers}")
        logging.info(f"Mixed precision: {use_amp}")
        
        return _train_model(
            img_dir=img_dir,
            sam_checkpoint=sam_checkpoint,
            out_dir=out_dir,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            val_split=val_split,
            save_freq=save_freq,
            num_workers=num_workers,
            use_amp=use_amp
        )
        
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False

# ===============================
# Core Training Function
# ===============================
def _train_model(
    img_dir: Path,
    sam_checkpoint: Path,
    out_dir: Path,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    val_split: float,
    save_freq: int,
    num_workers: int,
    use_amp: bool
) -> bool:
    """Core training logic - refactored from main() to be reusable."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # ─── Dataset & DataLoaders ────────────────────────────────────────────────
    try:
        full_ds = ImageDataset(img_dir, get_transform())
    except RuntimeError as e:
        logging.error(f"Failed to load dataset: {e}")
        return False
        
    val_size = int(len(full_ds) * val_split)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])

    logging.info(f"Dataset split -> train {len(train_ds)}, val {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,     
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),  # Use fewer workers for validation    
        pin_memory=True,
    )

    # ─── Teacher (SAM) Setup ─────────────────────────────────────────────────
    try:
        sam = sam_model_registry["vit_h"](checkpoint=str(sam_checkpoint))
        sam.to(device).eval()
        logging.info("SAM teacher model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load SAM model: {e}")
        return False

    # ─── Student + Optimizer + Scheduler ────────────────────────────────────
    student = TinyViT(
        img_size=1024, in_chans=3,
        embed_dims=[64,128,160,320],
        depths=[2,2,6,2],
        num_heads=[2,4,5,10],
        window_sizes=[7,7,14,7],
        drop_path_rate=0.0
    ).to(device)

    optimizer = AdamW(student.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Find existing checkpoints with more robust parsing
    ckpt_files = []
    try:
        for ckpt_path in out_dir.glob("checkpoint_epoch*.pth"):
            # Extract epoch number more safely
            stem = ckpt_path.stem  # e.g., "checkpoint_epoch005"
            if "epoch" in stem:
                epoch_str = stem.split("epoch")[-1]
                try:
                    epoch_num = int(epoch_str)
                    ckpt_files.append((epoch_num, ckpt_path))
                except ValueError:
                    logging.warning(f"Skipping malformed checkpoint: {ckpt_path}")
        # Sort by epoch number
        ckpt_files.sort(key=lambda x: x[0])
        ckpt_files = [path for _, path in ckpt_files]  # Extract just the paths
    except Exception as e:
        logging.warning(f"Error scanning checkpoints: {e}")
        ckpt_files = []

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

    val_log_path = out_dir / "val_log.csv"
    if not val_log_path.exists():
        with open(val_log_path, "w") as f:
            f.write("epoch,avg_train_loss,avg_val_loss\n")

    # ─── Training Loop ───────────────────────────────────────────────────────
    try:
        for epoch in range(start_epoch, epochs + 1):
            student.train()
            total_train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)

            for batch_idx, imgs in enumerate(pbar, start=1):
                imgs = imgs.to(device, non_blocking=True)

                # 1) teacher forward
                with torch.no_grad():
                    teacher_latents = sam.image_encoder(imgs)

                # 2) student forward + loss
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=use_amp):
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
                for imgs in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]  ", leave=False):
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
            if epoch % save_freq == 0 or epoch == epochs:
                ckpt = {
                    "epoch": epoch,
                    "student_state":   student.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state":    scaler.state_dict(),
                }
                save_checkpoint(ckpt, out_dir, epoch)

        logging.info("Training completed successfully!")
        return True
        
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        return False
    except Exception as e:
        logging.error(f"Training failed: {e}")
        return False

# ===============================
# Main Training Function
# ===============================
def main():
    args = parse_args()
    
    # Call the core training function
    success = _train_model(
        img_dir=args.img_dir,
        sam_checkpoint=args.sam_check,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        val_split=args.val_split,
        save_freq=args.save_freq,
        num_workers=args.num_workers,
        use_amp=args.use_amp
    )
    
    if not success:
        exit(1)

if __name__ == "__main__":
    main()