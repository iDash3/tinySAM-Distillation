"""
SAM Distillation Pipeline
Main orchestrator for the complete SAM model distillation process.
Handles dataset preparation, training, and model optimization.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

# SAM model weights configuration
SAM_MODELS = {
    "vit_h": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "filename": "sam_vit_h_4b8939.pth",
        "size_mb": 2564  # Approximate size in MB
    }
}


def download_sam_weights(model_variant: str = "vit_h", models_dir: Optional[Path] = None) -> Path:
    """Download SAM model weights if they don't exist."""
    models_dir = models_dir or Path("./models")
    sam_dir = models_dir / "sam"
    sam_dir.mkdir(parents=True, exist_ok=True)
    
    if model_variant not in SAM_MODELS:
        raise ValueError(f"Unknown model variant: {model_variant}. Available: {list(SAM_MODELS.keys())}")
    
    model_info = SAM_MODELS[model_variant]
    weights_path = sam_dir / model_info["filename"]
    
    # Check if weights already exist
    if weights_path.exists():
        file_size_mb = weights_path.stat().st_size / (1024 * 1024)
        print(f"SAM weights already exist: {weights_path}")
        print(f"File size: {file_size_mb:.1f}MB")
        return weights_path
    
    print(f"Downloading SAM {model_variant.upper()} weights...")
    print(f"URL: {model_info['url']}")
    print(f"Destination: {weights_path}")
    print(f"Expected size: ~{model_info['size_mb']}MB")
    print("This may take a few minutes depending on your internet connection...")
    
    try:
        import requests
        from tqdm import tqdm
        
        response = requests.get(model_info["url"], stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(weights_path, 'wb') as f, tqdm(
            desc="Downloading",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
        
        # Verify download
        if weights_path.exists():
            file_size_mb = weights_path.stat().st_size / (1024 * 1024)
            print(f"Download complete! File size: {file_size_mb:.1f}MB")
            return weights_path
        else:
            raise RuntimeError("Download failed - file not found after download")
            
    except ImportError:
        print("Error: requests and tqdm are required for downloading.")
        print("Please install with: pip install requests tqdm")
        return None
    except Exception as e:
        print(f"Failed to download SAM weights: {e}")
        if weights_path.exists():
            weights_path.unlink()  # Clean up partial download
        return None


def install_requirements():
    """Install required packages from requirements.txt."""
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("Error: requirements.txt not found.")
        print("Please ensure requirements.txt exists in the current directory.")
        return False
    
    print("Installing requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ], check=True)
        print("Requirements installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        print("You can manually install with: pip install -r requirements.txt")
        return False


def prepare_dataset(data_dir: Optional[Path] = None, interactive: bool = True) -> bool:
    """Prepare the SAM dataset for training."""
    try:
        from sam_processor import SAMProcessor, interactive_download
    except ImportError as e:
        print(f"Error importing sam_processor: {e}")
        print("Please ensure requirements are installed: pip install -r requirements.txt")
        return False
    
    print("\n" + "="*40)
    print("Step 2: DATASET PREPARATION")
    print("="*40)
    
    processor = SAMProcessor(data_dir)
    
    # Check if list.txt exists
    list_file = Path("list.txt")
    if not list_file.exists():
        print("Error: list.txt file not found.")
        print("Please ensure the dataset URL list file exists in the current directory.")
        return False
    
    # Check current dataset status
    status = processor.get_dataset_status()
    dataset_exists = status["download_progress"]["total"] > 0 or status["image_count"] > 0
    
    if dataset_exists:
        downloaded = status["download_progress"]["downloaded"]
        total = status["download_progress"]["total"] 
        images = status["image_count"]
        
        print(f"Existing dataset found:")
        print(f"  Files downloaded: {downloaded}/{total}")
        print(f"  Images extracted: {images/1000:.1f}k")
        print(f"  Validated: {'Yes' if status.get('validated', False) else 'No'}")
        
        if interactive:
            choice = input("\nUse existing dataset or download more? (use/download) [use]: ").strip().lower()
            if not choice:  # Empty input defaults to 'use'
                choice = 'use'
            
            if choice.startswith('u'):
                print("Using existing dataset.")
                return True
            elif choice.startswith('d'):
                return interactive_download(data_dir)
            else:
                print("Invalid choice. Using existing dataset.")
                return True
        else:
            # Non-interactive mode: use existing if sufficient, otherwise download more
            if images > 100:  # Arbitrary threshold for "sufficient" data
                print("Sufficient dataset found. Using existing dataset.")
                return True
            else:
                print("Insufficient dataset. Downloading more...")
                file_list_content = list_file.read_text()
                return processor.download_and_extract_batch(file_list_content, 0.1)  # Download 10% more
    else:
        print("No existing dataset found.")
        if interactive:
            return interactive_download(data_dir)
        else:
            # Non-interactive mode: download 10% by default
            print("Downloading 10% of dataset for training...")
            file_list_content = list_file.read_text()
            return processor.download_and_extract_batch(file_list_content, 0.1)


def run_training(data_dir: Optional[Path] = None, sam_weights_path: Optional[Path] = None, 
                batch_size: int = 8, epochs: int = 20, lr: float = 1e-4, 
                num_workers: int = 4, save_freq: int = 5):
    """Run the training script."""
    print("\n" + "="*40)
    print("Step 4: MODEL TRAINING")
    print("="*40)
    
    if not sam_weights_path or not sam_weights_path.exists():
        print("Error: SAM weights not found. Cannot proceed with training.")
        return False
    
    # Set up paths
    data_dir = data_dir or Path("./datasets/sam")
    images_dir = data_dir / "images"
    output_dir = Path("./models/tiny_sam")
    
    # Check if dataset exists
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print("Please run dataset preparation first.")
        return False
    
    # Count images
    image_files = [f for f in images_dir.iterdir() 
                   if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    
    if len(image_files) == 0:
        print("Error: No images found in dataset directory.")
        print("Please run dataset preparation first.")
        return False
    
    print(f"Found {len(image_files)/1000:.1f}k images for training")
    print(f"Dataset location: {images_dir}")
    print(f"SAM weights: {sam_weights_path}")
    print(f"Output directory: {output_dir}")
    print(f"Training config: {epochs} epochs, batch size {batch_size}, LR {lr}")
    
    try:
        # Import training module after requirements are installed
        from train import run_training as train_model
        
        print("\nStarting training...")
        print("This may take several hours depending on your hardware.")
        
        # Run training with user-specified parameters
        success = train_model(
            img_dir=images_dir,
            sam_checkpoint=sam_weights_path,
            out_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            weight_decay=1e-5,
            val_split=0.1,
            save_freq=save_freq,
            num_workers=num_workers,
            use_amp=True   # Enable mixed precision by default
        )
        
        if success:
            print(f"\nTraining completed successfully!")
            print(f"Model checkpoints saved to: {output_dir}")
            print(f"Training logs saved to: {output_dir}/training.log")
            print(f"Validation metrics saved to: {output_dir}/val_log.csv")
        else:
            print("\nTraining failed. Check the logs for details.")
            
        return success
        
    except ImportError as e:
        print(f"Error importing training module: {e}")
        print("Please ensure all requirements are installed.")
        return False
    except Exception as e:
        print(f"Training failed with error: {e}")
        return False


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="SAM Distillation Pipeline")
    parser.add_argument("--data-dir", type=Path, 
                       help="Directory to store dataset (default: ./datasets/sam)")
    parser.add_argument("--models-dir", type=Path,
                       help="Directory to store model weights (default: ./models)")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip requirements installation")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset preparation")
    parser.add_argument("--skip-weights", action="store_true",
                       help="Skip SAM weights download")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (dataset preparation only)")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run without user prompts")
    
    # Training parameters
    training_group = parser.add_argument_group("training parameters")
    training_group.add_argument("--batch-size", type=int, default=4,
                               help="Training batch size (default: 4)")
    training_group.add_argument("--epochs", type=int, default=20,
                               help="Number of training epochs (default: 20)")
    training_group.add_argument("--lr", type=float, default=1e-4,
                               help="Learning rate (default: 1e-4)")
    training_group.add_argument("--num-workers", type=int, default=10,
                               help="Number of dataloader workers (default: 10)")
    training_group.add_argument("--save-freq", type=int, default=1,
                               help="Checkpoint save frequency in epochs (default: 1)")
    
    args = parser.parse_args()
    
    print("\n" * 3)
    print("██████ ██████  ██   ██  ██   ██")
    print("  ██     ██    ███  ██  ██   ██")
    print("  ██     ██    ██ █ ██   ██ ██")
    print("  ██     ██    ██  ███    ██")
    print("  ██   ██████  ██   ██    ██      sam")
    print("\n" * 3)

    print("This pipeline will:")
    print("1. Install required dependencies")
    print("2. Prepare the SAM dataset (with automatic validation)")
    print("3. Download SAM weights")
    print("4. Run model training")
    print("\n" * 4)
    
    interactive = not args.non_interactive
    
    # Step 1: Install requirements
    if not args.skip_install:
        print("\n" + "="*40)
        print("Step 1: INSTALLING REQUIREMENTS")
        print("="*40)
        if not install_requirements():
            print("Failed to install requirements. Exiting.")
            sys.exit(1)
    else:
        print("\nStep 1: Skipping requirements installation...")
    
    # Step 2: Prepare dataset
    if not args.skip_dataset:
        if not prepare_dataset(args.data_dir, interactive):
            print("Dataset preparation failed. Exiting.")
            sys.exit(1)
    else:
        print("\nStep 2: Skipping dataset preparation...")
    
    # Step 3: Download SAM weights
    sam_weights_path = None
    if not args.skip_weights and not args.skip_training:
        print("\n" + "="*40)
        print("Step 3: DOWNLOADING SAM WEIGHTS")
        print("="*40)
        sam_weights_path = download_sam_weights(models_dir=args.models_dir)
        if not sam_weights_path:
            print("Failed to download SAM weights. Exiting.")
            sys.exit(1)
    else:
        print("\nStep 3: Skipping SAM weights download...")
    
    # Step 4: Run training
    if not args.skip_training:
        # Get training parameters from user if in interactive mode
        if interactive:
            print("\n" + "="*40)
            print("TRAINING CONFIGURATION")
            print("="*40)
            
            print("\nBATCH SIZE")
            print("Batch size determines how many images are processed simultaneously.")
            print("- Larger batch sizes (16-32): Faster training, better gradients, but need more GPU memory")
            print("- Smaller batch sizes (4-8): Slower training, but work with limited GPU memory")
            print("- If you get 'CUDA out of memory' errors, reduce batch size")
            
            while True:
                try:
                    batch_input = input(f"\nEnter batch size (4-32) [4]: ").strip()
                    if not batch_input:
                        args.batch_size = 4
                        break
                    batch_size = int(batch_input)
                    if 1 <= batch_size <= 64:
                        args.batch_size = batch_size
                        break
                    else:
                        print("Please enter a batch size between 1 and 64.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"\nNUMBER OF WORKERS")
            print("Workers control how many CPU processes load data in parallel.")
            print("- More workers (8-16): Faster data loading, but use more RAM")
            print("- Fewer workers (2-4): Slower data loading, but use less RAM")
            print("- If you get 'too many open files' or RAM errors, reduce workers")
            print("- Recommended: Start with 4, increase if you have >16GB RAM")
            
            while True:
                try:
                    workers_input = input(f"\nEnter number of workers (1-16) [10]: ").strip()
                    if not workers_input:
                        args.num_workers = 10
                        break
                    num_workers = int(workers_input)
                    if 1 <= num_workers <= 32:
                        args.num_workers = num_workers
                        break
                    else:
                        print("Please enter a number of workers between 1 and 32.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"\nNUMBER OF EPOCHS")
            print("Epochs determine how many times the model sees the entire dataset.")
            print("- More epochs (10-15): Better training, but takes longer and may overfit")
            print("- Fewer epochs (3-8): Faster training, but may underfit")
            print("- Recommended: Start with 5-10 for distillation tasks")
            print("- You can always resume training from checkpoints later")
            
            while True:
                try:
                    epochs_input = input(f"\nEnter number of epochs (1-15) [5]: ").strip()
                    if not epochs_input:
                        args.epochs = 5
                        break
                    epochs = int(epochs_input)
                    if 1 <= epochs <= 15:
                        args.epochs = epochs
                        break
                    else:
                        print("Please enter a number of epochs between 1 and 15.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print(f"\nTraining Configuration:")
            print(f"  Batch size: {args.batch_size}")
            print(f"  Workers: {args.num_workers}")
            print(f"  Epochs: {args.epochs}")
            print(f"  Learning rate: {args.lr}")
            
            print(f"\nTroubleshooting Tips:")
            print(f"  - GPU memory error: Reduce batch size to {max(1, args.batch_size // 2)}")
            print(f"  - Data loading slow: Increase workers to {min(16, args.num_workers * 2)}")
            print(f"  - RAM usage high: Reduce workers to {max(1, args.num_workers // 2)}")
            
            confirm = input(f"\nStart training with these settings? (y/n) [y]: ").strip().lower()
            if confirm and not confirm.startswith('y'):
                print("Training cancelled.")
                return
        
        if not run_training(args.data_dir, sam_weights_path, args.batch_size, args.epochs, args.lr, args.num_workers, args.save_freq):
            print("Training failed. Exiting.")
            sys.exit(1)
    else:
        print("\nStep 4: Skipping training...")
        print("Setup complete!")
    
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)


if __name__ == "__main__":
    main() 