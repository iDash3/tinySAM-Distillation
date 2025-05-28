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


def run_training(data_dir: Optional[Path] = None):
    """Run the training script (placeholder for now)."""
    print("\n" + "="*40)
    print("Step 3: MODEL TRAINING")
    print("="*40)
    
    print("Training script not yet implemented.")
    print("This is where the SAM distillation training would occur.")
    print(f"Dataset location: {data_dir or './datasets/sam'}")
    
    # Placeholder for future training script call:
    # training_script = Path("train.py")
    # if training_script.exists():
    #     subprocess.run([sys.executable, str(training_script), "--data-dir", str(data_dir)])
    # else:
    #     print("Training script not found.")


def main():
    """Main pipeline orchestrator."""
    parser = argparse.ArgumentParser(description="SAM Distillation Pipeline")
    parser.add_argument("--data-dir", type=Path, 
                       help="Directory to store dataset (default: ./datasets/sam)")
    parser.add_argument("--skip-install", action="store_true",
                       help="Skip requirements installation")
    parser.add_argument("--skip-dataset", action="store_true",
                       help="Skip dataset preparation")
    parser.add_argument("--skip-training", action="store_true",
                       help="Skip training (dataset preparation only)")
    parser.add_argument("--non-interactive", action="store_true",
                       help="Run without user prompts")
    
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
    print("3. Run model training")
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
    
    # Step 3: Run training
    if not args.skip_training:
        run_training(args.data_dir)
    else:
        print("\nStep 3: Skipping training...")
        print("Dataset preparation complete!")
    
    print("\n" + "="*40)
    print("PIPELINE COMPLETE")
    print("="*40)


if __name__ == "__main__":
    main() 