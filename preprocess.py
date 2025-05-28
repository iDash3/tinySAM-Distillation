"""
SAM Image Preprocessor
Standalone script to preprocess SAM images with resizing and padding.
This creates padded square images to avoid on-the-fly computation during training.
Note: This approximately doubles storage requirements.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from tqdm import tqdm


@dataclass
class ProcessingStats:
    """Statistics for image preprocessing."""
    total_images: int
    processed: int
    errors: int
    skipped: int
    start_time: float
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def success_rate(self) -> float:
        return (self.processed / self.total_images * 100) if self.total_images > 0 else 0.0


def resize_and_pad_worker(args: Tuple[Path, Path, int]) -> Tuple[bool, str, Optional[str]]:
    """Worker function to process a single image. Returns (success, filename, error_msg)."""
    src_path, dst_dir, target_size = args
    dst_path = dst_dir / src_path.name
    
    # Skip if already exists
    if dst_path.exists():
        return True, src_path.name, None
    
    try:
        # Load and convert to RGB
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            
            # Resize: scale longest side to target_size
            w, h = img.size
            scale = target_size / max(w, h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            
            # Pad to square with black background
            padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
            padded.paste(img, (0, 0))  # Top-left alignment
            
            # Save processed image
            padded.save(dst_path, "JPEG", quality=95, optimize=True)
            
        return True, src_path.name, None
        
    except Exception as e:
        return False, src_path.name, str(e)


class ImagePreprocessor:
    """Handles batch preprocessing of SAM images with resizing and padding."""
    
    def __init__(self, src_dir: Path, dst_dir: Path, target_size: int = 1024):
        self.src_dir = Path(src_dir)
        self.dst_dir = Path(dst_dir)
        self.target_size = target_size
        
        # Validate directories
        if not self.src_dir.exists():
            raise ValueError(f"Source directory does not exist: {self.src_dir}")
        
        # Create destination directory
        self.dst_dir.mkdir(parents=True, exist_ok=True)
    
    def get_image_files(self) -> List[Path]:
        """Get list of image files to process."""
        extensions = {".jpg", ".jpeg", ".png"}
        files = [
            f for f in self.src_dir.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        ]
        return sorted(files)
    
    def process_images(self, max_workers: Optional[int] = None, verbose: bool = True) -> ProcessingStats:
        """Process all images with multiprocessing."""
        image_files = self.get_image_files()
        
        if not image_files:
            print(f"No image files found in {self.src_dir}")
            return ProcessingStats(0, 0, 0, 0, time.time())
        
        # Setup processing
        max_workers = max_workers or min(os.cpu_count() or 1, 8)
        stats = ProcessingStats(
            total_images=len(image_files),
            processed=0,
            errors=0,
            skipped=0,
            start_time=time.time()
        )
        
        print(f"Processing {len(image_files)} images...")
        print(f"Source: {self.src_dir}")
        print(f"Destination: {self.dst_dir}")
        print(f"Target size: {self.target_size}x{self.target_size}")
        print(f"Using {max_workers} worker processes")
        print("-" * 50)
        
        # Prepare arguments for workers
        worker_args = [
            (img_path, self.dst_dir, self.target_size)
            for img_path in image_files
        ]
        
        # Process with multiprocessing and progress bar
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            if verbose:
                results = list(tqdm(
                    executor.map(resize_and_pad_worker, worker_args),
                    total=len(worker_args),
                    desc="Processing"
                ))
            else:
                results = list(executor.map(resize_and_pad_worker, worker_args))
        
        # Collect statistics
        error_files = []
        for success, filename, error_msg in results:
            if success:
                if error_msg is None:  # Newly processed
                    stats.processed += 1
                else:  # Already existed (skipped)
                    stats.skipped += 1
            else:
                stats.errors += 1
                error_files.append((filename, error_msg))
        
        # Print results
        self.print_summary(stats, error_files, verbose)
        return stats
    
    def print_summary(self, stats: ProcessingStats, error_files: List[Tuple[str, str]], verbose: bool):
        """Print processing summary."""
        print("\n" + "="*50)
        print("PREPROCESSING COMPLETE")
        print("="*50)
        
        print(f"Total images:     {stats.total_images:,}")
        print(f"Successfully processed: {stats.processed:,}")
        print(f"Already existed:  {stats.skipped:,}")
        print(f"Errors:           {stats.errors:,}")
        print(f"Success rate:     {stats.success_rate:.1f}%")
        print(f"Processing time:  {stats.elapsed_time:.1f}s")
        
        if stats.total_images > 0:
            rate = stats.total_images / stats.elapsed_time
            print(f"Processing rate:  {rate:.1f} images/sec")
        
        # Show storage information
        if stats.processed > 0 or stats.skipped > 0:
            total_processed = stats.processed + stats.skipped
            estimated_size_gb = total_processed * 0.5 / 1000  # Rough estimate: 0.5MB per image
            print(f"Estimated storage: ~{estimated_size_gb:.1f}GB for {total_processed:,} images")
        
        if error_files and verbose:
            print(f"\nErrors encountered:")
            for filename, error_msg in error_files[:10]:  # Show first 10 errors
                print(f"  {filename}: {error_msg}")
            if len(error_files) > 10:
                print(f"  ... and {len(error_files) - 10} more errors")


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(
        description="Preprocess SAM images with resizing and padding",
        epilog="This script creates padded square images to avoid on-the-fly computation during training."
    )
    
    parser.add_argument("--src-dir", type=Path, default="./datasets/sam/images",
                       help="Source directory containing images (default: ./datasets/sam/images)")
    parser.add_argument("--dst-dir", type=Path, default="./datasets/sam/processed_images",
                       help="Destination directory for processed images (default: ./datasets/sam/processed_images)")
    parser.add_argument("--size", type=int, default=1024,
                       help="Target size for square images (default: 1024)")
    parser.add_argument("--workers", type=int, 
                       help="Number of worker processes (default: auto-detect)")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce output verbosity")
    
    args = parser.parse_args()
    
    try:
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(
            src_dir=args.src_dir,
            dst_dir=args.dst_dir,
            target_size=args.size
        )
        
        # Process images
        stats = preprocessor.process_images(
            max_workers=args.workers,
            verbose=not args.quiet
        )
        
        # Exit with error code if too many failures
        if stats.errors > stats.processed:
            print("\nWarning: More errors than successful processing. Check your data.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

