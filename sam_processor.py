"""
SAM Dataset Processor
A standalone module for downloading, extracting, and validating SAM dataset files.
Can be used independently or as part of a larger pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tarfile
import textwrap
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from PIL import Image


@dataclass
class ImageStats:
    """Statistics for a processed image."""
    filename: str
    width: int
    height: int
    is_corrupted: bool
    error_msg: Optional[str] = None

    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio."""
        return self.width / self.height if self.height > 0 else 0.0


@dataclass
class DatasetSummary:
    """Summary statistics for the entire dataset."""
    total_images: int
    corrupted_images: int
    duplicate_filenames: List[str]
    image_stats: List[ImageStats]


def check_image_worker(image_path: Path) -> ImageStats:
    """Check a single image for corruption and get its stats. Standalone function for multiprocessing."""
    try:
        with Image.open(image_path) as img:
            img.load()  # Force load to check for corruption
            width, height = img.size
            
        return ImageStats(
            filename=image_path.name,
            width=width,
            height=height,
            is_corrupted=False
        )
    except (Image.UnidentifiedImageError, OSError) as e:
        return ImageStats(
            filename=image_path.name,
            width=0,
            height=0,
            is_corrupted=True,
            error_msg=str(e)
        )


class SAMProcessor:
    """Handles downloading, extracting, and validating SAM dataset files."""
    
    def __init__(self, data_dir: Path = None):
        """Initialize processor with data directory."""
        self.data_dir = data_dir or Path("./datasets/sam")
        self.json_path = self.data_dir / "download_status.json"
        self.tars_dir = self.data_dir / "tars"
        self.images_dir = self.data_dir / "images"
        self.json_dir = self.data_dir / "json"
        
        # Configuration
        self.chunk_size = 1 << 18  # 256 KiB
        self.timeout = 30  # seconds
        
        # Create directories
        for dir_path in [self.data_dir, self.tars_dir, self.images_dir, self.json_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def parse_file_list(self, content: str) -> List[Dict[str, str]]:
        """Parse TSV-like file list content into structured data."""
        entries = []
        for raw_line in textwrap.dedent(content).splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = re.split(r"\s+", line, maxsplit=1)
            if parts[0].lower() == "file_name":
                continue
            if len(parts) != 2:
                print(f"[WARN] Skipping malformed line: {raw_line!r}")
                continue
                
            filename, link = (p.strip() for p in parts)
            entries.append({"file_name": filename, "cdn_link": link})
        
        return entries

    def load_download_status(self, entries: List[Dict[str, str]]) -> Dict[str, Dict]:
        """Load or initialize download status tracking."""
        if self.json_path.exists():
            with self.json_path.open() as fp:
                status = json.load(fp)
            # Add any new entries
            for entry in entries:
                status.setdefault(
                    entry["file_name"],
                    {"cdn_link": entry["cdn_link"], "downloaded": False}
                )
            # Add validated field if missing
            if "validated" not in status:
                status["validated"] = False
        else:
            status = {
                entry["file_name"]: {**entry, "downloaded": False}
                for entry in entries
            }
            status["validated"] = False
        return status

    def save_download_status(self, status: Dict[str, Dict]) -> None:
        """Save download status to JSON file."""
        with self.json_path.open("w") as fp:
            json.dump(status, fp, indent=2)

    def get_pending_downloads(self, status: Dict[str, Dict], percentage: float = 1.0) -> List[str]:
        """Get list of files that haven't been downloaded yet, limited by percentage."""
        not_downloaded = [
            name for name, info in status.items() 
            if name != "validated" and isinstance(info, dict) and not info["downloaded"]
        ]
        total_files = len(not_downloaded)
        files_to_download = int(total_files * percentage)
        return not_downloaded[:files_to_download]

    def get_download_progress(self, status: Dict[str, Dict]) -> Tuple[int, int]:
        """Get current download progress as (downloaded, total) tuple."""
        file_items = {k: v for k, v in status.items() if k != "validated" and isinstance(v, dict)}
        downloaded = sum(1 for info in file_items.values() if info["downloaded"])
        total = len(file_items)
        return downloaded, total

    def download_file(self, filename: str, url: str) -> bool:
        """Download a single file from URL."""
        dest_path = self.tars_dir / filename
        try:
            response = requests.get(url, stream=True, timeout=self.timeout)
            response.raise_for_status()
            
            with dest_path.open("wb") as file_handle:
                for chunk in response.iter_content(self.chunk_size):
                    file_handle.write(chunk)
            
            return dest_path.stat().st_size > 0
        except Exception as exc:
            print(f"      Failed to download {filename}: {exc}")
            if dest_path.exists():
                dest_path.unlink(missing_ok=True)
            return False

    def extract_tar_file(self, tar_path: Path) -> bool:
        """Extract images and JSON files from tar archive."""
        try:
            with tarfile.open(tar_path) as tar:
                for member in tar.getmembers():
                    name = member.name
                    # Skip unsafe paths
                    if name.startswith("/") or ".." in name:
                        continue
                    
                    lower_name = name.lower()
                    if lower_name.endswith((".jpg", ".jpeg", ".png")):
                        tar.extract(member, self.images_dir)
                    elif lower_name.endswith(".json"):
                        tar.extract(member, self.json_dir)
            return True
        except Exception as exc:
            print(f"      Failed to extract {tar_path.name}: {exc}")
            return False

    def download_and_extract_batch(self, file_list_content: str, percentage: float = 1.0) -> bool:
        """Download and extract a batch of files based on percentage."""
        entries = self.parse_file_list(file_list_content)
        if not entries:
            print("No valid entries found in file list.")
            return False

        status = self.load_download_status(entries)
        batch = self.get_pending_downloads(status, percentage)
        
        if not batch:
            print("Nothing to download—all archive files are already present.")
            return True

        print(f"Downloading {len(batch)} archive file(s) ({percentage*100:.1f}% of remaining files)...")
        print("Each archive contains ~11,000 images (~11GB each) and will be extracted automatically.")
        print("This will give you ~", f"{len(batch) * 11_000:,}", "images total", f"({len(batch) * 11}GB storage).")
        
        try:
            for idx, filename in enumerate(batch, 1):
                tar_path = self.tars_dir / filename
                print(f"[{idx:3}/{len(batch)}] Processing archive {filename}")

                # Download file
                if not self.download_file(filename, status[filename]["cdn_link"]):
                    print(f"      Skipping extraction for {filename}")
                    continue

                status[filename]["downloaded"] = True
                print(f"      Downloaded archive {filename}")

                # Extract file
                if self.extract_tar_file(tar_path):
                    print(f"      Extracted images from {filename}")
                    tar_path.unlink(missing_ok=True)  # Clean up tar file
                    print(f"      Cleaned up archive {filename}")
                else:
                    print(f"      Extraction failed for {filename}")

                self.save_download_status(status)

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] Download stopped by user")
            return False
        finally:
            self.save_download_status(status)

        # Run validation after successful download
        print("\nValidating downloaded images...")
        summary = self.validate_images(verbose=False)
        self.print_summary(summary)
        
        # Mark as validated
        status["validated"] = True
        self.save_download_status(status)
        
        print("Validation complete!")
        return True

    def validate_images(self, verbose: bool = False) -> DatasetSummary:
        """Validate all images in the dataset and return summary statistics."""
        if not self.images_dir.exists():
            print(f"Images directory {self.images_dir} does not exist.")
            return DatasetSummary(0, 0, [], [])

        image_files = [
            f for f in self.images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
        
        if verbose:
            print("Filename           Width Height   Aspect ratio")
            print("-" * 55)

        print(f"Validating {len(image_files)} images using multiprocessing...")
        
        # Use ProcessPoolExecutor for true parallelism on CPU-bound tasks
        max_workers = min(os.cpu_count() or 1, 8)  # Limit to 8 processes max
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Process all images in parallel
            image_stats = list(executor.map(check_image_worker, sorted(image_files)))
        
        filenames = [stats.filename for stats in image_stats]
        
        if verbose:
            for stats in image_stats:
                if stats.is_corrupted:
                    print(f"{stats.filename:18s} CORRUPTED ({stats.error_msg})")
                else:
                    ratio = stats.aspect_ratio
                    print(f"{stats.filename:18s} {stats.width:4d}x{stats.height:<4d}   "
                          f"{ratio:5.3f}")

        # Check for duplicates
        duplicates = [name for name, count in Counter(filenames).items() if count > 1]
        
        # Calculate summary stats
        total_images = len(image_stats)
        corrupted_images = sum(1 for s in image_stats if s.is_corrupted)

        summary = DatasetSummary(
            total_images=total_images,
            corrupted_images=corrupted_images,
            duplicate_filenames=duplicates,
            image_stats=image_stats
        )

        return summary

    def print_summary(self, summary: DatasetSummary) -> None:
        """Print a formatted summary of dataset validation results."""
        if summary.duplicate_filenames:
            print("\nWARNING: Found duplicate filenames:")
            for dup in summary.duplicate_filenames:
                print(f"  - {dup}")
        else:
            print("\nAll filenames are unique!")

        print(f"\nDataset Summary:")
        print(f"  Total images:    {summary.total_images:,} ({summary.total_images/1000:.1f}k)")
        print(f"  Corrupted:       {summary.corrupted_images:,}")
        
        if summary.corrupted_images > 0:
            print(f"\nCorrupted files:")
            for stats in summary.image_stats:
                if stats.is_corrupted:
                    print(f"  - {stats.filename}: {stats.error_msg}")

    def get_dataset_status(self) -> Dict:
        """Get current dataset status including download progress and image counts."""
        status = {}
        
        # Check if dataset exists
        if self.json_path.exists():
            with self.json_path.open() as fp:
                download_status = json.load(fp)
            downloaded, total = self.get_download_progress(download_status)
            status["download_progress"] = {"downloaded": downloaded, "total": total}
            status["validated"] = download_status.get("validated", False)
        else:
            status["download_progress"] = {"downloaded": 0, "total": 0}
            status["validated"] = False
        
        # Check images
        if self.images_dir.exists():
            image_files = [
                f for f in self.images_dir.iterdir()
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
            status["image_count"] = len(image_files)
        else:
            status["image_count"] = 0
            
        return status


def interactive_download(data_dir: Path = None) -> bool:
    """Interactive download with user prompts for percentage selection."""
    processor = SAMProcessor(data_dir)
    
    # Check if list.txt exists
    list_file = Path("list.txt")
    if not list_file.exists():
        print("Error: list.txt file not found. Please ensure it exists in the current directory.")
        return False
    
    # Load file list content
    file_list_content = list_file.read_text()
    entries = processor.parse_file_list(file_list_content)
    total_files = len(entries)
    
    print(f"SAM Dataset Processor")
    print(f"Found {total_files} archive files available for download")
    print(f"(Each archive contains multiple images - thousands of images total)")
    
    # Check current status
    status = processor.get_dataset_status()
    
    if status["download_progress"]["total"] > 0:
        downloaded = status["download_progress"]["downloaded"] 
        total = status["download_progress"]["total"]
        print(f"Current progress: {downloaded}/{total} archive files downloaded")
        print(f"Images extracted: {status['image_count']}")
        
        use_existing = input("\nUse current dataset or download more? (use/download): ").strip().lower()
        if use_existing.startswith('u'):
            print("Using existing dataset.")
            return True
    
    # Get percentage from user
    while True:
        try:
            percentage_input = input(f"\nWhat percentage of the {total_files} archive files do you want to download? (0-100) [1]: ")
            if not percentage_input.strip():  # Empty input defaults to 1%
                percentage = 1.0 / 100
                break
            percentage = float(percentage_input) / 100
            if 0 <= percentage <= 1:
                break
            else:
                print("Please enter a number between 0 and 100.")
        except ValueError:
            print("Please enter a valid number.")
    
    if percentage == 0:
        print("No files to download.")
        return True
    
    files_to_download = int(total_files * percentage)
    print(f"\nThis will download ~ {files_to_download} archive files ({percentage*100:.1f}% of total)")
    print(f"Each archive contains multiple images, so you'll get thousands of images total.")
    print("This download process will take a while depending on your internet connection.")
    
    confirm = input("Continue? (y/n) [y]: ").strip().lower()
    if not confirm:  # Empty input defaults to 'y'
        confirm = 'y'
    if not confirm.startswith('y'):
        print("Download cancelled.")
        return False
    
    return processor.download_and_extract_batch(file_list_content, percentage)


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description="SAM Dataset Processor")
    parser.add_argument("--data-dir", type=Path, help="Directory to store dataset")
    parser.add_argument("--list-file", type=Path, default="list.txt", 
                       help="File containing dataset URLs")
    parser.add_argument("--percentage", type=float, 
                       help="Percentage of dataset to download (0-100)")
    parser.add_argument("--validate", action="store_true", 
                       help="Validate images after download")
    parser.add_argument("--status", action="store_true",
                       help="Show current dataset status")
    parser.add_argument("--interactive", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    processor = SAMProcessor(args.data_dir)
    
    if args.status:
        status = processor.get_dataset_status()
        print("Dataset Status:")
        print(f"  Downloaded files: {status['download_progress']['downloaded']}/{status['download_progress']['total']}")
        print(f"  Extracted images: {status['image_count']/1000:.1f}k")
        print(f"  Validated: {'Yes' if status['validated'] else 'No'}")
        return
    
    if args.interactive:
        success = interactive_download(args.data_dir)
    else:
        if not args.list_file.exists():
            print(f"Error: List file {args.list_file} not found.")
            sys.exit(1)
        
        file_list_content = args.list_file.read_text()
        percentage = (args.percentage or 100) / 100
        
        success = processor.download_and_extract_batch(file_list_content, percentage)
    
    if success and args.validate:
        print("\nValidating downloaded images...")
        summary = processor.validate_images(verbose=True)
        processor.print_summary(summary)
    
    if success:
        print("\nProcessing completed successfully!")
    else:
        print("\nProcessing failed.")
        sys.exit(1)


if __name__ == "__main__":
    main() 