"""
Box Label OCR Comparison - FiftyOne App

A FiftyOne application for comparing OCR engines (Tesseract, EasyOCR)
on box label images with ground truth validation.

Quick start:
    git clone <repo-url>
    cd ocr-comparison
    pip install -r requirements.txt
    python main.py

Open: http://localhost:5151
"""

import os
import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []

    try:
        import fiftyone
    except ImportError:
        missing.append("fiftyone")

    try:
        import pytesseract
    except ImportError:
        missing.append("pytesseract")

    try:
        import easyocr
    except ImportError:
        missing.append("easyocr")

    if missing:
        print("Missing dependencies. Please install:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)

    # Check Tesseract binary
    import shutil
    if not shutil.which("tesseract"):
        print("Warning: Tesseract binary not found.")
        print("  Install with: brew install tesseract (macOS)")
        print("  Or: apt-get install tesseract-ocr (Ubuntu)")


def setup_dataset():
    """Set up the FiftyOne dataset with images and run OCR."""
    import fiftyone as fo
    from ocr_engine import run_ocr_on_dataset

    dataset_name = "box_label_ocr"
    images_dir = PROJECT_ROOT / "data" / "images"
    labels_dir = PROJECT_ROOT / "labels" / "csv"

    # Check if dataset already exists with OCR results
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        # Check if OCR has been run
        if dataset.first() and dataset.first().get_field("easyocr_text"):
            print(f"Dataset '{dataset_name}' already exists with {len(dataset)} samples")
            return dataset
        else:
            print("Dataset exists but OCR not run. Running OCR...")
            run_ocr_on_dataset(dataset, labels_dir)
            return dataset

    print("Creating new dataset...")

    # Verify data exists
    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        print("Please add your images to data/images/")
        sys.exit(1)

    # Create dataset
    dataset = fo.Dataset(dataset_name)
    dataset.persistent = True

    # Add all images
    image_files = sorted(images_dir.glob("*.jpg"))
    if not image_files:
        print(f"Error: No .jpg files found in {images_dir}")
        sys.exit(1)

    print(f"Found {len(image_files)} images")

    for img_path in image_files:
        sample = fo.Sample(filepath=str(img_path.absolute()))
        sample["image_id"] = img_path.stem
        dataset.add_sample(sample)

    dataset.compute_metadata()

    # Run OCR on all images
    print("Running OCR on all images...")
    run_ocr_on_dataset(dataset, labels_dir)

    return dataset


def launch_app(dataset, port=5151):
    """Launch the FiftyOne app."""
    import fiftyone as fo

    print(f"\nLaunching FiftyOne app...")
    print(f"Open: http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    session = fo.launch_app(dataset, port=port, address="0.0.0.0")
    session.wait()


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Box Label OCR Comparison")
    parser.add_argument("--port", type=int, default=5151, help="Port for FiftyOne app")
    parser.add_argument("--rerun-ocr", action="store_true", help="Re-run OCR even if results exist")
    args = parser.parse_args()

    print("=" * 50)
    print("Box Label OCR Comparison")
    print("=" * 50)

    # Check dependencies
    check_dependencies()

    # Setup dataset
    import fiftyone as fo

    if args.rerun_ocr and fo.dataset_exists("box_label_ocr"):
        fo.delete_dataset("box_label_ocr")

    dataset = setup_dataset()

    # Print summary
    print(f"\nDataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")

    # Show OCR comparison summary
    tess_sims = [s.tesseract_similarity for s in dataset if s.tesseract_similarity]
    easy_sims = [s.easyocr_similarity for s in dataset if s.easyocr_similarity]

    if tess_sims:
        print(f"\nTesseract avg similarity: {sum(tess_sims)/len(tess_sims):.1f}%")
    if easy_sims:
        print(f"EasyOCR avg similarity: {sum(easy_sims)/len(easy_sims):.1f}%")

    # Launch app
    launch_app(dataset, port=args.port)


if __name__ == "__main__":
    main()
