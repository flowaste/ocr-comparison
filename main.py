"""
Box Label OCR Comparison - FiftyOne App

A FiftyOne application for comparing OCR engines on box label images.

Supported engines:
- tesseract (local, open source)
- easyocr (local, neural network)
- trocr (local, transformer-based)
- donut (local, document understanding transformer)
- gemini (cloud, requires GOOGLE_API_KEY)
- doctr (cloud, Roboflow DocTR, requires ROBOFLOW_API_KEY)

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


def setup_dataset(engines=None, force_rerun=False):
    """Set up the FiftyOne dataset with images and run OCR."""
    import fiftyone as fo
    from ocr_engine import run_ocr_on_dataset

    if engines is None:
        engines = ['tesseract', 'easyocr']

    dataset_name = "box_label_ocr"
    images_dir = PROJECT_ROOT / "data" / "images"
    labels_dir = PROJECT_ROOT / "labels" / "csv"

    # Check if dataset already exists
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)

        if force_rerun:
            print("Re-running OCR on existing dataset...")
            run_ocr_on_dataset(dataset, labels_dir, engines=engines)
            return dataset

        # Check if OCR has been run for the requested engines
        first_sample = dataset.first()
        missing_engines = [e for e in engines if not first_sample.get_field(f"{e}_text")]

        if missing_engines:
            print(f"Running OCR for new engines: {missing_engines}")
            run_ocr_on_dataset(dataset, labels_dir, engines=missing_engines)
        else:
            print(f"Dataset '{dataset_name}' already has results for {engines}")

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
    run_ocr_on_dataset(dataset, labels_dir, engines=engines)

    return dataset


def launch_app(dataset, port=5151):
    """Launch the FiftyOne app."""
    import fiftyone as fo

    print(f"\nLaunching FiftyOne app...")
    print(f"Open: http://localhost:{port}")
    print("Press Ctrl+C to stop\n")

    session = fo.launch_app(dataset, port=port, address="0.0.0.0")
    session.wait()


def print_summary(dataset, engines):
    """Print summary statistics for each engine."""
    print(f"\nDataset: {dataset.name}")
    print(f"Samples: {len(dataset)}")
    print("\nOCR Results Summary:")
    print("-" * 40)

    for engine in engines:
        sim_field = f"{engine}_similarity"
        sims = [getattr(s, sim_field) for s in dataset if getattr(s, sim_field, None)]
        if sims:
            avg = sum(sims) / len(sims)
            print(f"  {engine:12} avg similarity: {avg:.1f}%")


def main():
    """Main entry point."""
    import argparse
    from ocr_engine import list_engines, get_available_engines
    from fiftyone_ocr_models import list_fiftyone_models, FIFTYONE_OCR_MODELS

    all_engines = list_engines()
    all_fo_models = list_fiftyone_models()

    parser = argparse.ArgumentParser(
        description="Box Label OCR Comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available OCR engines (direct):
  Local (free):
    - tesseract    Traditional OCR engine
    - easyocr      Neural network OCR
    - trocr        Transformer-based OCR (requires transformers)
    - donut        Document Understanding Transformer (requires transformers)

  Cloud APIs (requires API keys):
    - gemini       Google Gemini (set GOOGLE_API_KEY)
    - doctr        Roboflow DocTR OCR (set ROBOFLOW_API_KEY)

FiftyOne Plugin Models (--fo-models):
    - kosmos2_5    Microsoft Kosmos-2.5 (document OCR)
    - olmocr2      Allen AI olmOCR-2 (markdown output)
    - florence2    Microsoft Florence-2 (vision foundation)
    - qwen2_vl     Qwen2.5-VL (multimodal VLM)
    - minicpm_v    MiniCPM-V 4.5 (efficient VLM)
    - moondream    Moondream 3 (lightweight VLM)

Examples:
  python main.py                                    # tesseract + easyocr
  python main.py -e tesseract easyocr gemini        # Add cloud API
  python main.py --fo-models kosmos2_5 florence2    # Use FiftyOne plugins
  python main.py --list-engines                     # Show all available
        """
    )
    parser.add_argument("--port", type=int, default=5151, help="Port for FiftyOne app")
    parser.add_argument("--rerun-ocr", action="store_true", help="Re-run OCR even if results exist")
    parser.add_argument("--engines", "-e", nargs="+", default=["tesseract", "easyocr"],
                       help=f"OCR engines to use (default: tesseract easyocr)")
    parser.add_argument("--fo-models", nargs="+", default=[],
                       help=f"FiftyOne plugin models to use: {all_fo_models}")
    parser.add_argument("--list-engines", action="store_true", help="List available OCR engines and exit")
    parser.add_argument("--no-app", action="store_true", help="Run OCR but don't launch FiftyOne app")
    args = parser.parse_args()

    if args.list_engines:
        print("\n=== Direct OCR Engines ===")
        print("All supported:", all_engines)
        print("Currently available:", get_available_engines())
        print("\nTo use cloud APIs, set environment variables:")
        print("  export GOOGLE_API_KEY=your-key      # for gemini")
        print("  export ROBOFLOW_API_KEY=your-key    # for doctr")
        print("\n=== FiftyOne Plugin Models ===")
        for name, info in FIFTYONE_OCR_MODELS.items():
            print(f"  {name:12} - {info['description']}")
        return

    print("=" * 50)
    print("Box Label OCR Comparison")
    print("=" * 50)
    print(f"Direct engines: {args.engines}")
    if args.fo_models:
        print(f"FiftyOne models: {args.fo_models}")

    # Check dependencies
    check_dependencies()

    # Setup dataset
    import fiftyone as fo

    if args.rerun_ocr and fo.dataset_exists("box_label_ocr"):
        fo.delete_dataset("box_label_ocr")

    dataset = setup_dataset(engines=args.engines, force_rerun=args.rerun_ocr)

    # Run FiftyOne plugin models if specified
    if args.fo_models:
        from fiftyone_ocr_models import run_fiftyone_ocr_comparison
        labels_dir = PROJECT_ROOT / "labels" / "csv"
        run_fiftyone_ocr_comparison(dataset, models=args.fo_models, labels_dir=labels_dir)

    # Print summary
    all_models = args.engines + args.fo_models
    print_summary(dataset, all_models)

    # Launch app
    if not args.no_app:
        launch_app(dataset, port=args.port)


if __name__ == "__main__":
    main()
