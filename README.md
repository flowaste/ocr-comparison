# Box Label OCR Comparison

A FiftyOne application for comparing OCR engines (Tesseract, EasyOCR) on box label images with ground truth validation.

## Features

- Compare **Tesseract** and **EasyOCR** side-by-side
- Automatic similarity scoring against ground truth
- Interactive FiftyOne visualization
- Filter and sort by OCR accuracy
- Field-level accuracy metrics

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd ocr-comparison

# Create virtual environment (Python 3.12 recommended)
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract binary (if not already installed)
# macOS:
brew install tesseract

# Ubuntu/Debian:
# sudo apt-get install tesseract-ocr

# Run the app
python main.py
```

Open: http://localhost:5151

## Project Structure

```
ocr-comparison/
├── main.py              # Main entry point
├── ocr_engine.py        # OCR processing module
├── requirements.txt     # Python dependencies
├── data/
│   └── images/          # Box label images (.jpg)
└── labels/
    └── csv/             # Ground truth CSV files
```

## Data Format

### Images
Place your box label images in `data/images/` as `.jpg` files.

### Ground Truth CSV
Each image should have a corresponding CSV file in `labels/csv/` with the same name (e.g., `4.jpg` → `4.csv`).

CSV columns:
- `Box Label` - Image filename
- `Barcode` - Barcode text
- `Box Number (?)` - Box identifier
- `Pack Date` - Pack date
- `Kill Date` - Kill/slaughter date
- `Net Weight (kg)` - Weight in kg
- `Net Weight (lb)` - Weight in lb
- And more...

## Usage

### Basic Usage
```bash
python main.py
```

### Options
```bash
python main.py --port 5152          # Use different port
python main.py --rerun-ocr          # Re-run OCR (clear cached results)
```

## FiftyOne Fields

After running, each sample will have these fields:

| Field | Description |
|-------|-------------|
| `tesseract_text` | Raw Tesseract OCR output |
| `tesseract_similarity` | Similarity to ground truth (%) |
| `tesseract_confidence` | Tesseract confidence score |
| `easyocr_text` | Raw EasyOCR output |
| `easyocr_similarity` | Similarity to ground truth (%) |
| `easyocr_confidence` | EasyOCR confidence score |
| `best_engine` | Which engine performed better |
| `best_similarity` | Best similarity score |
| `gt_*` | Ground truth fields |

## Filtering in FiftyOne

Use the sidebar to filter samples:
- Sort by `easyocr_similarity` to find best/worst results
- Filter by `best_engine` to see which engine wins more often
- Compare `tesseract_text` vs `easyocr_text` on individual samples

## Requirements

- Python 3.10-3.12 (3.12 recommended)
- Tesseract OCR binary
- ~2GB disk space for dependencies
