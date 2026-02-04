"""
OCR Engine Module

Provides OCR functionality using Tesseract and EasyOCR for the FiftyOne dataset.
"""

import csv
from pathlib import Path
from typing import Dict, Optional

# Lazy loading for OCR engines
_tesseract = None
_easyocr_reader = None


def get_tesseract():
    """Get pytesseract module (lazy load)."""
    global _tesseract
    if _tesseract is None:
        import pytesseract
        _tesseract = pytesseract
    return _tesseract


def get_easyocr():
    """Get EasyOCR reader (lazy load)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader


def run_tesseract(image_path: str) -> Dict:
    """Run Tesseract OCR on an image."""
    import time
    from PIL import Image

    pt = get_tesseract()
    start_time = time.time()

    img = Image.open(image_path)
    config = r'--oem 3 --psm 3'
    data = pt.image_to_data(img, output_type=pt.Output.DICT, config=config)

    texts = []
    confidences = []
    for i, conf in enumerate(data['conf']):
        if int(conf) > 0:
            texts.append(data['text'][i])
            confidences.append(int(conf))

    return {
        'text': ' '.join(texts),
        'confidence': sum(confidences) / len(confidences) if confidences else 0,
        'time': time.time() - start_time
    }


def run_easyocr(image_path: str) -> Dict:
    """Run EasyOCR on an image."""
    import time

    reader = get_easyocr()
    start_time = time.time()

    results = reader.readtext(image_path)

    texts = []
    confidences = []
    for (bbox, text, conf) in results:
        texts.append(text)
        confidences.append(conf)

    return {
        'text': ' '.join(texts),
        'confidence': sum(confidences) / len(confidences) * 100 if confidences else 0,
        'time': time.time() - start_time
    }


def load_ground_truth(csv_path: str) -> Optional[Dict[str, str]]:
    """Load ground truth from a CSV file."""
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                return dict(row)
    except FileNotFoundError:
        return None
    return None


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate text similarity using fuzzy matching."""
    from rapidfuzz import fuzz
    if not text1 or not text2:
        return 0.0
    return fuzz.ratio(text1.lower().strip(), text2.lower().strip())


def calculate_metrics(ocr_text: str, ground_truth: Dict[str, str]) -> Dict[str, float]:
    """Calculate metrics comparing OCR output to ground truth."""
    # Combine all ground truth text
    gt_combined = ' '.join(str(v) for v in ground_truth.values() if v and v != 'NONE')

    metrics = {
        'overall_similarity': calculate_similarity(ocr_text, gt_combined)
    }

    # Check key fields
    key_fields = ['Barcode', 'Box Number (?)', 'Pack Date', 'Kill Date', 'Net Weight (kg)', 'Net Weight (lb)']
    found = 0
    total = 0

    for field in key_fields:
        if field in ground_truth and ground_truth[field] not in ('', 'NONE'):
            total += 1
            if ground_truth[field].lower() in ocr_text.lower():
                found += 1

    metrics['field_accuracy'] = (found / total * 100) if total > 0 else 0

    return metrics


def run_ocr_on_dataset(dataset, labels_dir: Path):
    """Run OCR on all samples in a FiftyOne dataset."""
    from tqdm import tqdm

    labels_dir = Path(labels_dir)

    print(f"Running OCR on {len(dataset)} samples...")

    for sample in tqdm(dataset, desc="Processing"):
        image_id = sample.image_id or Path(sample.filepath).stem
        csv_path = labels_dir / f"{image_id}.csv"

        # Load ground truth
        gt = load_ground_truth(str(csv_path))

        # Run Tesseract
        try:
            tess = run_tesseract(sample.filepath)
            sample['tesseract_text'] = tess['text']
            sample['tesseract_confidence'] = tess['confidence']
            sample['tesseract_time'] = tess['time']

            if gt:
                metrics = calculate_metrics(tess['text'], gt)
                sample['tesseract_similarity'] = metrics['overall_similarity']
                sample['tesseract_field_accuracy'] = metrics['field_accuracy']
        except Exception as e:
            print(f"Tesseract error on {image_id}: {e}")

        # Run EasyOCR
        try:
            easy = run_easyocr(sample.filepath)
            sample['easyocr_text'] = easy['text']
            sample['easyocr_confidence'] = easy['confidence']
            sample['easyocr_time'] = easy['time']

            if gt:
                metrics = calculate_metrics(easy['text'], gt)
                sample['easyocr_similarity'] = metrics['overall_similarity']
                sample['easyocr_field_accuracy'] = metrics['field_accuracy']
        except Exception as e:
            print(f"EasyOCR error on {image_id}: {e}")

        # Add ground truth fields
        if gt:
            sample['gt_box_number'] = gt.get('Box Number (?)', '')
            sample['gt_pack_date'] = gt.get('Pack Date', '')
            sample['gt_kill_date'] = gt.get('Kill Date', '')
            sample['gt_weight_kg'] = gt.get('Net Weight (kg)', '')
            sample['gt_weight_lb'] = gt.get('Net Weight (lb)', '')
            sample['gt_sku'] = gt.get('SKU Name (?)', '')

        # Determine best engine
        tess_sim = sample.get_field('tesseract_similarity') or 0
        easy_sim = sample.get_field('easyocr_similarity') or 0
        sample['best_engine'] = 'tesseract' if tess_sim > easy_sim else 'easyocr'
        sample['best_similarity'] = max(tess_sim, easy_sim)

        sample.save()

    print("OCR complete!")
