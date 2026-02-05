"""
OCR Engine Module

Modular OCR engine supporting multiple models:
- Tesseract (local)
- EasyOCR (local)
- TrOCR (local, transformer-based)
- Google Gemini (cloud API)
- OpenAI GPT-4o (cloud API)
- Claude (cloud API)

Add new models by implementing the OCREngine interface.
"""

import csv
import time
import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class OCRResult:
    """Result from an OCR engine."""
    text: str
    confidence: float
    processing_time: float
    engine_name: str
    raw_output: Optional[Dict] = None


class OCREngine(ABC):
    """Base class for OCR engines."""

    name: str = "base"
    requires_api_key: bool = False

    @abstractmethod
    def run(self, image_path: str) -> OCRResult:
        """Run OCR on an image and return results."""
        pass

    def is_available(self) -> bool:
        """Check if this engine is available (dependencies installed, API key set, etc.)."""
        return True


# =============================================================================
# LOCAL OCR ENGINES
# =============================================================================

class TesseractEngine(OCREngine):
    """Tesseract OCR engine."""

    name = "tesseract"
    requires_api_key = False
    _instance = None

    def __init__(self):
        import pytesseract
        self.pytesseract = pytesseract

    def run(self, image_path: str) -> OCRResult:
        from PIL import Image

        start_time = time.time()
        img = Image.open(image_path)
        config = r'--oem 3 --psm 3'
        data = self.pytesseract.image_to_data(img, output_type=self.pytesseract.Output.DICT, config=config)

        texts = []
        confidences = []
        for i, conf in enumerate(data['conf']):
            if int(conf) > 0:
                texts.append(data['text'][i])
                confidences.append(int(conf))

        return OCRResult(
            text=' '.join(texts),
            confidence=sum(confidences) / len(confidences) if confidences else 0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )


class EasyOCREngine(OCREngine):
    """EasyOCR engine."""

    name = "easyocr"
    requires_api_key = False
    _reader = None

    def __init__(self, languages: List[str] = None, gpu: bool = False):
        import easyocr
        self.languages = languages or ['en']
        self.gpu = gpu
        if EasyOCREngine._reader is None:
            EasyOCREngine._reader = easyocr.Reader(self.languages, gpu=self.gpu)
        self.reader = EasyOCREngine._reader

    def run(self, image_path: str) -> OCRResult:
        start_time = time.time()
        results = self.reader.readtext(image_path)

        texts = []
        confidences = []
        for (bbox, text, conf) in results:
            texts.append(text)
            confidences.append(conf)

        return OCRResult(
            text=' '.join(texts),
            confidence=sum(confidences) / len(confidences) * 100 if confidences else 0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )


class TrOCREngine(OCREngine):
    """TrOCR transformer-based OCR engine."""

    name = "trocr"
    requires_api_key = False
    _model = None
    _processor = None

    def __init__(self, model_name: str = "microsoft/trocr-base-printed"):
        self.model_name = model_name

    def _load_model(self):
        if TrOCREngine._model is None:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            TrOCREngine._processor = TrOCRProcessor.from_pretrained(self.model_name)
            TrOCREngine._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
        return TrOCREngine._processor, TrOCREngine._model

    def run(self, image_path: str) -> OCRResult:
        from PIL import Image

        start_time = time.time()
        processor, model = self._load_model()

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return OCRResult(
            text=text,
            confidence=0,  # TrOCR doesn't provide confidence
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False


# =============================================================================
# CLOUD API OCR ENGINES
# =============================================================================

class GeminiOCREngine(OCREngine):
    """Google Gemini OCR engine."""

    name = "gemini"
    requires_api_key = True

    def __init__(self, api_key: str = None, model: str = "gemini-2.0-flash"):
        import os
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.model = model

    def run(self, image_path: str) -> OCRResult:
        import google.generativeai as genai
        from PIL import Image

        start_time = time.time()
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)

        image = Image.open(image_path)
        prompt = "Extract ALL text from this image exactly as written. Return only the extracted text, nothing else."

        response = model.generate_content([prompt, image])
        text = response.text if response.text else ""

        return OCRResult(
            text=text,
            confidence=0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        return bool(self.api_key)


class RoboflowDocTREngine(OCREngine):
    """Roboflow DocTR OCR API engine."""

    name = "doctr"
    requires_api_key = True

    def __init__(self, api_key: str = None, api_url: str = "https://infer.roboflow.com"):
        import os
        self.api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
        self.api_url = api_url

    def run(self, image_path: str) -> OCRResult:
        from inference_sdk import InferenceHTTPClient

        start_time = time.time()

        client = InferenceHTTPClient(
            api_url=self.api_url,
            api_key=self.api_key
        )

        result = client.ocr_image(inference_input=image_path)

        # Extract text from result
        text = result.get('result', '') if isinstance(result, dict) else str(result)

        return OCRResult(
            text=text,
            confidence=0,
            processing_time=time.time() - start_time,
            engine_name=self.name,
            raw_output=result if isinstance(result, dict) else None
        )

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        try:
            from inference_sdk import InferenceHTTPClient
            return True
        except ImportError:
            return False


class DonutOCREngine(OCREngine):
    """Donut Document Understanding Transformer OCR engine."""

    name = "donut"
    requires_api_key = False
    _model = None
    _processor = None

    def __init__(self, model_name: str = "naver-clova-ix/donut-base-finetuned-cord-v2"):
        self.model_name = model_name

    def _load_model(self):
        if DonutOCREngine._model is None:
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            import torch

            DonutOCREngine._processor = DonutProcessor.from_pretrained(self.model_name)
            DonutOCREngine._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)

            # Move to GPU if available
            if torch.cuda.is_available():
                DonutOCREngine._model = DonutOCREngine._model.cuda()

        return DonutOCREngine._processor, DonutOCREngine._model

    def run(self, image_path: str) -> OCRResult:
        from PIL import Image
        import torch

        start_time = time.time()
        processor, model = self._load_model()

        image = Image.open(image_path).convert("RGB")

        # Prepare input
        pixel_values = processor(image, return_tensors="pt").pixel_values

        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()

        # Generate output
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids

        if torch.cuda.is_available():
            decoder_input_ids = decoder_input_ids.cuda()

        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )

        # Decode output
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        text = processor.token2json(sequence)

        # Convert dict to string if needed
        if isinstance(text, dict):
            text = str(text)

        return OCRResult(
            text=text,
            confidence=0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        try:
            from transformers import DonutProcessor
            return True
        except ImportError:
            return False


class PaddleOCREngine(OCREngine):
    """PaddleOCR engine - high-accuracy multilingual OCR."""

    name = "paddleocr"
    requires_api_key = False
    _ocr = None

    def __init__(self, lang: str = 'en', use_gpu: bool = False):
        self.lang = lang
        self.use_gpu = use_gpu

    def _load_model(self):
        if PaddleOCREngine._ocr is None:
            from paddleocr import PaddleOCR
            PaddleOCREngine._ocr = PaddleOCR(
                use_angle_cls=True,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False
            )
        return PaddleOCREngine._ocr

    def run(self, image_path: str) -> OCRResult:
        start_time = time.time()
        ocr = self._load_model()

        result = ocr.ocr(image_path, cls=True)

        # Extract text from results
        text_parts = []
        confidences = []
        if result and result[0]:
            for line in result[0]:
                text_parts.append(line[1][0])  # text
                confidences.append(line[1][1])  # confidence

        avg_conf = sum(confidences) / len(confidences) * 100 if confidences else 0

        return OCRResult(
            text=' '.join(text_parts),
            confidence=avg_conf,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False


class KerasOCREngine(OCREngine):
    """Keras-OCR engine - lightweight CNN-based OCR."""

    name = "keras_ocr"
    requires_api_key = False
    _pipeline = None

    def _load_pipeline(self):
        if KerasOCREngine._pipeline is None:
            import keras_ocr
            KerasOCREngine._pipeline = keras_ocr.pipeline.Pipeline()
        return KerasOCREngine._pipeline

    def run(self, image_path: str) -> OCRResult:
        import keras_ocr

        start_time = time.time()
        pipeline = self._load_pipeline()

        # Read image
        image = keras_ocr.tools.read(image_path)
        results = pipeline.recognize([image])

        # Extract text from results
        text_parts = []
        if results and len(results) > 0:
            for text, box in results[0]:
                text_parts.append(text)

        return OCRResult(
            text=' '.join(text_parts),
            confidence=0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        try:
            import keras_ocr
            return True
        except ImportError:
            return False


class DocTRLocalEngine(OCREngine):
    """DocTR local engine - runs locally without API."""

    name = "doctr_local"
    requires_api_key = False
    _model = None

    def _load_model(self):
        if DocTRLocalEngine._model is None:
            from doctr.models import ocr_predictor
            DocTRLocalEngine._model = ocr_predictor(pretrained=True)
        return DocTRLocalEngine._model

    def run(self, image_path: str) -> OCRResult:
        from doctr.io import DocumentFile

        start_time = time.time()
        model = self._load_model()

        # Load document
        doc = DocumentFile.from_images(image_path)
        result = model(doc)

        # Extract text
        text_parts = []
        for page in result.pages:
            for block in page.blocks:
                for line in block.lines:
                    line_text = ' '.join([word.value for word in line.words])
                    text_parts.append(line_text)

        return OCRResult(
            text=' '.join(text_parts),
            confidence=0,
            processing_time=time.time() - start_time,
            engine_name=self.name
        )

    def is_available(self) -> bool:
        try:
            from doctr.models import ocr_predictor
            return True
        except ImportError:
            return False


# =============================================================================
# ENGINE REGISTRY
# =============================================================================

# Available engines (add new engines here)
AVAILABLE_ENGINES = {
    'tesseract': TesseractEngine,
    'easyocr': EasyOCREngine,
    'trocr': TrOCREngine,
    'donut': DonutOCREngine,
    'paddleocr': PaddleOCREngine,
    'keras_ocr': KerasOCREngine,
    'doctr_local': DocTRLocalEngine,
    'doctr': RoboflowDocTREngine,
    'gemini': GeminiOCREngine,
}


def get_engine(name: str, **kwargs) -> OCREngine:
    """Get an OCR engine by name."""
    if name not in AVAILABLE_ENGINES:
        raise ValueError(f"Unknown engine: {name}. Available: {list(AVAILABLE_ENGINES.keys())}")
    return AVAILABLE_ENGINES[name](**kwargs)


def list_engines() -> List[str]:
    """List all available OCR engines."""
    return list(AVAILABLE_ENGINES.keys())


def get_available_engines() -> List[str]:
    """List engines that are currently usable (dependencies installed, API keys set)."""
    available = []
    for name, engine_class in AVAILABLE_ENGINES.items():
        try:
            engine = engine_class()
            if engine.is_available():
                available.append(name)
        except Exception:
            pass
    return available


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
    gt_combined = ' '.join(str(v) for v in ground_truth.values() if v and v != 'NONE')

    metrics = {
        'overall_similarity': calculate_similarity(ocr_text, gt_combined)
    }

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


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def run_ocr_on_dataset(dataset, labels_dir: Path, engines: List[str] = None):
    """Run OCR on all samples in a FiftyOne dataset.

    Args:
        dataset: FiftyOne dataset
        labels_dir: Path to CSV labels directory
        engines: List of engine names to use (default: ['tesseract', 'easyocr'])
    """
    from tqdm import tqdm

    labels_dir = Path(labels_dir)

    if engines is None:
        engines = ['tesseract', 'easyocr']

    # Initialize engines
    engine_instances = {}
    for name in engines:
        try:
            engine = get_engine(name)
            if engine.is_available():
                engine_instances[name] = engine
                print(f"  Loaded engine: {name}")
            else:
                print(f"  Skipping {name}: not available (missing API key or dependencies)")
        except Exception as e:
            print(f"  Skipping {name}: {e}")

    if not engine_instances:
        print("No OCR engines available!")
        return

    print(f"\nRunning OCR on {len(dataset)} samples with engines: {list(engine_instances.keys())}")

    for sample in tqdm(dataset, desc="Processing"):
        image_id = sample.image_id or Path(sample.filepath).stem
        csv_path = labels_dir / f"{image_id}.csv"
        gt = load_ground_truth(str(csv_path))

        similarities = {}

        # Run each engine
        for name, engine in engine_instances.items():
            try:
                result = engine.run(sample.filepath)
                sample[f'{name}_text'] = result.text
                sample[f'{name}_confidence'] = result.confidence
                sample[f'{name}_time'] = result.processing_time

                if gt:
                    metrics = calculate_metrics(result.text, gt)
                    sample[f'{name}_similarity'] = metrics['overall_similarity']
                    sample[f'{name}_field_accuracy'] = metrics['field_accuracy']
                    similarities[name] = metrics['overall_similarity']
            except Exception as e:
                print(f"\n{name} error on {image_id}: {e}")

        # Add ground truth fields
        if gt:
            sample['gt_box_number'] = gt.get('Box Number (?)', '')
            sample['gt_pack_date'] = gt.get('Pack Date', '')
            sample['gt_kill_date'] = gt.get('Kill Date', '')
            sample['gt_weight_kg'] = gt.get('Net Weight (kg)', '')
            sample['gt_weight_lb'] = gt.get('Net Weight (lb)', '')
            sample['gt_sku'] = gt.get('SKU Name (?)', '')

        # Determine best engine
        if similarities:
            best = max(similarities, key=similarities.get)
            sample['best_engine'] = best
            sample['best_similarity'] = similarities[best]

        sample.save()

    print("OCR complete!")
