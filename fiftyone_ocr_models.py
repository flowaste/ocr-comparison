"""
FiftyOne OCR Plugin Models

This module provides integration with FiftyOne's model zoo OCR plugins.
These models are registered from GitHub and run through FiftyOne's apply_model().

NOTE: These plugins require FiftyOne Enterprise for full functionality.
Most models need GPU (NVIDIA CUDA) and significant disk space.

Available models:
- Document OCR models (Kosmos-2.5, olmOCR-2, DeepSeek-OCR, etc.)
- Document parsing (MinerU)
- Traditional OCR plugins (PyTesseract, Donut)
"""

import time
from typing import Dict, List, Optional
from pathlib import Path

# =============================================================================
# FIFTYONE OCR PLUGIN REGISTRY (OCR-SPECIFIC MODELS ONLY)
# =============================================================================

FIFTYONE_OCR_MODELS = {
    # -------------------------------------------------------------------------
    # Document OCR Models
    # -------------------------------------------------------------------------
    'kosmos2_5': {
        'source': 'https://github.com/harpreetsahota204/kosmos2_5',
        'model_name': 'microsoft/kosmos-2.5',
        'label_field': 'kosmos_text',
        'description': 'Microsoft Kosmos-2.5 - Document OCR with bounding boxes',
        'operation': 'ocr',
        'size': '~5GB',
        'requires_gpu': True,
    },
    'olmocr2': {
        'source': 'https://github.com/harpreetsahota204/olmOCR-2',
        'model_name': 'allenai/olmOCR-2-7B-1025',
        'label_field': 'olmocr_text',
        'description': 'Allen AI olmOCR-2 - Markdown formatted OCR output',
        'size': '~14GB',
        'requires_gpu': True,
    },
    'deepseek_ocr': {
        'source': 'https://github.com/harpreetsahota204/deepseek-ocr',
        'model_name': 'deepseek-ai/DeepSeek-OCR',
        'label_field': 'deepseek_text',
        'description': 'DeepSeek-OCR - High accuracy OCR model',
        'size': '~7GB',
        'requires_gpu': True,
    },
    'nanonets_ocr2': {
        'source': 'https://github.com/harpreetsahota204/nanonets-ocr',
        'model_name': 'nanonets-ocr2',
        'label_field': 'nanonets_text',
        'description': 'Nanonets-OCR2 - Enterprise OCR solution',
        'size': '~5GB',
        'requires_gpu': True,
    },
    'florence2': {
        'source': 'https://github.com/harpreetsahota204/florence-2',
        'model_name': 'microsoft/Florence-2-large',
        'label_field': 'florence_text',
        'description': 'Microsoft Florence-2 - Vision model with OCR task',
        'task': '<OCR>',
        'size': '~1.5GB',
        'requires_gpu': False,  # Can run on CPU
    },

    # -------------------------------------------------------------------------
    # Document Parsing
    # -------------------------------------------------------------------------
    'mineru_2_5': {
        'source': 'https://github.com/harpreetsahota204/mineru',
        'model_name': 'mineru-2.5',
        'label_field': 'mineru_text',
        'description': 'MinerU2.5 - Document parsing and OCR',
        'size': '~3GB',
        'requires_gpu': True,
    },

    # -------------------------------------------------------------------------
    # Traditional OCR Plugins
    # -------------------------------------------------------------------------
    'pytesseract_plugin': {
        'source': 'https://github.com/jacobmarks/pytesseract-ocr-plugin',
        'model_name': 'pytesseract',
        'label_field': 'pytesseract_plugin_text',
        'description': 'PyTesseract - FiftyOne plugin wrapper for Tesseract',
        'size': '~50MB',
        'requires_gpu': False,
    },
    'donut_plugin': {
        'source': 'https://github.com/harpreetsahota204/donut',
        'model_name': 'naver-clova-ix/donut-base',
        'label_field': 'donut_plugin_text',
        'description': 'Donut - Document Understanding Transformer',
        'size': '~800MB',
        'requires_gpu': True,
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def list_fiftyone_models() -> List[str]:
    """List all available FiftyOne OCR models."""
    return list(FIFTYONE_OCR_MODELS.keys())


def get_model_info(name: str) -> Dict:
    """Get information about a FiftyOne OCR model."""
    if name not in FIFTYONE_OCR_MODELS:
        raise ValueError(f"Unknown model: {name}. Available: {list_fiftyone_models()}")
    return FIFTYONE_OCR_MODELS[name]


def list_models_by_size() -> Dict[str, List[str]]:
    """Group models by size category."""
    small = []   # < 2GB
    medium = []  # 2-10GB
    large = []   # > 10GB

    for name, info in FIFTYONE_OCR_MODELS.items():
        size = info.get('size', '~5GB')
        # Parse size string
        if 'GB' in size:
            gb = float(size.replace('~', '').replace('GB', ''))
            if gb < 2:
                small.append(name)
            elif gb <= 10:
                medium.append(name)
            else:
                large.append(name)
        else:
            small.append(name)

    return {'small': small, 'medium': medium, 'large': large}


def list_cpu_compatible() -> List[str]:
    """List models that can run without GPU."""
    return [name for name, info in FIFTYONE_OCR_MODELS.items()
            if not info.get('requires_gpu', True)]


def register_model_source(name: str) -> bool:
    """Register a FiftyOne OCR model source."""
    import fiftyone.zoo as foz

    info = get_model_info(name)
    try:
        foz.register_zoo_model_source(info['source'], overwrite=True)
        print(f"Registered model source: {name}")
        return True
    except Exception as e:
        print(f"Failed to register {name}: {e}")
        return False


def register_all_sources():
    """Register all FiftyOne OCR model sources."""
    for name in FIFTYONE_OCR_MODELS:
        register_model_source(name)


def load_fiftyone_model(name: str, **kwargs):
    """Load a FiftyOne OCR model."""
    import fiftyone.zoo as foz

    info = get_model_info(name)

    # Register source if needed
    register_model_source(name)

    # Load model
    model = foz.load_zoo_model(info['model_name'], **kwargs)

    # Set operation mode if specified
    if 'operation' in info:
        model.operation = info['operation']
    if 'task' in info:
        model.task = info['task']

    return model


def apply_fiftyone_ocr(dataset, model_name: str, **kwargs) -> str:
    """Apply a FiftyOne OCR model to a dataset.

    Args:
        dataset: FiftyOne dataset
        model_name: Name of the OCR model to use
        **kwargs: Additional arguments for the model

    Returns:
        Name of the label field where results are stored
    """
    info = get_model_info(model_name)
    label_field = info['label_field']

    print(f"Loading {model_name} model...")
    model = load_fiftyone_model(model_name, **kwargs)

    print(f"Applying {model_name} to {len(dataset)} samples...")
    start_time = time.time()

    dataset.apply_model(model, label_field=label_field)

    elapsed = time.time() - start_time
    print(f"Completed in {elapsed:.1f}s ({elapsed/len(dataset):.2f}s per sample)")

    return label_field


def run_fiftyone_ocr_comparison(
    dataset,
    models: List[str] = None,
    labels_dir: Path = None
):
    """Run multiple FiftyOne OCR models on a dataset and calculate metrics.

    Args:
        dataset: FiftyOne dataset
        models: List of model names to run (default: all available)
        labels_dir: Path to ground truth labels (optional)
    """
    from ocr_engine import load_ground_truth, calculate_similarity

    if models is None:
        models = list_fiftyone_models()

    print(f"Running FiftyOne OCR comparison with models: {models}")

    for model_name in models:
        try:
            label_field = apply_fiftyone_ocr(dataset, model_name)

            # Calculate similarity if ground truth is available
            if labels_dir:
                labels_dir = Path(labels_dir)
                for sample in dataset:
                    image_id = sample.image_id or Path(sample.filepath).stem
                    csv_path = labels_dir / f"{image_id}.csv"
                    gt = load_ground_truth(str(csv_path))

                    if gt:
                        # Get OCR text from the label field
                        ocr_result = sample.get_field(label_field)
                        if ocr_result:
                            # Handle different output formats
                            if hasattr(ocr_result, 'detections'):
                                # Detections format
                                text = ' '.join([d.label for d in ocr_result.detections])
                            elif isinstance(ocr_result, str):
                                text = ocr_result
                            else:
                                text = str(ocr_result)

                            gt_combined = ' '.join(str(v) for v in gt.values() if v and v != 'NONE')
                            similarity = calculate_similarity(text, gt_combined)
                            sample[f'{model_name}_similarity'] = similarity

                    sample.save()

        except Exception as e:
            print(f"Error running {model_name}: {e}")
            continue

    print("FiftyOne OCR comparison complete!")


def print_fiftyone_models():
    """Print information about available FiftyOne OCR models."""
    print("\n" + "=" * 70)
    print("FiftyOne OCR Plugin Models")
    print("=" * 70)
    print(f"\nTotal models: {len(FIFTYONE_OCR_MODELS)}")

    # Group by size
    by_size = list_models_by_size()
    print(f"\nBy size:")
    print(f"  Small (<2GB):    {len(by_size['small'])} models")
    print(f"  Medium (2-10GB): {len(by_size['medium'])} models")
    print(f"  Large (>10GB):   {len(by_size['large'])} models")

    # CPU compatible
    cpu_models = list_cpu_compatible()
    print(f"\nCPU compatible: {len(cpu_models)} models")
    print(f"  {', '.join(cpu_models)}")

    print("\n" + "-" * 70)
    print("All Models:")
    print("-" * 70)

    for name, info in FIFTYONE_OCR_MODELS.items():
        gpu = "GPU" if info.get('requires_gpu', True) else "CPU"
        size = info.get('size', '?')
        print(f"\n  {name}:")
        print(f"    {info['description']}")
        print(f"    Size: {size} | {gpu} | Model: {info['model_name']}")


if __name__ == '__main__':
    print_fiftyone_models()
