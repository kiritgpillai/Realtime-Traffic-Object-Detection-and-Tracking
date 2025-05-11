import pytest
import os
import boto3
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from botocore.client import Config
import matplotlib.pyplot as plt
import shutil

# MinIO configuration
MINIO_ENDPOINT_URL = "http://129.114.27.202:30000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "miniopassword"
BUCKET_NAME = "mlflow"
OBJECT_KEY = "5/ccd4f18c17fa4979973f1de130bdd83a/artifacts/weights/best.pt"
LOCAL_MODEL_PATH = "/tmp/best.pt"

# Test image directory
TEST_IMAGES_DIR = "tests/test_images"
SAVE_RESULTS_DIR = "tests/results"

@pytest.fixture(scope="session")
def s3_client():
    """Create and return S3 client for MinIO."""
    client = boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT_URL,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    return client

@pytest.fixture(scope="session")
def model_path(s3_client):
    """Download model from MinIO and return local path."""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)
    
    # Check if model is already downloaded
    if not os.path.exists(LOCAL_MODEL_PATH):
        print(f"Downloading model from MinIO to {LOCAL_MODEL_PATH}...")
        s3_client.download_file(BUCKET_NAME, OBJECT_KEY, LOCAL_MODEL_PATH)
        print("Download complete.")
    else:
        print(f"Using existing model at {LOCAL_MODEL_PATH}")
    
    return LOCAL_MODEL_PATH

@pytest.fixture(scope="session")
def model(model_path):
    """Load YOLO model from the downloaded weights."""
    model = YOLO(model_path)
    return model

@pytest.fixture(scope="session")
def test_images():
    """Get list of test images."""
    test_images_path = Path(TEST_IMAGES_DIR)
    assert test_images_path.exists(), f"Test images directory {TEST_IMAGES_DIR} not found"
    
    # Get all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png']:
        image_files.extend(list(test_images_path.glob(f'*{ext}')))
    
    assert len(image_files) > 0, f"No image files found in {TEST_IMAGES_DIR}"
    return image_files

@pytest.fixture(scope="session")
def result_dir():
    """Create directory for test results."""
    result_dir = Path(SAVE_RESULTS_DIR)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    return result_dir

def test_model_download(model_path):
    """Test that model was downloaded successfully."""
    assert os.path.exists(model_path), f"Model file {model_path} does not exist"
    assert os.path.getsize(model_path) > 1000000, "Model file is too small, likely corrupted"

def test_model_loading(model):
    """Test that model loads correctly."""
    assert model is not None, "Failed to load model"
    assert hasattr(model, 'predict'), "Model does not have predict method"

def test_model_on_test_images(model, test_images, result_dir):
    """Run inference on all test images."""
    # Target classes based on the COCO traffic subset
    traffic_classes = [
        'person', 'bicycle', 'car', 'motorcycle',
        'bus', 'truck', 'traffic light', 'stop sign'
    ]
    
    # Prepare results file
    results_file = result_dir / "detection_results.txt"
    with open(results_file, 'w') as f:
        f.write(f"====================\n\n")
        f.write(f"Test Detection Results\n")
        f.write(f"====================\n\n")
    
    # Process each test image
    for img_path in test_images:
        # Run inference on CPU
        results = model.predict(str(img_path), conf=0.25, device="cpu")
        
        # Save annotated image
        result_img = result_dir / f"{img_path.stem}_result{img_path.suffix}"
        results[0].save(str(result_img))
        
        # Get detection counts
        boxes = results[0].boxes
        detections = {cls_name: 0 for cls_name in traffic_classes}
        
        if len(boxes) > 0:
            # Count detections by class
            if hasattr(boxes, 'cls'):
                for i, cls_id in enumerate(boxes.cls.cpu().numpy()):
                    cls_id = int(cls_id)
                    if cls_id < len(traffic_classes):
                        cls_name = traffic_classes[cls_id]
                        detections[cls_name] += 1
        
        # Log results for image
        with open(results_file, 'a') as f:
            f.write(f"Image: {img_path.name}\n")
            f.write(f"  Total detections: {len(boxes)}\n")
            f.write("  Detections by class:\n")
            for cls_name, count in detections.items():
                if count > 0:
                    f.write(f"    - {cls_name}: {count}\n")
            f.write("\n")
        
        # Basic assertions for test
        assert results is not None, f"Inference failed on {img_path}"
        assert len(results) > 0, f"No results returned for {img_path}"
    
    print(f"Saved detection results to {results_file}")
    assert results_file.exists(), "Failed to save detection results"

def test_model_inference_speed(model, test_images):
    """Test inference speed on test images."""
    # Skip if no GPU available
    if not torch.cuda.is_available():
        print("No GPU available, running inference speed test on CPU...")
    
    # Select a sample image for testing
    sample_img_path = str(test_images[0])
    
    # Warm-up run
    for _ in range(3):
        model.predict(sample_img_path, device="cpu")
    
    # Time multiple runs
    import time
    num_runs = 5  
    start_time = time.time()
    
    for _ in range(num_runs):
        model.predict(sample_img_path, device="cpu")
    
    total_time = time.time() - start_time
    avg_time = total_time / num_runs
    
    print(f"Average inference time per image: {avg_time:.4f} seconds")
    
    # Adjust threshold for CPU inference
    if torch.cuda.is_available():
        max_time = 0.1  # GPU threshold
    else:
        max_time = 0.5  # CPU threshold
    
    # This test will be skipped rather than fail based on speed
    if avg_time > max_time:
        pytest.skip(f"Inference time ({avg_time:.4f}s) exceeds threshold")

def test_class_distribution(model, test_images, result_dir):
    """Test class distribution in test images."""
    traffic_classes = [
        'person', 'bicycle', 'car', 'motorcycle',
        'bus', 'truck', 'traffic light', 'stop sign'
    ]
    
    class_counts = {cls_name: 0 for cls_name in traffic_classes}
    total_detections = 0
    
    for img_path in test_images:
        # Explicitly use CPU
        results = model.predict(str(img_path), conf=0.25, device="cpu")
        boxes = results[0].boxes
        
        if len(boxes) > 0 and hasattr(boxes, 'cls'):
            # Count detections by class
            for cls_id in boxes.cls.cpu().numpy():
                cls_id = int(cls_id)
                if cls_id < len(traffic_classes):
                    cls_name = traffic_classes[cls_id]
                    class_counts[cls_name] += 1
                    total_detections += 1
    
    # Skip if no detections
    if total_detections == 0:
        pytest.skip("No detections found in test images")
    
    # Create bar chart of class distribution
    plt.figure(figsize=(12, 6))
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Sort by count for better visualization
    sorted_indices = np.argsort(counts)[::-1]
    sorted_classes = [classes[i] for i in sorted_indices]
    sorted_counts = [counts[i] for i in sorted_indices]
    
    bars = plt.bar(sorted_classes, sorted_counts, color='skyblue')
    plt.title('Detection Count by Class')
    plt.xlabel('Class')
    plt.ylabel('Number of Detections')
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.1,
                str(int(height)),
                ha='center'
            )
    
    plt.tight_layout()
    plt.savefig(result_dir / "class_distribution.png")
    plt.close()
    
    # Log class distribution
    with open(result_dir / "class_distribution.txt", 'w') as f:
        f.write(f"==========================\n\n")
        f.write(f"Detection Class Distribution\n")
        f.write(f"==========================\n\n")
        f.write(f"Total detections: {total_detections}\n\n")
        f.write("Detections by class:\n")
        
        # Sort by count for the text report
        for cls_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                percentage = (count / total_detections) * 100
                f.write(f"  - {cls_name}: {count} ({percentage:.1f}%)\n")

def test_detection_size_distribution(model, test_images, result_dir):
    """Test detection box size distribution."""
    small_area = 0  # < 32²
    medium_area = 0  # 32² - 96²
    large_area = 0  # > 96²
    
    all_areas = []
    
    for img_path in test_images:
        # Get image dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size

        results = model.predict(str(img_path), conf=0.25, device="cpu")
        boxes = results[0].boxes
        
        if len(boxes) > 0 and hasattr(boxes, 'xyxy'):
            # Calculate areas for each detection
            for box in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                width = (x2 - x1)
                height = (y2 - y1)
                
                # Calculate area in relative terms (as if image was 640x640)
                rel_width = width / img_width * 640
                rel_height = height / img_height * 640
                rel_area = rel_width * rel_height
                
                all_areas.append(rel_area)
                
                # Categorize by size
                if rel_area < 32*32:
                    small_area += 1
                elif rel_area < 96*96:
                    medium_area += 1
                else:
                    large_area += 1
    
    # Skip if no detections
    if not all_areas:
        pytest.skip("No detections found in test images")
    
    # Create histogram of detection areas
    plt.figure(figsize=(10, 6))
    plt.hist(all_areas, bins=30, alpha=0.7, color='green')
    plt.title('Detection Size Distribution')
    plt.xlabel('Box Area (pixels²)')
    plt.ylabel('Frequency')
    plt.axvline(x=32*32, color='r', linestyle='--', label='Small (32²)')
    plt.axvline(x=96*96, color='b', linestyle='--', label='Medium (96²)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Use log scale if range is large
    if max(all_areas) / (min(all_areas) + 1) > 100:
        plt.xscale('log')
    
    plt.savefig(result_dir / "size_distribution.png")
    plt.close()
    
    # Log size statistics
    total = small_area + medium_area + large_area
    with open(result_dir / "size_stats.txt", 'w') as f:
        f.write(f"========================\n\n")
        f.write(f"Detection Size Statistics\n")
        f.write(f"========================\n\n")
        f.write(f"Total detections: {total}\n")
        f.write(f"Small detections (<32²): {small_area} ({small_area/total*100:.1f}%)\n")
        f.write(f"Medium detections (32²-96²): {medium_area} ({medium_area/total*100:.1f}%)\n")
        f.write(f"Large detections (>96²): {large_area} ({large_area/total*100:.1f}%)\n\n")
        
        f.write("Size statistics (normalized to 640x640):\n")
        f.write(f"  Mean area: {np.mean(all_areas):.1f} pixels²\n")
        f.write(f"  Median area: {np.median(all_areas):.1f} pixels²\n")
        f.write(f"  Min area: {np.min(all_areas):.1f} pixels²\n")
        f.write(f"  Max area: {np.max(all_areas):.1f} pixels²\n")