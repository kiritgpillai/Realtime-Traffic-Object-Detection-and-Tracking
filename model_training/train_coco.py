#!/usr/bin/env python3
# amd_mi100_yolo_coco_training.py
"""
Efficient YOLO training on COCO dataset with MLflow tracking and MinIO storage.

This script is optimized for AMD MI100 GPUs using ROCm, with automatic multi-GPU utilization,
MLflow experiment tracking, and MinIO S3 storage for artifacts.
"""

import os
import time
import argparse
import subprocess
from pathlib import Path
import json
import glob
import re
from typing import Dict, Any, Optional, List, Tuple

import yaml
import torch
import mlflow
from ultralytics import YOLO, settings
from ultralytics.utils.torch_utils import select_device

# Set MLflow tracking URI and experiment name
MLFLOW_TRACKING_URI = "http://129.114.25.200:30938"
MLFLOW_EXPERIMENT_NAME = "coco_runs"

# Set MinIO configuration
MINIO_ENDPOINT_URL = "http://129.114.25.200:30000"
MINIO_ACCESS_KEY = "minioadmin"
MINIO_SECRET_KEY = "miniopassword"

# Default project directory
DEFAULT_PROJECT_DIR = "coco_runs"

# Base directory for storing run counters
RUN_COUNTER_DIR = os.path.expanduser("~/.yolo_runs")
RUN_COUNTER_FILE = os.path.join(RUN_COUNTER_DIR, "run_counter.json")

# Set required environment variables for MinIO integration
os.environ["MLFLOW_S3_ENDPOINT_URL"] = MINIO_ENDPOINT_URL
os.environ["AWS_ACCESS_KEY_ID"] = MINIO_ACCESS_KEY
os.environ["AWS_SECRET_ACCESS_KEY"] = MINIO_SECRET_KEY

# Add ROCm debugging environment variables
os.environ["HSA_ENABLE_SDMA"] = "0"  # Can help with some AMD GPU issues
os.environ["GPU_MAX_HEAP_SIZE"] = "100"  # Increase GPU memory heap size percentage
os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"  # Allow using 100% of GPU memory
os.environ["HIP_VISIBLE_DEVICES"] = "0,1"  # Make both GPUs visible


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLO on COCO dataset with AMD MI100 GPUs")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data", type=str, default="traffic.yaml", help="Path to data config")
    parser.add_argument("--weights", type=str, help="Override model weights from config")
    parser.add_argument("--epochs", type=int, help="Override epochs from config")
    parser.add_argument("--batch-size", type=int, help="Override batch size from config")
    parser.add_argument("--img-size", type=int, help="Override image size from config")
    parser.add_argument("--device", type=str, help="Override device selection from config")
    parser.add_argument("--workers", type=int, help="Override number of workers from config")
    parser.add_argument("--cache", action="store_true", help="Force image caching regardless of config")
    parser.add_argument("--no-cache", action="store_false", dest="cache", help="Disable image caching")
    parser.add_argument("--run-name", type=str, help="Override run name from config")
    parser.add_argument("--tracking-uri", type=str, help="Override MLflow tracking URI")
    parser.add_argument("--minio-endpoint", type=str, help="Override MinIO endpoint URL")
    parser.add_argument("--experiment-name", type=str, help="Override MLflow experiment name")
    return parser.parse_args()


def get_next_run_number() -> int:
    """
    Get the next sequential run number from the counter file.
    
    Returns:
        Next run number as integer
    """
    # Create counter directory if it doesn't exist
    os.makedirs(RUN_COUNTER_DIR, exist_ok=True)
    
    # Read current counter or initialize to 0
    try:
        if os.path.exists(RUN_COUNTER_FILE):
            with open(RUN_COUNTER_FILE, 'r') as f:
                counter_data = json.load(f)
                current_number = counter_data.get('next_run', 0)
        else:
            current_number = 0
            
        # Check existing run directories if counter is out of sync
        run_dirs = glob.glob(f"{DEFAULT_PROJECT_DIR}/run_*")
        run_numbers = []
        for dir_name in run_dirs:
            match = re.search(r'run_(\d+)', dir_name)
            if match:
                run_numbers.append(int(match.group(1)))
        
        # If there are existing run directories, make sure we're higher than all of them
        if run_numbers:
            current_number = max(current_number, max(run_numbers) + 1)
        
        # Increment counter for next use
        next_number = current_number + 1
        
        # Save updated counter
        with open(RUN_COUNTER_FILE, 'w') as f:
            json.dump({'next_run': next_number}, f)
        
        return current_number
    
    except Exception as e:
        print(f"⚠️ Warning: Could not read/write run counter file: {str(e)}")
        # Fallback: use timestamp
        return int(time.time()) % 10000  # Last 4 digits of timestamp


def detect_amd_gpus() -> Tuple[int, str]:
    """
    Detect available AMD GPUs with ROCm and return count and type.
    Returns:
        Tuple containing:
            - Number of GPUs
            - GPU type string
    """
    try:
        result = subprocess.run(
            ['rocm-smi', '--showproductname'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            gpu_count = 0
            gpu_types = []
            for line in result.stdout.splitlines():
                if 'GPU' in line and 'MI100' in line:  # Specifically looking for MI100
                    gpu_count += 1
                    gpu_type = line.split(':')[-1].strip()
                    gpu_types.append(gpu_type)
            
            if gpu_count > 0:
                gpu_type_str = "ROCm: " + ", ".join(gpu_types)
                print(f"Detected {gpu_count} AMD MI100 GPU(s): {gpu_type_str}")
                return gpu_count, gpu_type_str
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        print(f"Error checking for AMD GPUs with rocm-smi: {e}")
    
    # Try alternative method using hip-smi if rocm-smi failed
    try:
        result = subprocess.run(
            ['hip-smi'],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True, 
            timeout=5
        )
        if result.returncode == 0:
            # Count GPU mentions in hip-smi output
            gpu_count = 0
            for line in result.stdout.splitlines():
                if 'GPU' in line and any(['MI100' in line, 'gfx908' in line, 'AMD' in line]):
                    gpu_count += 1
            
            if gpu_count > 0:
                print(f"Detected {gpu_count} AMD GPU(s) using hip-smi")
                return gpu_count, f"ROCm: {gpu_count} AMD GPUs"
    except (subprocess.SubprocessError, subprocess.TimeoutExpired) as e:
        print(f"Error checking for AMD GPUs with hip-smi: {e}")
        
    # If direct detection failed, check for AMD environment variables
    hip_devices = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip_devices and hip_devices != "-1":
        gpu_count = len(hip_devices.split(","))
        print(f"Using {gpu_count} AMD GPU(s) from HIP_VISIBLE_DEVICES environment variable")
        return gpu_count, f"ROCm: {gpu_count} AMD GPUs (from env)"
        
    print("No AMD GPUs detected, falling back to CPU")
    return 0, "None"


def create_run_folder_name() -> str:
    """
    Create a sequential run folder name in the format run_X.
    
    Returns:
        Run folder name as string
    """
    run_number = get_next_run_number()
    return f"run_{run_number}"


def setup_mlflow(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    """
    Set up MLflow and MinIO configurations.
    
    Args:
        cfg: Configuration dictionary
        args: Command line arguments
    """
    # Make sure to end any lingering MLflow runs
    if mlflow.active_run():
        print("Ending any existing MLflow runs before starting")
        mlflow.end_run()
    
    # Use global constants as defaults, but allow overrides
    tracking_uri = args.tracking_uri or cfg.get("tracking_uri", MLFLOW_TRACKING_URI)
    experiment_name = args.experiment_name or cfg.get("experiment_name", MLFLOW_EXPERIMENT_NAME)
    minio_endpoint = args.minio_endpoint or cfg.get("minio_endpoint", MINIO_ENDPOINT_URL)
    
    # Update the config with the values we'll use
    cfg["tracking_uri"] = tracking_uri
    cfg["experiment_name"] = experiment_name
    cfg["minio_endpoint"] = minio_endpoint
    
    # Check if MLflow is already initialized to avoid multiple setups
    if mlflow.get_tracking_uri() != tracking_uri:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI set to: {tracking_uri}")
    else:
        print(f"MLflow already configured with tracking URI: {tracking_uri}")
    
    # Set MLflow experiment
    current_experiment = mlflow.get_experiment_by_name(experiment_name)
    if current_experiment:
        # Use existing experiment
        mlflow.set_experiment(experiment_name)
        print(f"Using existing MLflow experiment: {experiment_name} (ID: {current_experiment.experiment_id})")
    else:
        # Create new experiment
        mlflow.set_experiment(experiment_name)
        print(f"Created new MLflow experiment: {experiment_name}")
    
    # Configure MinIO (S3) endpoint for MLflow artifact storage
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = minio_endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = cfg.get("minio_access_key", MINIO_ACCESS_KEY)
    os.environ["AWS_SECRET_ACCESS_KEY"] = cfg.get("minio_secret_key", MINIO_SECRET_KEY)
    print(f"MinIO endpoint set to: {minio_endpoint}")
    
    # Validate connection to MLflow server
    try:
        # Try to get or create experiment - this will validate the tracking URI
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            print(f"Successfully connected to MLflow server and found experiment: {experiment_name}")
        else:
            print(f"Failed to retrieve experiment, something may be wrong with the MLflow connection")
    except Exception as e:
        print(f"⚠️ Warning: Could not connect to MLflow server at {tracking_uri}: {str(e)}")
        print("Training will continue but metrics may not be logged correctly.")


def setup_dataset(cfg: Dict[str, Any], data_yaml_path: str) -> Dict[str, Any]:
    """
    Set up dataset configurations.
    
    Args:
        cfg: Configuration dictionary
        data_yaml_path: Path to data YAML file
        
    Returns:
        Dataset configuration dictionary
    """
    # Set dataset directory in ultralytics settings
    coco_root = cfg.get("coco_root", "/home/jovyan/work/datasets/coco")
    settings.update({"datasets_dir": str(Path(coco_root).parent)})
    
    # Load base data config
    with open(data_yaml_path) as f:
        base_cfg = yaml.safe_load(f)
    
    # Create dataset config
    data_cfg = {
        "path":        coco_root,
        "train":       os.path.join(coco_root, "images", "train2017"),
        "val":         os.path.join(coco_root, "images", "val2017"),
        "test":        os.path.join(coco_root, "images", "test2017"),
        "annotations": os.path.join(coco_root, "annotations"),
        "labels":      os.path.join(coco_root, "labels"),
        "nc":          base_cfg.get("nc", 8),
        "names":       base_cfg.get("names", []),
    }
    
    print(f"Dataset configuration:")
    print(f"  COCO root      : {data_cfg['path']}")
    print(f"  Train images   : {data_cfg['train']}")
    print(f"  Val images     : {data_cfg['val']}")
    print(f"  Test images    : {data_cfg['test']}")
    
    # Verify dataset paths exist
    for key in ["train", "val"]:
        if not os.path.exists(data_cfg[key]):
            print(f"⚠️ Warning: {key} dataset path does not exist: {data_cfg[key]}")
    
    return data_cfg


def get_optimal_batch_size(gpu_count: int, base_batch: int) -> int:

    if gpu_count == 0:  # CPU only
        return max(1, base_batch // 4)  # Reduce batch size for CPU
    
    # MI100 has 32GB of HBM2 memory, so pushing it for training
    batch_per_gpu = base_batch
    
    # Scale batch size by number of GPUs
    # being conservative to avoid OOM
    total_batch = batch_per_gpu * gpu_count
    
    return max(1, total_batch)


def get_optimal_workers(gpu_count: int, cpu_count: int) -> int:
    """
    Calculate optimal number of dataloader workers for AMD GPUs.
    
    Args:
        gpu_count: Number of available GPUs
        cpu_count: Number of available CPU cores
        
    Returns:
        Optimized number of workers
    """
    # Reserve some cores for system operations
    available_cores = max(1, cpu_count - 4)  # Reserve more cores for AMD setup
    
    if gpu_count == 0:
        # CPU training - use fewer workers to avoid overload
        return min(4, available_cores)
    
    # For AMD GPU training - allocate workers based on GPU count
    return min(6 * gpu_count, available_cores)  # Increasing from 4 to 6 workers per GPU


def log_artifacts(mlflow_run, results_dir: str, best_model_path: str, run_name: str) -> None:
    """
    Log artifacts to MLflow within a consistent subfolder structure.
    
    Args:
        mlflow_run: Active MLflow run
        results_dir: Directory containing results
        best_model_path: Path to best model weights
        run_name: Name of the current run folder
    """
    run_id = mlflow_run.info.run_id
    print(f"Logging artifacts to MLflow under run folder: {run_name} (run_id: {run_id})")
    
    # Dictionary to track what we've already logged to avoid duplicates
    logged_artifacts = set()
    
    # Log best model
    if os.path.exists(best_model_path):
        if best_model_path not in logged_artifacts:
            mlflow.log_artifact(best_model_path, f"{run_name}/models")
            logged_artifacts.add(best_model_path)
            print(f"Logged best model: {best_model_path}")
    else:
        print(f"⚠️ Warning: Best model file not found at {best_model_path}")
    
    # Log dataset info
    if os.path.exists("coco_data_info.txt") and "coco_data_info.txt" not in logged_artifacts:
        mlflow.log_artifact("coco_data_info.txt", f"{run_name}/dataset")
        logged_artifacts.add("coco_data_info.txt")
        print("Logged dataset info")
    
    # Log plots and other artifacts
    if os.path.isdir(results_dir):
        artifacts_logged = 0
        
        # Create categories for different types of artifacts
        categories = {
            '.png': f"{run_name}/plots",
            '.jpg': f"{run_name}/plots",
            '.jpeg': f"{run_name}/plots",
            '.csv': f"{run_name}/metrics",
            '.yaml': f"{run_name}/configs",
            '.json': f"{run_name}/configs",
            '.txt': f"{run_name}/logs",
        }
        
        for fn in os.listdir(results_dir):
            # Determine the appropriate category for this file
            file_ext = os.path.splitext(fn)[1].lower()
            artifact_dir = categories.get(file_ext, f"{run_name}/other")
            
            artifact_path = os.path.join(results_dir, fn)
            if os.path.isfile(artifact_path) and artifact_path not in logged_artifacts:
                try:
                    mlflow.log_artifact(artifact_path, artifact_dir)
                    logged_artifacts.add(artifact_path)
                    artifacts_logged += 1
                    print(f"Logged artifact: {artifact_path} to {artifact_dir}")
                except Exception as e:
                    print(f"⚠️ Error logging artifact {artifact_path}: {str(e)}")
        
        print(f"Total artifacts logged: {artifacts_logged}")
        
        # Log model information as JSON if not already logged
        model_info_path = "model_info.json"
        if model_info_path not in logged_artifacts:
            try:
                model_info = {
                    "model_type": "YOLOv8",
                    "training_dataset": "COCO",
                    "run_folder": run_name,
                    "run_id": run_id,
                    "hardware": "AMD MI100 GPUs",
                    "backend": "ROCm"
                }
                
                with open(model_info_path, "w") as f:
                    json.dump(model_info, f, indent=2)
                
                mlflow.log_artifact(model_info_path, f"{run_name}/metadata")
                logged_artifacts.add(model_info_path)
                print(f"Logged model information to {run_name}/metadata")
            except Exception as e:
                print(f"⚠️ Note: Error with model info logging: {str(e)}")
    else:
        print(f"⚠️ Warning: Results directory not found at {results_dir}")


def main() -> None:
    """Main execution function."""
    # Make sure to end any lingering MLflow runs at the beginning
    if mlflow.active_run():
        print("Ending any existing MLflow runs before starting")
        mlflow.end_run()
    
    args = parse_args()
    start_time = time.time()
    
    # Display welcome message
    print("\n" + "="*80)
    print("  AMD MI100 YOLO COCO TRAINING SCRIPT")
    print("  - Optimized for AMD MI100 GPUs with ROCm")
    print("  - Multi-GPU support")
    print("  - MLflow tracking with MinIO storage")
    print("="*80 + "\n")
    
    # Print information about the runtime environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"ROCm backend: {hasattr(torch, 'version') and hasattr(torch.version, 'hip')}")
    if hasattr(torch, 'version') and hasattr(torch.version, 'hip'):
        print(f"ROCm version: {torch.version.hip}")
    print(f"MLflow version: {mlflow.__version__}")
    
    # Detect available AMD GPUs
    num_gpus, gpu_type = detect_amd_gpus()
    if num_gpus == 0:
        print("⚠️ Warning: No AMD MI100 GPUs detected!")
        print("This script is optimized for AMD MI100 GPUs. Performance may be suboptimal.")
    else:
        print(f"AMD GPU setup looks good: {num_gpus} MI100 GPU(s) detected")
    
    # Check AMD GPU capabilities
    try:
        rocm_info = subprocess.run(['rocm-smi', '--showmeminfo', 'vram'], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    timeout=5)
        print("GPU Memory Information:")
        for line in rocm_info.stdout.splitlines():
            if "vram" in line.lower() or "memory" in line.lower():
                print(f"  {line.strip()}")
    except:
        print("Could not retrieve detailed GPU information")
    
    # Load configuration
    try:
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        print(f"Loaded configuration from {args.config}")
    except Exception as e:
        print(f"Error loading config from {args.config}: {str(e)}")
        print(f"Using default configuration")
        cfg = {
            "model_weights": "yolov8n.pt",
            "epochs": 3,
            "imgsz": 640,
            "batch_size": 16,
            "project": DEFAULT_PROJECT_DIR,
            "run_name": "detection_v1",
            "cache": True,
            "patience": 30,
        }
    
    # Update config with command line arguments
    if args.weights:
        cfg["model_weights"] = args.weights
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.img_size:
        cfg["imgsz"] = args.img_size
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.run_name:
        cfg["run_name"] = args.run_name
    if args.workers:
        cfg["workers"] = args.workers
    
    # Generate sequential run folder name
    run_folder_name = create_run_folder_name()
    print(f"Using run folder: {run_folder_name}")
    
    # Ensure project directory matches experiment name for consistency
    if "project" not in cfg or not cfg["project"]:
        cfg["project"] = DEFAULT_PROJECT_DIR
    else:
        # Notify if we're overriding the project directory for consistency
        if cfg["project"] != DEFAULT_PROJECT_DIR:
            print(f"Note: Changing project directory from '{cfg['project']}' to '{DEFAULT_PROJECT_DIR}' for consistency")
            cfg["project"] = DEFAULT_PROJECT_DIR
    
    # Set run name to match the sequential folder
    cfg["run_name"] = run_folder_name
    print(f"Set run name to: {cfg['run_name']}")
    
    # Override cache setting if specified
    if "cache" in args:
        cfg["cache"] = args.cache
    
    # Set up MLflow and MinIO
    setup_mlflow(cfg, args)
    
    # Determine device setting - for AMD GPUs we use indexed notation
    if args.device:
        device = args.device
    elif num_gpus > 0:
        # Use all available GPUs - specify explicitly for ROCm
        device = ",".join(str(i) for i in range(num_gpus))
    else:
        device = "cpu"
    
    # Get CPU count for worker optimization
    cpu_count = os.cpu_count() or 8
    
    # Set up optimal workers for AMD GPUs
    workers = args.workers if args.workers else get_optimal_workers(num_gpus, cpu_count)
    
    # Calculate optimal batch size for AMD MI100 GPUs
    batch_size = get_optimal_batch_size(num_gpus, cfg["batch_size"])
    
    # Set up dataset
    data_cfg = setup_dataset(cfg, args.data)
    
    # Configure training parameters with AMD-specific optimizations
    model_weights = cfg["model_weights"]
    train_kwargs = {
        "data":    args.data,
        "epochs":  cfg["epochs"],
        "imgsz":   cfg["imgsz"],
        "batch":   batch_size,
        "project": cfg["project"],
        "name":    cfg["run_name"],
        "device":  device,
        "cache":   cfg.get("cache", True),
        "workers": workers,
        "patience": cfg.get("patience", 30),
        "exist_ok": True,  # Overwrite existing runs
        "verbose": True,   # Verbose output
        "amp":     True,   # Use mixed precision training if available
    }
    
    # Log info about the run
    print(f"Starting training with:")
    print(f"  Model weights: {model_weights}")
    print(f"  Device: {device} ({gpu_type})")
    print(f"  Batch size: {batch_size}")
    print(f"  Workers: {workers}")
    print(f"  Image size: {cfg['imgsz']}")
    print(f"  Epochs: {cfg['epochs']}")
    print(f"  Run name: {cfg['run_name']}")
    
    # Start MLflow run and log everything
    # Make sure to use a single run and prevent nested runs
    if mlflow.active_run():
        print("Ending any lingering MLflow runs")
        mlflow.end_run()
        
    print(f"Starting new MLflow run: {cfg['run_name']}")
    with mlflow.start_run(run_name=cfg["run_name"]) as active_run:
        run_id = active_run.info.run_id
        print(f"MLflow run ID: {run_id}")
        print(f"MLflow tracking URL: {mlflow.get_tracking_uri()}")
        print(f"MLflow experiment: {mlflow.get_experiment(active_run.info.experiment_id).name}")
        
        # Create run directory in a consistent way that allows for sequential numbering
        run_name = cfg["run_name"]
        
        # To prevent duplicate logging, only log each parameter once
        params_to_log = {
            "device": device,
            "num_gpus": num_gpus,
            "gpu_type": gpu_type,
            "pytorch_version": torch.__version__,
            "coco_root": data_cfg["path"],
            "data_yaml": args.data,
            "model_weights": model_weights,
            "batch_size_per_device": batch_size / max(1, len(device.split(",")) if device != "cpu" else 1),
            "total_batch_size": batch_size,
            "cpu_count": cpu_count,
            "run_folder": run_name,
            "hardware": "AMD MI100",
            "backend": "ROCm"
        }
        
        # Add training kwargs to params
        for k, v in train_kwargs.items():
            params_to_log[k] = v
        
        # Log all parameters in a single batch to avoid multiple API calls
        mlflow.log_params(params_to_log)
        
        # Log dataset info as artifact
        info_file = "coco_data_info.txt"
        with open(info_file, "w") as f:
            for key in ("path", "train", "val", "test", "annotations", "labels", "nc", "names"):
                f.write(f"{key}: {data_cfg[key]}\n")
        
        # Execute training
        print("\n🚀 Starting YOLO training on AMD MI100 GPUs with MLflow tracking...")
        try:
            # Load model
            model = YOLO(model_weights)
            
            # Training
            results = model.train(**train_kwargs)
            
            # Log training time
            training_time = time.time() - start_time
            
            # Log metrics in a single batch
            metrics_to_log = {"training_time_seconds": training_time}
            
            # Add results metrics
            if hasattr(results, "results_dict"):
                for name, val in results.results_dict.items():
                    safe_name = name.replace("(", "").replace(")", "")
                    metrics_to_log[f"final_{safe_name}"] = val
            
            # Log all metrics in a single batch to avoid multiple API calls
            mlflow.log_metrics(metrics_to_log)
            
            print(f"Training completed in {training_time:.2f} seconds")
            
            # Log model artifact and plots - using the structured folder organization
            results_dir = f"{train_kwargs['project']}/{train_kwargs['name']}"
            best_model = f"{results_dir}/weights/best.pt"
            log_artifacts(active_run, results_dir, best_model, run_name)
            
            # Save a direct link to the MLflow run UI for easy access
            run_url = f"{mlflow.get_tracking_uri().rstrip('/')}/experiments/{active_run.info.experiment_id}/runs/{run_id}"
            with open("mlflow_run_link.txt", "w") as f:
                f.write(f"View run details at: {run_url}\n")
            mlflow.log_artifact("mlflow_run_link.txt", f"{run_name}/links")
            
            print(f"\n✅ MLflow run completed: {run_id}")
            print(f"View run details at: {run_url}")
            
            # Create a symlink to make results more accessible
            try:
                output_dir = os.path.abspath(results_dir)
                link_name = f"latest_{run_name}"
                if os.path.exists(link_name):
                    os.remove(link_name)
                os.symlink(output_dir, link_name)
                print(f"Created symlink to results: {link_name} -> {output_dir}")
            except Exception as e:
                print(f"Note: Could not create symlink to results: {str(e)}")
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            mlflow.log_param("error", str(e))
            
            # For AMD-specific errors, provide more context
            if any(x in str(e).lower() for x in ["rocm", "hip", "amd", "gpu"]):
                print("This appears to be an AMD GPU-specific error.")
                print("Common solutions:")
                print(" - Check for sufficient GPU memory")
                print(" - Reduce batch size")
                print(" - Check ROCm installation")
                print(" - Ensure PyTorch was built for ROCm")
                
                # Try to get more diagnostic information
                try:
                    subprocess.run(['rocm-smi'], stdout=subprocess.PIPE, text=True, timeout=5)
                except Exception as diag_e:
                    print(f"Could not run diagnostics: {diag_e}")
            
            print("Falling back to CPU...")
            train_kwargs["device"] = "cpu"
            train_kwargs["batch"] = max(1, train_kwargs["batch"] // 4)  # Reduce batch size for CPU
            mlflow.log_param("device_fallback", "cpu")
            mlflow.log_param("batch_size_fallback", train_kwargs["batch"])
            
            try:
                model = YOLO(model_weights)
                results = model.train(**train_kwargs)
                
                # Log training time after fallback
                training_time = time.time() - start_time
                
                # Create a single batch of metrics to log
                metrics_to_log = {"training_time_seconds": training_time}
                
                # Add final metrics from CPU fallback
                if hasattr(results, "results_dict"):
                    for name, val in results.results_dict.items():
                        safe_name = name.replace("(", "").replace(")", "")
                        metrics_to_log[f"final_{safe_name}"] = val
                
                # Log all metrics in a single batch
                mlflow.log_metrics(metrics_to_log)
                
                # Log model artifact and plots after fallback
                results_dir = f"{train_kwargs['project']}/{train_kwargs['name']}"
                best_model = f"{results_dir}/weights/best.pt"
                log_artifacts(active_run, results_dir, best_model, run_name)
                
                print("✅ CPU fallback training succeeded")
            except Exception as cpu_e:
                print(f"❌ CPU fallback also failed: {cpu_e}")
                mlflow.log_param("cpu_fallback_error", str(cpu_e))
    
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()