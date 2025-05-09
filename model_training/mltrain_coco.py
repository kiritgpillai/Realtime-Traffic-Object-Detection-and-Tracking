#!/usr/bin/env python3
# mltrain_coco.py

import os
import mlflow
import yaml
import ultralytics
from ultralytics import YOLO
from ultralytics import settings
from ultralytics.data.utils import check_det_dataset
import time
import subprocess
import torch
from pathlib import Path

def check_rocm_gpus():
    """Check for ROCm GPUs and return the number available."""
    try:
        # Try to run rocm-smi to check for AMD GPUs
        result = subprocess.run(['rocm-smi', '--showproductname'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            # Count GPU lines in the output
            gpu_count = 0
            for line in result.stdout.split('\n'):
                if 'GPU' in line and 'MI100' in line:
                    gpu_count += 1
            
            print(f"Detected {gpu_count} AMD MI100 GPU(s) with ROCm")
            return gpu_count
        else:
            print("rocm-smi command failed, assuming no AMD GPUs")
            return 0
    except Exception as e:
        print(f"Error checking for ROCm GPUs: {e}")
        return 0

def main():
    # ── Load config.yaml ──────────────────────────────────────────────────────
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    # ── Point MLflow at your tracking server ─────────────────────────────────
    tracking_uri = cfg["tracking_uri"]
    mlflow.set_tracking_uri(tracking_uri)
    print(f"MLflow tracking URI set to: {tracking_uri}")

    # ── Check for ROCm GPUs ───────────────────────────────────────────────────
    num_gpus = check_rocm_gpus()
    
    # Check PyTorch environment
    print(f"PyTorch version: {torch.__version__}")
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch ROCm available: {hasattr(torch, 'hip') and torch.hip.is_available() if hasattr(torch, 'hip') else 'N/A'}")
    
    # Determine device based on available hardware
    if num_gpus > 0 and hasattr(torch, 'hip') and torch.hip.is_available():
        # Use ROCm for AMD GPUs
        device = "0" if num_gpus == 1 else ",".join(str(i) for i in range(num_gpus))
        print(f"Using AMD GPU(s) with ROCm: device={device}")
    else:
        # Fall back to CPU
        device = "cpu"
        print("Using CPU for training")

    # ── Dataset settings ──────────────────────────────────────────────────────
    # 1. Your project root
    COCO_ROOT = '/home/jovyan/work/coco'
    
    # 2. Tell it where to find the built-in coco.yaml
    coco_yaml = os.path.join(ultralytics.__path__[0], 'cfg', 'datasets', 'coco.yaml')
    
    # Update settings to point to the correct datasets directory
    settings.update({"datasets_dir": os.path.dirname(COCO_ROOT)})
    
    # 3. Validate the existing files (with download if needed)
    data_info = check_det_dataset(coco_yaml, autodownload=True)
    
    # 4. Print out the resolved paths
    print(f"COCO root:        {data_info['path']}")
    print(f" Train images:    {data_info['train']}")   # should be /work/coco/dataset/train2017
    print(f" Val images:      {data_info['val']}")     # should be /work/coco/dataset/val2017
    if 'test' in data_info:
        print(f" Test images:     {data_info['test']}")  # e.g. /work/coco/dataset/test2017
    print(f" Annotations dir: {os.path.join(COCO_ROOT, 'annotations')}")
    print(f" Labels dir:      {os.path.join(COCO_ROOT, 'labels')}")

    # ── Training settings ─────────────────────────────────────────────────────
    model_weights = cfg["model_weights"]
    train_kwargs = {
        "data": coco_yaml,  # Use the COCO yaml directly
        "epochs": cfg["epochs"],
        "imgsz": cfg["imgsz"],
        "batch": cfg["batch_size"],
        "project": cfg["project"],
        "name": cfg["run_name"],
        "device": device,  # Use the detected device
        "cache": cfg.get("cache", True),
        "workers": cfg.get("workers", 8),
    }

    # ── Start MLflow run and log everything ───────────────────────────────────
    with mlflow.start_run(run_name=cfg["run_name"]):
        # Log start time
        start_time = time.time()
        
        # Log hardware info
        mlflow.log_param("device", device)
        mlflow.log_param("num_gpus", num_gpus)
        mlflow.log_param("gpu_type", "AMD MI100" if num_gpus > 0 else "None")
        mlflow.log_param("pytorch_version", torch.__version__)
        
        # Log dataset params
        mlflow.log_param("coco_root", COCO_ROOT)
        mlflow.log_param("coco_yaml", coco_yaml)
        mlflow.log_param("autodownload", True)
        
        # Log model & training params
        mlflow.log_param("model_weights", model_weights)
        for k, v in train_kwargs.items():
            mlflow.log_param(k, v)
        
        # Log dataset info as artifact
        info_file = "coco_data_info.txt"
        with open(info_file, "w") as f:
            f.write(f"COCO root:        {data_info['path']}\n")
            f.write(f"Train images:     {data_info['train']}\n")
            f.write(f"Val images:       {data_info['val']}\n")
            if 'test' in data_info:
                f.write(f"Test images:      {data_info['test']}\n")
            f.write(f"Annotations dir:  {os.path.join(COCO_ROOT, 'annotations')}\n")
            f.write(f"Labels dir:       {os.path.join(COCO_ROOT, 'labels')}\n")
        mlflow.log_artifact(info_file)

        # ── Execute training ──────────────────────────────────────────────────
        print(f"\n🚀 Starting YOLO training with MLflow tracking...")
        try:
            model = YOLO(model_weights)
            results = model.train(**train_kwargs)
            
            # Log training time
            training_time = time.time() - start_time
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Log final metrics
            if hasattr(results, 'results_dict'):
                for metric_name, metric_value in results.results_dict.items():
                    mlflow.log_metric(f"final_{metric_name}", metric_value)
            
            # Log model files as artifacts
            best_model_path = f"{train_kwargs['project']}/{train_kwargs['name']}/weights/best.pt"
            if os.path.exists(best_model_path):
                mlflow.log_artifact(best_model_path, "models")
            
            # Log result plots
            results_dir = f"{train_kwargs['project']}/{train_kwargs['name']}"
            if os.path.exists(results_dir):
                for filename in os.listdir(results_dir):
                    if filename.endswith(('.png', '.jpg')):
                        mlflow.log_artifact(os.path.join(results_dir, filename), "plots")

            run_id = mlflow.active_run().info.run_id
            print(f"\n✅ MLflow run created: {run_id}")
            print(f"   • COCO root: {data_info['path']}")
            print(f"   • Train dir: {data_info['train']}")
            print(f"   • Val dir:   {data_info['val']}")
            
        except Exception as e:
            # Log the error
            mlflow.log_param("error", str(e))
            print(f"❌ Training failed: {e}")
            
            # If CUDA error, try falling back to CPU
            if "CUDA" in str(e) or "GPU" in str(e):
                print("GPU error detected, falling back to CPU...")
                # Update device to CPU
                train_kwargs["device"] = "cpu"
                mlflow.log_param("device_fallback", "cpu")
                
                try:
                    # Try again with CPU
                    model = YOLO(model_weights)
                    results = model.train(**train_kwargs)
                    
                    # Log training time
                    training_time = time.time() - start_time
                    mlflow.log_metric("training_time_seconds", training_time)
                    
                    run_id = mlflow.active_run().info.run_id
                    print(f"\n✅ MLflow run created with CPU fallback: {run_id}")
                except Exception as cpu_e:
                    print(f"❌ CPU fallback training also failed: {cpu_e}")
                    mlflow.log_param("cpu_fallback_error", str(cpu_e))
            else:
                # Re-raise non-GPU related errors
                raise

if __name__ == "__main__":
    main()