#!/usr/bin/env python3
"""Inference script for YOLO models stored in MinIO bucket with result upload."""

import os
import sys
import time
import json
import argparse
import tempfile
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import torch
from minio import Minio
from minio.error import S3Error
import mlflow
from ultralytics import YOLO
import cv2
import numpy as np


class Config:
    """Default configuration parameters."""
    MINIO_ENDPOINT = "129.114.27.202:30000"
    MINIO_ACCESS_KEY = "minioadmin"
    MINIO_SECRET_KEY = "miniopassword"
    MINIO_SECURE = False
    MLFLOW_TRACKING_URI = "http://129.114.27.202:30938"
    MLFLOW_EXPERIMENT_NAME = "coco_runs"
    DEFAULT_CONF_THRESHOLD = 0.40


class GPUConfig:
    """GPU configuration."""
    @staticmethod
    def set_amd_env_vars():
        """Set AMD GPU environment variables."""
        os.environ["HSA_ENABLE_SDMA"] = "0"
        os.environ["GPU_MAX_HEAP_SIZE"] = "100"
        os.environ["GPU_MAX_ALLOC_PERCENT"] = "100"
        os.environ["HIP_VISIBLE_DEVICES"] = "0,1"


class ArgumentParser:
    """Command line argument parser."""
    
    @staticmethod
    def parse_args() -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="YOLO inference with MinIO integration")
        
        # Model selection options
        model_group = parser.add_argument_group("Model Selection")
        model_group.add_argument("--run-id", type=str, help="MLflow run ID")
        model_group.add_argument("--run-name", type=str, help="MLflow run name")
        model_group.add_argument("--model-path", type=str, help="Direct path 'bucket/path/to/model.pt'")
        model_group.add_argument("--latest", action="store_true", help="Use latest model")
        model_group.add_argument("--explore", action="store_true", help="Explore MinIO bucket")
        model_group.add_argument("--bucket", type=str, help="Bucket name to explore")
        model_group.add_argument("--list-buckets", action="store_true", help="List all buckets")
        
        # Input options
        input_group = parser.add_argument_group("Input")
        input_group.add_argument("--source", type=str, help="Source for inference")
        input_group.add_argument("--conf", type=float, default=Config.DEFAULT_CONF_THRESHOLD, help="Confidence threshold")
        input_group.add_argument("--iou", type=float, default=0.45, help="IOU threshold for NMS")
        input_group.add_argument("--max-det", type=int, default=300, help="Maximum detections per image")
        
        # Output options
        output_group = parser.add_argument_group("Output")
        output_group.add_argument("--save-dir", type=str, default="results", help="Local save directory")
        output_group.add_argument("--show", action="store_true", help="Display results")
        output_group.add_argument("--save", action="store_true", help="Save results")
        output_group.add_argument("--upload", action="store_true", help="Upload results to MinIO")
        
        # MinIO and MLflow options
        server_group = parser.add_argument_group("Server Configuration")
        server_group.add_argument("--minio-endpoint", type=str, help="MinIO endpoint")
        server_group.add_argument("--minio-access-key", type=str, help="MinIO access key")
        server_group.add_argument("--minio-secret-key", type=str, help="MinIO secret key")
        server_group.add_argument("--mlflow-uri", type=str, help="MLflow tracking URI")
        server_group.add_argument("--experiment-name", type=str, help="MLflow experiment name")
        
        # Device options
        device_group = parser.add_argument_group("Device")
        device_group.add_argument("--device", type=str, default="", help="Device to use")
        
        args = parser.parse_args()
        
        # Ensure model selection or exploration is specified
        if not (args.run_id or args.run_name or args.model_path or args.latest or args.explore or args.list_buckets or args.bucket):
            parser.error("Specify a model selection method: --run-id, --run-name, --model-path, --latest, --explore")
        
        # If not in exploration mode, ensure source is provided
        if not (args.explore or args.list_buckets) and not args.source:
            parser.error("You must specify a source for inference with --source")
        
        # Default to saving and uploading if not specified
        if args.source and not (args.show or args.save or args.upload):
            args.save = True
            args.upload = True
        
        return args


class MinIOClient:
    """MinIO client wrapper."""
    
    def __init__(self, args: Optional[argparse.Namespace]):
        """Initialize MinIO client."""
        self.client = None
        if args is not None:
            self.client = self._create_client(args)
    
    def _create_client(self, args: argparse.Namespace) -> Minio:
        """Create MinIO client instance."""
        endpoint = args.minio_endpoint or Config.MINIO_ENDPOINT
        access_key = args.minio_access_key or Config.MINIO_ACCESS_KEY
        secret_key = args.minio_secret_key or Config.MINIO_SECRET_KEY
        
        try:
            client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=Config.MINIO_SECURE)
            print(f"Connected to MinIO at {endpoint}")
            return client
        except Exception as e:
            print(f"Error connecting to MinIO: {e}")
            sys.exit(1)
    
    def list_buckets(self) -> None:
        """List all buckets in MinIO."""
        try:
            buckets = self.client.list_buckets()
            print("\nAvailable buckets:")
            for bucket in buckets:
                print(f"  - {bucket.name} (created: {bucket.creation_date})")
        except Exception as e:
            print(f"Error listing buckets: {e}")
    
    def explore_bucket(self, bucket_name: str, prefix: str = "", recursive: bool = True) -> None:
        """Explore contents of a MinIO bucket."""
        try:
            if not self.client.bucket_exists(bucket_name):
                print(f"Bucket {bucket_name} does not exist")
                return
            
            objects = self.client.list_objects(bucket_name, prefix=prefix, recursive=recursive)
            model_files = []
            
            print(f"\nExploring bucket: {bucket_name} (prefix: {prefix or '(none)'})")
            
            for obj in objects:
                if obj.object_name.endswith(".pt"):
                    model_files.append((obj.object_name, obj.size))
            
            model_files.sort(key=lambda x: x[0])
            
            if model_files:
                print(f"\nFound {len(model_files)} model files:")
                for name, size in model_files:
                    print(f"  - {bucket_name}/{name} ({size/1024/1024:.2f} MB)")
            else:
                print("No model files found in this location.")
                
                if prefix:
                    print("\nSearching entire bucket for model files...")
                    broader_objects = self.client.list_objects(bucket_name, recursive=True)
                    
                    broader_models = []
                    for obj in broader_objects:
                        if obj.object_name.endswith(".pt"):
                            broader_models.append((obj.object_name, obj.size))
                    
                    broader_models.sort(key=lambda x: x[0])
                    
                    if broader_models:
                        print(f"\nFound {len(broader_models)} model files in the bucket:")
                        for name, size in broader_models:
                            print(f"  - {bucket_name}/{name} ({size/1024/1024:.2f} MB)")
                    else:
                        print("No model files found in the entire bucket.")
        
        except Exception as e:
            print(f"Error exploring bucket {bucket_name}: {e}")
    
    def download_model(self, bucket_name: str, object_path: str) -> Optional[str]:
        """Download model from MinIO bucket."""
        try:
            if not self.client.bucket_exists(bucket_name):
                print(f"Bucket {bucket_name} does not exist")
                return None
            
            try:
                stat = self.client.stat_object(bucket_name, object_path)
                print(f"Model file found: {object_path} ({stat.size/1024/1024:.2f} MB)")
            except S3Error:
                print(f"Model file not found at {bucket_name}/{object_path}")
                return None
            
            temp_dir = tempfile.mkdtemp()
            local_model_path = os.path.join(temp_dir, "model.pt")
            
            print(f"Downloading model from s3://{bucket_name}/{object_path}")
            self.client.fget_object(bucket_name, object_path, local_model_path)
            
            return local_model_path
        
        except Exception as e:
            print(f"Error downloading model: {e}")
            return None
    
    def upload_results(self, bucket_name: str, base_path: str, local_results_dir: str) -> bool:
        """Upload inference results to MinIO bucket."""
        try:
            if not self.client.bucket_exists(bucket_name):
                print(f"Bucket {bucket_name} does not exist for uploading results")
                return False
            
            # Create a timestamp-based inference folder
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            inference_folder = f"{base_path}/inference_{timestamp}"
            
            print(f"Uploading inference results to s3://{bucket_name}/{inference_folder}")
            
            # Upload all files in the results directory
            upload_count = 0
            for root, _, files in os.walk(local_results_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    
                    # Get relative path from local_results_dir
                    rel_path = os.path.relpath(local_path, local_results_dir)
                    
                    # Construct target path in MinIO
                    target_path = f"{inference_folder}/{rel_path}"
                    
                    # Get content type based on file extension
                    content_type = self._get_content_type(file)
                    
                    # Upload file
                    self.client.fput_object(
                        bucket_name, 
                        target_path, 
                        local_path,
                        content_type=content_type
                    )
                    
                    upload_count += 1
                    print(f"Uploaded {rel_path} to {target_path}")
            
            print(f"Successfully uploaded {upload_count} result files to MinIO")
            print(f"Results available at: s3://{bucket_name}/{inference_folder}")
            return True
        
        except Exception as e:
            print(f"Error uploading results to MinIO: {e}")
            return False
    
    @staticmethod
    def _get_content_type(filename: str) -> str:
        """Determine content type based on file extension."""
        if filename.endswith((".jpg", ".jpeg")):
            return "image/jpeg"
        elif filename.endswith(".png"):
            return "image/png"
        elif filename.endswith(".json"):
            return "application/json"
        elif filename.endswith(".txt"):
            return "text/plain"
        else:
            return "application/octet-stream"


class MLFlowManager:
    """MLFlow manager."""
    
    def __init__(self, args: Optional[argparse.Namespace]):
        """Initialize MLFlow manager."""
        if args is not None:
            self.tracking_uri = args.mlflow_uri or Config.MLFLOW_TRACKING_URI
            self.experiment_name = args.experiment_name or Config.MLFLOW_EXPERIMENT_NAME
        else:
            self.tracking_uri = Config.MLFLOW_TRACKING_URI
            self.experiment_name = Config.MLFLOW_EXPERIMENT_NAME
            
        self._setup_mlflow()
    
    def _setup_mlflow(self) -> None:
        """Set up MLFlow connection."""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            print(f"Connected to MLflow: {self.tracking_uri}, experiment: {self.experiment_name}")
        except Exception as e:
            print(f"Error connecting to MLflow: {e}")
            sys.exit(1)
    
    def find_artifact_location(self) -> Optional[str]:
        """Find MLFlow artifact location."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                return None
            
            artifact_location = experiment.artifact_location
            print(f"MLflow artifact location: {artifact_location}")
            
            if artifact_location.startswith("s3://"):
                path = artifact_location[5:]
                bucket = path.split("/", 1)[0]
                return bucket
            else:
                print(f"MLflow artifacts not in S3/MinIO: {artifact_location}")
                return None
        
        except Exception as e:
            print(f"Error finding MLflow artifact location: {e}")
            return None
    
    def find_model_by_run_id(self, minio_client: Minio, run_id: str) -> Optional[Tuple[str, str, str]]:
        """Find model path by MLFlow run ID."""
        try:
            run = mlflow.get_run(run_id)
            if not run:
                print(f"Run with ID {run_id} not found")
                return None
            
            artifact_uri = run.info.artifact_uri
            print(f"Artifact URI: {artifact_uri}")
            
            if not artifact_uri.startswith("s3://"):
                print(f"Artifact URI not in expected format: {artifact_uri}")
                return None
            
            s3_path = artifact_uri[5:]
            parts = s3_path.split("/", 1)
            if len(parts) < 2:
                print(f"Invalid artifact URI format: {artifact_uri}")
                return None
            
            bucket_name = parts[0]
            base_path = parts[1]
            
            if not minio_client.bucket_exists(bucket_name):
                print(f"Bucket {bucket_name} does not exist")
                return None
            
            # Get run name
            run_name = run.data.tags.get("mlflow.runName", "")
            
            # Search for model file
            model_path = self._find_model_file(minio_client, bucket_name, base_path, run_name)
            if model_path:
                return bucket_name, model_path, base_path
            
            print(f"No model files found for run ID {run_id}")
            return None
        
        except Exception as e:
            print(f"Error finding model by run ID: {e}")
            return None
    
    def find_model_by_run_name(self, minio_client: Minio, run_name: str) -> Optional[Tuple[str, str, str]]:
        """Find model path by MLFlow run name."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                print(f"Experiment {self.experiment_name} not found")
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'"
            )
            
            if runs.empty:
                print(f"No runs found with name {run_name}")
                return None
            
            recent_run = runs.iloc[0]
            run_id = recent_run.run_id
            
            print(f"Found run with ID {run_id} for name {run_name}")
            return self.find_model_by_run_id(minio_client, run_id)
        
        except Exception as e:
            print(f"Error finding model by run name: {e}")
            return None
    
    def find_latest_model(self, minio_client: Minio) -> Optional[Tuple[str, str, str]]:
        """Find latest model in MLFlow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if not experiment:
                print(f"Experiment {self.experiment_name} not found")
                return None
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"]
            )
            
            if runs.empty:
                print(f"No runs found in experiment {self.experiment_name}")
                return None
            
            recent_run = runs.iloc[0]
            run_id = recent_run.run_id
            
            print(f"Using latest run: {run_id} (started at {recent_run.start_time})")
            return self.find_model_by_run_id(minio_client, run_id)
        
        except Exception as e:
            print(f"Error finding latest model: {e}")
            return None
    
    @staticmethod
    def _find_model_file(minio_client: Minio, bucket_name: str, base_path: str, run_name: str = "") -> Optional[str]:
        """Search for model file in various possible locations."""
        # Try several possible paths for the model
        possible_paths = [
            f"{base_path}/models/best.pt",
            f"{base_path}/model/best.pt",
            f"{base_path}/weights/best.pt",
            f"{base_path}/best.pt",
        ]
        
        if run_name:
            possible_paths.extend([
                f"{base_path}/{run_name}/models/best.pt",
                f"{base_path}/{run_name}/model/best.pt",
                f"{base_path}/{run_name}/weights/best.pt",
                f"{base_path}/{run_name}/best.pt",
            ])
        
        for path in possible_paths:
            try:
                minio_client.stat_object(bucket_name, path)
                print(f"Found model at: {bucket_name}/{path}")
                return path
            except S3Error:
                pass
        
        # Search for any .pt file in the artifact path
        objects = minio_client.list_objects(bucket_name, prefix=base_path, recursive=True)
        
        for obj in objects:
            if obj.object_name.endswith(".pt"):
                print(f"Found model file: {bucket_name}/{obj.object_name}")
                return obj.object_name
        
        return None


class ModelFinder:
    """Model finder utility."""
    
    def __init__(self, minio_client: MinIOClient, mlflow_manager: MLFlowManager):
        """Initialize model finder."""
        self.minio_client = minio_client
        self.mlflow_manager = mlflow_manager
    
    def find_model(self, args: argparse.Namespace) -> Optional[Tuple[str, str, str]]:
        """Find model based on command line arguments."""
        if args.run_id:
            print(f"Finding model for run ID: {args.run_id}")
            return self.mlflow_manager.find_model_by_run_id(self.minio_client.client, args.run_id)
        elif args.run_name:
            print(f"Finding model for run name: {args.run_name}")
            return self.mlflow_manager.find_model_by_run_name(self.minio_client.client, args.run_name)
        elif args.latest:
            print("Finding latest model in experiment")
            return self.mlflow_manager.find_latest_model(self.minio_client.client)
        elif args.model_path:
            print(f"Using specified model path: {args.model_path}")
            return self.parse_model_path(args.model_path)
        else:
            print("No model selection method specified")
            return None
    
    @staticmethod
    def parse_model_path(model_path: str) -> Optional[Tuple[str, str, str]]:
        """Parse direct model path."""
        parts = model_path.split("/", 1)
        if len(parts) < 2:
            print(f"Invalid model path format: {model_path}")
            return None
        
        bucket_name = parts[0]
        object_path = parts[1]
        
        # For direct paths, determine the base path (for saving results)
        try:
            # Extract a reasonable base path - up to the last folder
            path_parts = object_path.split("/")
            if len(path_parts) > 1:
                # Remove the filename and go up one directory
                base_path = "/".join(path_parts[:-1])
            else:
                base_path = ""
        except Exception:
            base_path = ""
        
        return bucket_name, object_path, base_path


class YOLOModel:
    """YOLO model wrapper."""
    
    @staticmethod
    def load_model(model_path: str, device: str = "") -> Optional[YOLO]:
        """Load YOLO model from file."""
        try:
            print(f"Loading model from {model_path}")
            model = YOLO(model_path)
            
            if device:
                print(f"Using device: {device}")
                model.to(device)
            
            return model
        
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    @staticmethod
    def run_inference(
        model: YOLO, 
        source: str, 
        conf: float = Config.DEFAULT_CONF_THRESHOLD,
        iou: float = 0.45,
        max_det: int = 300,
        show: bool = False,
        save: bool = True,
        save_dir: str = "results"
    ) -> Dict[str, Any]:
        """Run inference with YOLO model."""
        try:
            if save and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            print(f"Running inference on {source}")
            results = model.predict(
                source=source,
                conf=conf,
                iou=iou,
                max_det=max_det,
                save=save,
                project=save_dir,
                name="predict",
                exist_ok=True,
                verbose=True
            )
            
            # Process results
            inference_results = {
                "timestamp": datetime.datetime.now().isoformat(),
                "source": source,
                "conf_threshold": conf,
                "iou_threshold": iou,
                "max_detections": max_det,
                "results": []
            }
            
            print("\nResults summary:")
            for i, result in enumerate(results):
                boxes = result.boxes
                num_detections = len(boxes)
                
                # Get label counts
                labels = []
                for box in boxes:
                    cls = int(box.cls[0].item())
                    label = model.names[cls]
                    labels.append(label)
                
                # Count occurrences of each label
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                # Print info
                print(f"Image {i+1}: Found {num_detections} objects")
                for label, count in label_counts.items():
                    print(f"  - {label}: {count}")
                
                # Add to results
                image_result = {
                    "image_id": i,
                    "file_name": Path(source).name if isinstance(source, str) else f"image_{i}",
                    "num_detections": num_detections,
                    "labels": label_counts
                }
                inference_results["results"].append(image_result)
            
            if show:
                model.show(results)
                
            if save:
                local_results_path = os.path.join(save_dir, "predict")
                print(f"Results saved to {local_results_path}")
                
                # Save results summary as JSON
                summary_path = os.path.join(local_results_path, "summary.json")
                with open(summary_path, 'w') as f:
                    json.dump(inference_results, f, indent=2)
                
                return {
                    "result_dir": local_results_path,
                    "summary_path": summary_path,
                    "inference_data": inference_results
                }
            
            return {"inference_data": inference_results}
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return {}


class YOLOInferenceApp:
    """Main application class."""
    
    def __init__(self):
        """Initialize application."""
        GPUConfig.set_amd_env_vars()
        self.args = ArgumentParser.parse_args()
        self.minio_client = MinIOClient(self.args)
        self.mlflow_manager = MLFlowManager(self.args)
        self.model_finder = ModelFinder(self.minio_client, self.mlflow_manager)
    
    def run(self) -> None:
        """Run the application."""
        # List buckets if requested
        if self.args.list_buckets:
            self.minio_client.list_buckets()
            return
        
        # Explore MinIO if requested
        if self.args.explore:
            bucket_name = self.args.bucket
            if not bucket_name:
                bucket_name = self.mlflow_manager.find_artifact_location()
            
            if bucket_name:
                self.minio_client.explore_bucket(bucket_name)
            return
        
        # Find model
        model_info = self.model_finder.find_model(self.args)
        if not model_info:
            print("Could not find model. Exiting.")
            sys.exit(1)
        
        bucket_name, object_path, base_path = model_info
        
        # Download model
        model_path = self.minio_client.download_model(bucket_name, object_path)
        if not model_path:
            print("Failed to download model. Exiting.")
            sys.exit(1)
        
        # If no source specified, we're done after downloading the model
        if not self.args.source:
            print(f"Model downloaded successfully to {model_path}")
            return
        
        # Load model
        model = YOLOModel.load_model(model_path, device=self.args.device)
        if not model:
            print("Failed to load model. Exiting.")
            sys.exit(1)
        
        # Run inference
        result_data = YOLOModel.run_inference(
            model=model,
            source=self.args.source,
            conf=self.args.conf,
            iou=self.args.iou,
            max_det=self.args.max_det,
            show=self.args.show,
            save=self.args.save or self.args.upload,
            save_dir=self.args.save_dir
        )
        
        # Upload results to MinIO if requested
        if self.args.upload and result_data and "result_dir" in result_data:
            self.minio_client.upload_results(
                bucket_name, 
                base_path, 
                result_data["result_dir"]
            )
        
        print("Inference completed successfully.")


def main() -> None:
    """Main entry point."""
    app = YOLOInferenceApp()
    app.run()


if __name__ == "__main__":
    main()