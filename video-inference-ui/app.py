#!/usr/bin/env python3
"""
FastAPI Video Inference Service - Handles video uploads, inference, and results streaming.
"""

import os
import sys
import time
import uuid
import json
import asyncio
import tempfile
import shutil
import requests
from pathlib import Path
from typing import Dict, Any, Optional, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.requests import Request
from sse_starlette.sse import EventSourceResponse

import torch
from minio import Minio
from minio.error import S3Error
import mlflow
from ultralytics import YOLO
import cv2
import numpy as np

# Import your existing inference script
import inference_yolo
from inference_yolo import (
    Config, GPUConfig, MinIOClient, MLFlowManager, ModelFinder, YOLOModel
)

# External API configuration
EXTERNAL_API_URL = "http://129.114.27.202:30501"

# Create progress tracking global state
INFERENCE_JOBS = {}

class InferenceJob:
    """Tracks the progress of an inference job."""
    def __init__(self, job_id: str, file_name: str, model_path: str, file_type: str = "video"):
        self.job_id = job_id
        self.file_name = file_name
        self.file_type = file_type  # 'image' or 'video'
        self.model_path = model_path
        self.progress = 0
        self.status = "initializing"
        self.message = "Preparing for inference..."
        self.result_path = None
        self.result_url = None
        self.frames_processed = 0
        self.total_frames = 0 if file_type == "video" else 1
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.logs = []
        
    def update(self, progress: float, status: str, message: str):
        """Update job progress."""
        self.progress = progress
        self.status = status
        self.message = message
        self.last_update_time = time.time()
        self.logs.append({"time": time.time(), "message": message})
        
    def to_dict(self):
        """Convert job status to dictionary."""
        return {
            "job_id": self.job_id,
            "file_name": self.file_name,
            "file_type": self.file_type,
            "progress": self.progress,
            "status": self.status,
            "message": self.message,
            "result_path": self.result_path,
            "result_url": self.result_url,
            "frames_processed": self.frames_processed,
            "total_frames": self.total_frames,
            "elapsed_time": time.time() - self.start_time,
            "last_update": time.time() - self.last_update_time
        }

class VideoInferenceService:
    """Service for handling video inference."""
    
    def __init__(self):
        """Initialize service components."""
        # Set up GPU environment variables
        GPUConfig.set_amd_env_vars()
        
        # Initialize MinIO client
        self.minio_client = Minio(
            Config.MINIO_ENDPOINT,
            access_key=Config.MINIO_ACCESS_KEY,
            secret_key=Config.MINIO_SECRET_KEY,
            secure=Config.MINIO_SECURE
        )
        
        # Initialize MLFlow
        mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
        
        # Create MinIO bucket for results if it doesn't exist
        self.results_bucket = "inference-results"
        try:
            if not self.minio_client.bucket_exists(self.results_bucket):
                self.minio_client.make_bucket(self.results_bucket)
                print(f"Created MinIO bucket: {self.results_bucket}")
        except Exception as e:
            print(f"Error with MinIO bucket setup: {e}")
        
        # Initialize model cache
        self.model_cache = {}
        
        # Setup wrappers from inference script
        # Create a dummy args object for compatibility
        class DummyArgs:
            def __init__(self):
                self.run_id = None
                self.run_name = None
                self.model_path = None
                self.latest = True
                self.minio_endpoint = Config.MINIO_ENDPOINT
                self.minio_access_key = Config.MINIO_ACCESS_KEY
                self.minio_secret_key = Config.MINIO_SECRET_KEY
                self.mlflow_uri = Config.MLFLOW_TRACKING_URI
                self.experiment_name = Config.MLFLOW_EXPERIMENT_NAME
                
        self.dummy_args = DummyArgs()
        self.minio_wrapper = MinIOClient(self.dummy_args)
        self.minio_wrapper.client = self.minio_client
        self.mlflow_manager = MLFlowManager(self.dummy_args)
        self.model_finder = ModelFinder(self.minio_wrapper, self.mlflow_manager)
    
    async def process_media(self, job_id: str, file_path: str, file_type: str, conf: float = 0.25, iou: float = 0.45, max_det: int = 300):
        """
        Process video or image file with YOLO model.
        
        Args:
            job_id: Unique ID for this job
            file_path: Path to uploaded file
            file_type: Type of file ('image' or 'video')
            conf: Confidence threshold
            iou: IOU threshold
            max_det: Maximum detections
        """
        job = INFERENCE_JOBS[job_id]
        job.update(0, "starting", f"Starting {file_type} processing...")
        
        try:
            # Create temporary directory for results
            temp_dir = tempfile.mkdtemp()
            
            if file_type == 'image':
                # Process image using local model but log data from external API for analytics
                job.update(30, "processing", f"Processing image: {job.file_name}")
                
                # First, perform local inference
                try:
                    # Use the specific model path for image processing (same as video)
                    job.update(5, "finding_model", "Using specified YOLO model...")
                    bucket_name = "mlflow"
                    object_path = "5/ccd4f18c17fa4979973f1de130bdd83a/artifacts/models/best.pt"
                    base_path = None
                    
                    # Download model
                    job.update(10, "downloading_model", f"Downloading model from {bucket_name}/{object_path}...")
                    model_path = self.minio_wrapper.download_model(bucket_name, object_path)
                    
                    if not model_path:
                        job.update(0, "error", "Failed to download model")
                        return
                        
                    job.model_path = model_path
                    
                    # Load model
                    job.update(20, "loading_model", "Loading YOLO model...")
                    
                    # Check if model already in cache
                    if model_path in self.model_cache:
                        model = self.model_cache[model_path]
                        job.update(25, "using_cached_model", "Using cached model")
                    else:
                        model = YOLOModel.load_model(model_path)
                        if not model:
                            job.update(0, "error", "Failed to load YOLO model")
                            return
                        # Cache the model for future use
                        self.model_cache[model_path] = model
                    
                    # Read image
                    image = cv2.imread(file_path)
                    if image is None:
                        job.update(0, "error", "Could not open image file")
                        return
                    
                    # Run inference locally
                    job.update(40, "processing", "Running local inference on image")
                    start_time = time.time()
                    results = model.predict(
                        source=image,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        verbose=False
                    )
                    local_inference_time = time.time() - start_time
                    print(f"Local inference completed in {local_inference_time:.4f} seconds")
                    
                    # Draw results on image
                    annotated_image = results[0].plot()
                    
                    # Save output image
                    output_path = os.path.join(temp_dir, f"processed_{Path(file_path).name}")
                    cv2.imwrite(output_path, annotated_image)
                    
                    # In parallel, log analytics data from the external API (non-blocking)
                    asyncio.create_task(self._log_external_api_analytics(job_id, file_path, conf, iou, max_det))
                    
                    job.update(90, "uploading", "Uploading processed image to MinIO...")
                    
                    # Upload processed image to MinIO
                    object_name = f"{job_id}/{Path(output_path).name}"
                    self.minio_client.fput_object(
                        self.results_bucket,
                        object_name,
                        output_path,
                        content_type="image/jpeg"
                    )
                    
                except Exception as e:
                    job.update(0, "error", f"Error during local image processing: {str(e)}")
                    print(f"Error processing image: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
                
            else:  # Video processing
                # Use the specific model path for video processing
                job.update(5, "finding_model", "Using specified YOLO model...")
                bucket_name = "mlflow"
                object_path = "5/ccd4f18c17fa4979973f1de130bdd83a/artifacts/models/best.pt"
                base_path = None
                
                # Download model
                job.update(10, "downloading_model", f"Downloading model from {bucket_name}/{object_path}...")
                model_path = self.minio_wrapper.download_model(bucket_name, object_path)
                
                if not model_path:
                    job.update(0, "error", "Failed to download model")
                    return
                    
                job.model_path = model_path
                
                # Load model
                job.update(20, "loading_model", "Loading YOLO model...")
                
                # Check if model already in cache
                if model_path in self.model_cache:
                    model = self.model_cache[model_path]
                    job.update(25, "using_cached_model", "Using cached model")
                else:
                    model = YOLOModel.load_model(model_path)
                    if not model:
                        job.update(0, "error", "Failed to load YOLO model")
                        return
                    # Cache the model for future use
                    self.model_cache[model_path] = model
                
                job.update(30, "processing", f"Processing video: {job.file_name}")
                
                # Get video information
                video = cv2.VideoCapture(file_path)
                if not video.isOpened():
                    job.update(0, "error", "Could not open video file")
                    return
                    
                fps = video.get(cv2.CAP_PROP_FPS)
                width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
                job.total_frames = total_frames
                
                # Create output video writer
                output_path = os.path.join(temp_dir, f"processed_{Path(file_path).name}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Process video frame by frame
                frame_count = 0
                while video.isOpened():
                    success, frame = video.read()
                    if not success:
                        break
                        
                    frame_count += 1
                    job.frames_processed = frame_count
                    progress = min(30 + int(60 * frame_count / total_frames), 90)
                    
                    # Only update UI every 5 frames to reduce overhead
                    if frame_count % 5 == 0:
                        job.update(progress, "processing", 
                                f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames*100):.1f}%)")
                    
                    # Run inference on frame
                    results = model.predict(
                        source=frame,
                        conf=conf,
                        iou=iou,
                        max_det=max_det,
                        verbose=False
                    )
                    
                    # Draw results on frame
                    annotated_frame = results[0].plot()
                    
                    # Write frame to output video
                    writer.write(annotated_frame)
                
                # Release video resources
                video.release()
                writer.release()
                
                job.update(90, "uploading", "Uploading processed video to MinIO...")
                
                # Upload processed video to MinIO
                object_name = f"{job_id}/{Path(output_path).name}"
                self.minio_client.fput_object(
                    self.results_bucket,
                    object_name,
                    output_path,
                    content_type="video/mp4"
                )
            
            # Generate results metadata
            metadata = {
                "job_id": job_id,
                "original_filename": job.file_name,
                "file_type": file_type,
                "processing_type": "local_model",  # Now all processing is done with local model
                "model_path": bucket_name + "/" + object_path,
                "conf_threshold": conf,
                "iou_threshold": iou,
                "max_detections": max_det,
                "frames_processed": job.frames_processed if file_type == 'video' else 1,
                "total_frames": job.total_frames if file_type == 'video' else 1,
                "timestamp": time.time(),
                "processing_time": time.time() - job.start_time,
                "external_api": {
                    "url": EXTERNAL_API_URL if file_type == "image" else None,
                    "status": "analytics_only" if file_type == "image" else "not_used"
                }
            }
            
            # Save metadata to MinIO
            metadata_path = os.path.join(temp_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
                
            self.minio_client.fput_object(
                self.results_bucket,
                f"{job_id}/metadata.json",
                metadata_path,
                content_type="application/json"
            )
            
            # Update job with result information
            result_url = f"/api/results/{job_id}"
            job.result_path = output_path
            job.result_url = result_url
            job.update(100, "completed", f"{file_type.capitalize()} processing completed successfully")
            
            # Clean up temporary directories after some time
            # We keep the file for download but schedule cleanup
            asyncio.create_task(self._delayed_cleanup(temp_dir, 3600))  # Clean up after 1 hour
            
        except Exception as e:
            job.update(0, "error", f"Error during processing: {str(e)}")
            print(f"Error processing {file_type}: {str(e)}")
    
    async def _delayed_cleanup(self, directory, delay_seconds):
        """Clean up a directory after a delay."""
        await asyncio.sleep(delay_seconds)
        try:
            shutil.rmtree(directory)
            print(f"Cleaned up temporary directory: {directory}")
        except Exception as e:
            print(f"Error cleaning up directory {directory}: {str(e)}")

    async def _log_external_api_analytics(self, job_id, file_path, conf, iou, max_det):
        """Log analytics data from external API without blocking the main workflow."""
        try:
            # Read the image file
            with open(file_path, 'rb') as img_file:
                image_data = img_file.read()
            
            print(f"[Analytics] Sending request to {EXTERNAL_API_URL}/predict for analytics")
            
            # Make the API request for analytics purposes
            response = requests.post(
                f"{EXTERNAL_API_URL}/predict",
                files={'image': (Path(file_path).name, image_data, 'image/jpeg')},
                data={
                    'confidence': str(conf),
                    'iou': str(iou),
                    'max_det': str(max_det)
                },
                timeout=30  # Shorter timeout for analytics
            )
            
            if response.status_code == 200:
                # Log the external API analytics data
                analytics_data = {
                    "job_id": job_id,
                    "external_api": EXTERNAL_API_URL,
                    "timestamp": time.time(),
                    "response_time": response.elapsed.total_seconds(),
                    "status_code": response.status_code,
                }
                
                # Extract performance metrics if available
                try:
                    result_json = response.json()
                    if 'images' in result_json and len(result_json['images']) > 0:
                        if 'speed' in result_json['images'][0]:
                            analytics_data["external_inference_speed"] = result_json['images'][0]['speed']
                except Exception as e:
                    print(f"[Analytics] Error parsing API response: {e}")
                
                print(f"[Analytics] External API analytics logged: {analytics_data}")
                
                # Could save analytics to a database or file here
            else:
                print(f"[Analytics] External API request failed: {response.status_code}")
        
        except Exception as e:
            print(f"[Analytics] Error logging external API data: {e}")
            # Non-blocking, so we just log the error and continue

# Initialize service as a global
video_service = VideoInferenceService()

# Define the app and lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize necessary resources
    print("Starting FastAPI Video Inference Service")
    yield
    # Shutdown: Clean up
    print("Shutting down FastAPI Video Inference Service")
    # Clean up model cache and other resources
    for job_id, job in INFERENCE_JOBS.items():
        if job.status != "completed" and job.status != "error":
            job.update(0, "cancelled", "Service shutdown")

# Create FastAPI app
app = FastAPI(
    title="Video Inference API",
    description="API for YOLO model inference on videos",
    version="1.0.0",
    lifespan=lifespan
)

# Set up CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up templates
templates = Jinja2Templates(directory="templates")

# Make sure directories exist
os.makedirs("static", exist_ok=True)
os.makedirs("static/css", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

class InferenceRequest(BaseModel):
    """Inference request model."""
    confidence: float = 0.25
    iou: float = 0.45
    max_det: int = 300

@app.get("/")
async def root(request: Request):
    """Serve the main page or API info."""
    # Check if the request is from a browser
    accept_header = request.headers.get("accept", "")
    if "text/html" in accept_header:
        # Serve HTML page
        return templates.TemplateResponse("index.html", {"request": request})
    else:
        # Serve API info for programmatic requests
        return {
            "service": "Video and Image Inference Service",
            "version": "1.0",
            "endpoints": {
                "/api/inference": "POST - Detect objects in uploaded video or image",
                "/api/results/{job_id}": "GET - Get processed media file",
                "/api/jobs/{job_id}": "GET - Get job status",
                "/api/jobs/{job_id}/stream": "GET - Stream job progress updates",
                "/health": "GET - Check service health",
                "/metrics": "GET - Prometheus metrics"
            },
            "external_integrations": {
                "image_api": EXTERNAL_API_URL
            }
        }

@app.post("/api/inference")
async def start_inference(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    confidence: float = Form(0.25),
    iou: float = Form(0.45),
    max_det: int = Form(300)
):
    """Start media inference process for images or videos."""
    try:
        # Log request details for debugging
        print(f"Received inference request for file: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(f"Parameters: confidence={confidence}, iou={iou}, max_det={max_det}")
        
        # Generate a unique job ID
        job_id = str(uuid.uuid4())
        
        # Determine file type
        file_type = "image" if file.content_type and file.content_type.startswith("image/") else "video"
        print(f"Determined file type: {file_type}")
        
        # Save uploaded file to temporary file
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, file.filename)
        
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())
        
        # Create job tracking object
        job = InferenceJob(job_id, file.filename, None, file_type)
        INFERENCE_JOBS[job_id] = job
        
        # Start processing in background
        background_tasks.add_task(
            video_service.process_media,
            job_id,
            temp_file_path,
            file_type,
            confidence,
            iou,
            max_det
        )
        
        return {"job_id": job_id, "message": f"{file_type.capitalize()} inference job started"}
    
    except Exception as e:
        print(f"Error in start_inference: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error starting inference: {str(e)}")

@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an inference job."""
    if job_id not in INFERENCE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = INFERENCE_JOBS[job_id]
    
    # Log job details for debugging
    print(f"Job status request for job_id: {job_id}")
    print(f"Job status: {job.status}")
    print(f"Job file_type: {job.file_type}")
    print(f"Job result_path: {job.result_path}")
    
    return job.to_dict()

@app.get("/api/results/{job_id}")
async def get_processed_media(job_id: str):
    """Get the processed media (image or video) for a completed job."""
    if job_id not in INFERENCE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = INFERENCE_JOBS[job_id]
    
    # Log details for debugging
    print(f"Result request for job_id: {job_id}")
    print(f"Job status: {job.status}")
    print(f"Job file_type: {job.file_type}")
    print(f"Job result_path: {job.result_path}")
    
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status})")
    
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    
    media_type = "image/jpeg" if job.file_type == "image" else "video/mp4"
    
    return FileResponse(
        path=job.result_path,
        media_type=media_type,
        filename=f"processed_{job.file_name}"
    )

# Keep this endpoint for backward compatibility
@app.get("/api/video/{job_id}")
async def get_processed_video(job_id: str):
    """Get the processed video for a completed job (legacy endpoint)."""
    print(f"Legacy video endpoint request for job_id: {job_id}")
    
    if job_id not in INFERENCE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = INFERENCE_JOBS[job_id]
    
    # For backward compatibility, ensure this endpoint works for both videos and images
    # This helps with the frontend fallback mechanism
    if job.status != "completed":
        raise HTTPException(status_code=400, detail=f"Job not completed (status: {job.status})")
    
    if not job.result_path or not os.path.exists(job.result_path):
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Always set media type based on the actual file type
    media_type = "image/jpeg" if job.file_type == "image" else "video/mp4"
    
    return FileResponse(
        path=job.result_path,
        media_type=media_type,
        filename=f"processed_{job.file_name}"
    )

@app.get("/api/jobs/{job_id}/stream")
async def stream_job_progress(request: Request, job_id: str):
    """Stream job progress updates using Server-Sent Events."""
    if job_id not in INFERENCE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    async def event_generator():
        job = INFERENCE_JOBS[job_id]
        last_progress = -1
        
        # Send initial state
        yield {
            "event": "update",
            "data": json.dumps(job.to_dict())
        }
        
        # Stream updates
        while job.status not in ["completed", "error"]:
            if job.progress != last_progress:
                yield {
                    "event": "update",
                    "data": json.dumps(job.to_dict())
                }
                last_progress = job.progress
            
            # Check if client disconnected
            if await request.is_disconnected():
                break
                
            await asyncio.sleep(0.5)
        
        # Send final update
        yield {
            "event": "update",
            "data": json.dumps(job.to_dict())
        }
        
        # Send completion event if job is done
        if job.status in ["completed", "error"]:
            yield {
                "event": "complete",
                "data": json.dumps({"status": job.status, "message": job.message})
            }
    
    return EventSourceResponse(event_generator())

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its resources."""
    if job_id not in INFERENCE_JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = INFERENCE_JOBS[job_id]
    
    # Clean up MinIO objects
    try:
        minio = video_service.minio_client
        bucket = video_service.results_bucket
        
        # List objects with prefix job_id
        objects = minio.list_objects(bucket, prefix=job_id, recursive=True)
        for obj in objects:
            minio.remove_object(bucket, obj.object_name)
            
        print(f"Removed MinIO objects for job {job_id}")
    except Exception as e:
        print(f"Error removing MinIO objects: {e}")
    
    # Clean up local files
    if job.result_path and os.path.exists(job.result_path):
        try:
            parent_dir = os.path.dirname(job.result_path)
            shutil.rmtree(parent_dir)
            print(f"Removed local directory: {parent_dir}")
        except Exception as e:
            print(f"Error removing local directory: {e}")
    
    # Remove job from tracking
    del INFERENCE_JOBS[job_id]
    
    return {"message": f"Job {job_id} deleted"}

@app.get("/health")
async def health_check():
    """Check service health."""
    # Check MLflow connection
    try:
        # Perform a simple operation to verify MLflow connection
        mlflow.list_experiments()
        mlflow_status = "connected"
    except Exception as e:
        mlflow_status = f"error: {str(e)}"
    
    # Check MinIO connection
    try:
        minio_client = video_service.minio_client
        minio_client.list_buckets()
        minio_status = "connected"
    except Exception as e:
        minio_status = f"error: {str(e)}"
    
    # Check external API connection
    try:
        response = requests.get(f"{EXTERNAL_API_URL}/health")
        if response.status_code == 200:
            external_api_status = "connected"
        else:
            external_api_status = f"error: status code {response.status_code}"
    except Exception as e:
        external_api_status = f"error: {str(e)}"
    
    return {
        "status": "healthy" if all(s == "connected" for s in [mlflow_status, minio_status, external_api_status]) else "degraded",
        "timestamp": time.time(),
        "components": {
            "mlflow": mlflow_status,
            "minio": minio_status,
            "external_api": external_api_status
        },
        "jobs": {
            "active": sum(1 for job in INFERENCE_JOBS.values() if job.status not in ["completed", "error"]),
            "completed": sum(1 for job in INFERENCE_JOBS.values() if job.status == "completed"),
            "error": sum(1 for job in INFERENCE_JOBS.values() if job.status == "error"),
            "total": len(INFERENCE_JOBS)
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Get Prometheus metrics."""
    # Count jobs by status
    job_statuses = {}
    for job in INFERENCE_JOBS.values():
        job_statuses[job.status] = job_statuses.get(job.status, 0) + 1
    
    # Calculate average processing time for completed jobs
    completed_jobs = [job for job in INFERENCE_JOBS.values() if job.status == "completed"]
    avg_processing_time = sum((job.last_update_time - job.start_time) for job in completed_jobs) / len(completed_jobs) if completed_jobs else 0
    
    # Count jobs by file type
    file_types = {}
    for job in INFERENCE_JOBS.values():
        file_types[job.file_type] = file_types.get(job.file_type, 0) + 1
    
    # Generate metrics in Prometheus format
    metrics = []
    
    # Add job status metrics
    metrics.append("# HELP inference_jobs_total Total number of inference jobs by status")
    metrics.append("# TYPE inference_jobs_total gauge")
    for status, count in job_statuses.items():
        metrics.append(f'inference_jobs_total{{status="{status}"}} {count}')
    
    # Add file type metrics
    metrics.append("# HELP inference_file_types_total Total number of inference jobs by file type")
    metrics.append("# TYPE inference_file_types_total gauge")
    for file_type, count in file_types.items():
        metrics.append(f'inference_file_types_total{{type="{file_type}"}} {count}')
    
    # Add processing time metric
    metrics.append("# HELP inference_avg_processing_time_seconds Average processing time for completed jobs")
    metrics.append("# TYPE inference_avg_processing_time_seconds gauge")
    metrics.append(f"inference_avg_processing_time_seconds {avg_processing_time:.2f}")
    
    # Add system metrics
    metrics.append("# HELP system_memory_usage_bytes Memory usage in bytes")
    metrics.append("# TYPE system_memory_usage_bytes gauge")
    metrics.append(f"system_memory_usage_bytes {torch.cuda.memory_allocated() if torch.cuda.is_available() else 0}")
    
    return "\n".join(metrics)

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn when the script is executed directly
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=8000,
        reload=True
    )