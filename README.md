# Realtime Traffic Object Detection and Tracking

## Project Overview

Our project aims to enhance real-time traffic monitoring by replacing manual or rule-based traffic observation with intelligent ML-driven automation. Currently, many traffic management systems rely on human operators or simple sensors, which is labor-intensive and often limited in scope. By using YOLOv8 for object detection, this system can automatically detect vehicles in live video feeds, providing accurate real-time analytics for urban mobility. This improvement boosts efficiency (less manual monitoring), reduces operational costs, and enables data-driven decision-making for city planning and public safety. In summary, the system promises to improve traffic flow management and safety metrics by leveraging state-of-the-art machine learning in place of the status quo.


## Contributors

| Name                     | Responsible for                           | Link to the commits in the repo                                                                                       |
| ------------------------ | ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Kirit Govindaraja Pillai | Model training, testing and Inferencing   | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=kiritgpillai) |
| Sancho Wong              | Data pipeline and real-time data handling | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=sanchoo1)     |
| Ruochong Wang            | Deployment and monitoring infrastructure  | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=bBlakesy)     |

## Target Customer and Value Proposition

All design decisions in this project are specifically tailored to meet the needs of the New York City Department of Transportation (NYC DOT). As one of the most complex and heavily trafficked urban environments, New York City requires a robust, scalable, and real-time traffic monitoring system to address congestion, traffic violations, and pedestrian safety. Therefore, our system is optimized for the unique challenges faced by NYC DOT, focusing on automated traffic monitoring through detection systems to reduce manual oversight and leverage real-time data from existing CCTV infrastructure.

## System Architecture
![System Architecture](https://github.com/user-attachments/assets/72ab68e4-e5a3-4f83-b82b-ac1d01b695d1)

#### Data [(ETL Pipeline)](./model_pipeline)

* An ETL pipeline for downloading, extracting and sorting the COCO dataset, specifically catered to traffic-related classes, live camera feeds (e.g., CCTV or streamed videos), and simulated video streams.
* The pipeline extracts frames or images, performs necessary preprocessing (resizing, normalization, etc.), and feeds the cleaned data into the model, which then moves for inferencing.

#### Model Training Platform

* A training module where the YOLOv8 object detection model is trained and logged on the COCO dataset.
* The model is trained on either the MI100 or A100 GPU, depending on lease availability at the time.
* The model is logged to MLflow, with artifacts pushed to a MinIO bucket corresponding to the run ID and saved as a .pt file of the best model of the run.
* Inferencing is performed by referencing the model from the bucket and passing the video or image uploaded by the user for evaluation.
* Hardware includes GPU instances (e.g., AMD MI100 GPUs and NVIDIA A100 GPUs for high-performance training) on Chameleon Cloud.
* The training process logs metrics and saves models to the registry (using MLflow for experiment tracking and model versioning).

#### Model Serving and API Layer

* Once trained, the YOLOv8 model is deployed behind a FastAPI service, providing a RESTful API endpoint for inference.

#### CI/CD Pipeline

* Continuous integration and deployment pipelines are established using GitHub Actions.
* GitHub Actions also manage hosting an ArgoCD key for automated deployments.
* Terraform and Ansible are used for infrastructure management and configuration.
* Any code changes or new model training runs trigger automated testing and deployment scripts.
* [Terraform](./model-serving/infra) scripts manage cloud infrastructure, including VMs, containers, and networking.
* [Ansible](./model-serving/ansible) was used to automate the configuration of servers and manage deployment environments.

#### Monitoring and Evaluation Tools

* The system undergoes testing phases before production updates.
* Grafana and Prometheus are used for real-time monitoring, displaying metrics such as throughput, inference rate, and system load.
* FastAPI serves as the API framework for handling inference requests, enabling integration with monitoring tools.
* A batch of offline evaluation and inference is conducted using pytest to ensure model connectivity, accuracy, and performance.

## Datasets and Models

### Data Preprocessing and Class Concentration

The COCO dataset originally contains a wide range of object classes. For our application, we concentrated the dataset to include only the classes relevant to traffic monitoring, such as cars, buses, trucks, motorcycles, and bicycles. We carefully filtered out unrelated classes to streamline the model's focus on vehicle detection, which significantly improved inference speed and accuracy for our use case. This class concentration was crucial to ensure the model's robustness when deployed in real-time traffic scenarios.

The COCO dataset consists of approximately **118,000 training images**, **5,000 validation images**, and **20,000 testing images**, summing up to around **25GB** including the labels. These images cover a diverse set of real-world scenarios and include annotations for multiple object classes. By filtering for traffic-specific classes, we tailored the dataset to better match our application requirements.

| Name   | Description/How It Was Created                                                                                                           | Conditions of Use   |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------- |
| COCO   | A large-scale object detection dataset containing diverse objects in context, used as the base for fine-tuning traffic-specific classes. | Open-source (GPLv3) |
| YOLOv8 | An object detection model in the Ultralytics library known for its high speed and accuracy, suitable for real-time applications.         | Open-source (GPLv3) |

---
## Infrastructure Requirements

| Requirement                  | Quantity/Timing                             | Justification                                                                     |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------- |
| m1.large VM instances        | 1 VM (continuous for development & hosting) | General-purpose VMs for data processing and hosting the API backend.              |
| GPU nodes (gpu\_mi100 /a100) | 2x weekly, 6-hour blocks (on-demand)        | High-performance GPU instances for model training jobs.                           |
| Floating IP                  | 2 (persistent)                              | Ensures external connectivity to the containers and the persistent storage on the cloud. |
| Storage (Block/Object)       | \~80 GB persistent storage                  | To save processed data sets, trained models, logs, and hold the training code.    |


## Frontend and Inference Service 

The [video inference service](./video-inference-ui) provides a robust API for object detection using YOLO models. It handles both video and image uploads, processes them using pre-trained models stored in MLflow (http://129.114.27.202:30938/), and returns annotated results with detected objects. The system features a model caching mechanism for improved performance, progress tracking via server-sent events, and comprehensive Prometheus-style metrics collection for monitoring system health, resource usage, and inference performance. All processed media files and job metadata are stored in MinIO (http://129.114.27.202:30001/) for persistence. The service includes a middleware for capturing detailed HTTP request metrics and offers both synchronous and streaming endpoints for job status updates. Additional features include Python garbage collection monitoring, process-level resource tracking, and flexible job management with cleanup of completed tasks. The service integrates with external APIs for analytics while performing local inference, offering a complete solution for video and image object detection workloads.

## Monitoring and Experiment Tracking

The system is monitored using Grafana (http://129.114.27.202:30091/) for visualization and Prometheus (http://129.114.27.202:30090/) for metrics collection. The Grafana dashboards displays real-time data including inference latency, system performance, and resource utilization. Prometheus scrapes metrics exposed via the FastAPI service and logs them for further analysis and alerting. [PyTest](./tests/offline) was also used to perfom some offline evaluation of the model.

## Challenges and Mitigation

* Pipeline Setup Issues: Faced challenges while setting up some components in the pipeline, leading to unexpected delays and troubleshooting. As a stopgap measure, we initiated training instances manually and re-logged the new data to restore the pipeline functionality.

* Data Loss from Block Storage: Encountered a sudden loss of logging data from the block storage, necessitating the implementation of redundant data backup solutions. As a mitigation, we scheduled periodic data integrity checks and automated backups to minimize the impact of future data loss.

## Conclusion

Our real-time traffic monitoring system leverages YOLOv8 to automate vehicle detection, significantly improving traffic analytics compared to traditional methods. The project is designed with scalability in mind, incorporating CI/CD and continuous training practices to adapt to evolving data patterns, thereby supporting city planning and public safety initiatives efficiently.
