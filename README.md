# Realtime-Traffic-Object-Detection-and-Tracking

Our project aims to enhance real-time traffic monitoring by replacing manual or rule-based approaches with intelligent ML-driven automation. By using YOLOv8 for object detection and DeepSORT for tracking, we aim to provide accurate and real-time analytics for urban mobility, improving efficiency, reducing costs, and enabling data-driven decisions for city planning and public safety.

### Contributors

| Name                     | Responsible for                           | Link to the commits in the repo |
|--------------------------|-------------------------------------------|------------------------------------|
| All team members         | General project implementation            |                                    |
| Kirit Govindaraja Pillai | Model training and hyperparameter tuning  | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=kiritgpillai) |
| Sancho Wong              | Data pipeline and real-time data handling | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=sanchoo1) |
| Ruochong Wang            | Deployment and monitoring infrastructure  | [Commits](https://github.com/kiritgpillai/Realtime-Traffic-Object-Detection-and-Tracking/commits?author=bBlakesy) |

### System diagram

![System Diagram](https://github.com/user-attachments/assets/e6a1f5f1-f596-4fdd-8011-0616f7598a27)

This architecture includes:
- ETL pipeline for ingesting data from BDD100K, live feeds, and simulated streams.
- Model training with YOLOv8 and DeepSORT, orchestrated with GitHub Actions and Terraform.
- CI/CD pipeline to support seamless deployment to model serving platforms with APIs and MLflow tracking.
- Testing phases (Locust, Evidently) before production with monitoring tools like Label Studio and dashboards.

### Summary of outside materials

| Name         | How it was created                                                                              | Conditions of use                                   |
|--------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------|
| BDD100K      | Collected by UC Berkeley with self-driving vehicles in diverse weather, light, and location   | Public license for academic and non-commercial use  |
| YOLOv8       | Trained on COCO dataset; open-source by Ultralytics                                           | Open-source (GPLv3)                                 |
| DeepSORT     | Extension of SORT using Kalman Filters + ReID network                                         | Open-source                                         |

### Summary of infrastructure requirements

| Requirement     | How many/when                     | Justification                                        |
|-----------------|-----------------------------------|-----------------------------------------------------|
| `m1.medium` VMs | 2 for data processing + API       | Persistent processing and hosting needs             |
| `gpu_mi100`     | 2x weekly 4hr blocks              | Required for training YOLOv8 on custom datasets     |
| Floating IPs    | 1 static for backend              | Access from external apps and frontend              |
| Storage         | 100GB                             | Store processed frames, models, and logs            |

### Detailed design plan

#### Model training and training platforms

1. **Strategy**: Fine-tune YOLOv8 on a curated subset of BDD100K. Use DeepSORT for object tracking post-detection. GitHub Actions used to automate training pipeline.
2. **Justification**: Real-time capability and performance balance; both models are widely tested in research and industry.
3. **Relation to lecture**: Covers Units 4 and 5 – model tuning, training reproducibility.
4. **Difficulty**: Advanced dataset preprocessing, GPU training, and tracking integration.

#### Model serving and monitoring platforms

1. **Strategy**: Serve models via an API with Redis + FastAPI. MLflow used for model versioning. Locust simulates load; Evidently monitors drift.
2. **Relation to lecture**: Covers Units 6 and 7 – real-time serving, monitoring, testing.
3. **Difficulty**: Canary testing and dashboard integration for observability.

#### Data pipeline

1. **Strategy**: ETL pipeline extracts frames from video feeds, preprocesses data, and pushes it to the model.
2. **Tools**: Uses Docker containers for reproducibility and Terraform for resource provisioning.
3. **Relation to lecture**: Covers Unit 8 – structured, automated data ingestion.

#### Continuous X

1. **Strategy**: GitHub Actions for CI/CD from training to deployment.
2. **Tools**: Terraform for infra provisioning; test coverage tracking for quality assurance.
3. **Relation to lecture**: Covers Unit 3 – continuous integration and infrastructure-as-code.

