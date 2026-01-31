# YOLOv8 Object Detection Pipeline – COCO6

## Overview
End-to-end object detection pipeline using YOLOv8, covering dataset preparation,
baseline training, hyperparameter analysis, data augmentation strategies,
model export, and real-world inference.

## Dataset
- Source: COCO (Roboflow v2, CC BY 4.0)
- Classes: person, cat, cell phone, sports ball, bottle, chair
- Motivation for reduced-class setup
- YOLO directory structure and data.yaml

## Baseline Training
- Model: YOLOv8n
- Training setup
- Validation metrics
- Training dynamics and convergence

## Hyperparameter Experiments
- Image resolution scaling
- Batch size scaling
- Accuracy vs efficiency trade-offs

## Data Augmentation Study
- Minimal augmentation
- Aggressive augmentation
- Task-specific augmentation
- Comparative analysis and discussion

## Model Export & Deployment
- ONNX export
- Accuracy comparison
- Inference speed analysis

## Real-World Inference
- Dataset test images
- Self-collected images
- Live RTSP inference
- Failure cases

## Key Takeaways
- Augmentation must match model capacity
- Resolution improves small object detection at higher cost
- Deployment format affects runtime behavior
