# YOLOv8 End-to-End Object Detection – COCO6

## Overview
This project implements an **end-to-end object detection pipeline using YOLOv8**, covering dataset preparation, baseline training, controlled hyperparameter experiments, data augmentation strategies, model export, and real-world inference.

The goal is not only to train a YOLOv8 model, but also to **analyze robustness, generalization, and deployment trade-offs** using a reduced COCO subset and systematic experimentation.

> This project was developed as part of an academic course and extended as a practical computer vision case study.


## Dataset
- **Source:** Microsoft COCO (Roboflow v2, CC BY 4.0)
- **Classes (6):** person, cat, cell phone, sports ball, bottle, chair
- **Motivation:** Training on all 80 COCO classes is computationally expensive. A reduced 6-class subset enables faster experimentation while preserving small vs large objects, rigid vs non-rigid objects, and real-world scene diversity.
- **Structure:** Standard YOLO format with `data.yaml` configuration
- **Challenge:** Occlusion, scale variation, cluttered backgrounds


## Baseline Training
- **Model:** YOLOv8n (≈3M parameters)
- **Input size:** 640 × 640
- **Batch size:** 8
- **Epochs:** 20
- **Hardware:** RTX 3050 (8 GB)
- **Optimizer:** SGD (lr=0.01, momentum=0.9)
- **AMP & deterministic training:** enabled

### Baseline Validation Results (COCO6)
| Metric | Value |
|---|---:|
| Precision | **0.605** |
| Recall | **0.470** |
| mAP@50 | **0.495** |
| mAP@50–95 | **0.337** |

Stable convergence was observed. Recall limitations mainly affect **small or partially occluded objects** such as *cell phone* and *chair*.



## Hyperparameter Experiments
Two controlled experiments were conducted while keeping all other parameters fixed.

### Results Summary
| Configuration | Image Size | Batch | mAP@50 | mAP@50–95 | Precision | Recall | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 640 | 8 | 0.495 | 0.337 | 0.605 | 0.470 | ~4.5 h |
| Resolution ↑ | 768 | 8 | **0.514** | **0.353** | 0.613 | 0.486 | ~6.0 h |
| Batch ↑ | 640 | 16 | 0.501 | 0.345 | 0.613 | 0.473 | ~4.1 h |

**Key observations**
- Increasing image resolution provides the **largest accuracy improvement**, especially at higher IoU thresholds.
- Increasing batch size improves **training efficiency**, with limited accuracy gains.
- Resolution scaling introduces a clear **accuracy vs latency trade-off**.



## Data Augmentation Study
Three augmentation strategies were evaluated using the same architecture and dataset split.

| Augmentation Strategy | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Minimal | 0.591 | 0.451 | 0.471 | 0.317 |
| Aggressive | **0.616** | 0.435 | 0.471 | 0.306 |
| Task-Specific | 0.575 | **0.467** | **0.480** | 0.317 |

**Findings**
- Aggressive augmentation increases precision but **harms recall**, especially for small objects.
- Task-specific augmentation achieves the **best robustness–accuracy balance**.
- Overly strong augmentation can degrade performance in **lightweight models** such as YOLOv8n.



## Model Export & Deployment
The best-performing model (imgsz=768) was exported to ONNX.

### Accuracy Comparison
| Format | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| PyTorch (.pt) | 0.618 | 0.485 | 0.514 | 0.354 |
| ONNX (.onnx) | 0.612 | 0.482 | 0.511 | 0.351 |

Accuracy remains almost unchanged after conversion.

### Inference Speed
| Format | Inference Time (ms/image) |
|---|---:|
| PyTorch (.pt) | ~9.4 |
| ONNX (.onnx) | ~11.9 |

ONNX inference is slightly slower under default runtime settings, highlighting the importance of runtime optimization.



## Real-World Inference
The model was evaluated beyond the dataset split:
- **Dataset test images:** Reliable detection for dominant classes (*person*, *cat*)
- **Self-collected images:** Good generalization under unseen indoor conditions
- **Live RTSP inference:** Stable real-time detection with occasional false positives caused by shadows or visually similar textures

**Common failure cases**
- Small objects
- Partial occlusion
- Low contrast regions
- Complex backgrounds



## Key Takeaways
- Data augmentation must be **task-aware and model-capacity-aware**.
- Higher input resolution improves accuracy but increases inference latency.
- Aggressive augmentation is not always beneficial.
- ONNX export preserves accuracy but requires runtime tuning.
- Validation behavior aligns well with real-world inference patterns.



## Quickstart
```bash
pip install -r requirements.txt
python src/train.py --config configs/train_baseline.yaml
python src/export_onnx.py
python src/predict_rtsp.py --source rtsp://<camera>
```