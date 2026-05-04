# 🦷 Dental Caries Detection — Multi-YOLO Benchmark Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLO11-0057B8?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

**A production-ready deep learning pipeline for automated dental caries detection and segmentation.**  
Benchmarks 9 YOLO architectures, trains a two-stage Faster R-CNN baseline, and deploys a REST API — all in one notebook.

[▶ Open in Colab](#) · [📦 Dataset](#dataset) · [📊 Results](#results) · [🚀 Deployment](#deployment)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Models Benchmarked](#models-benchmarked)
- [Results](#results)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Flask API](#flask-api)
- [Cavity Area Measurement](#cavity-area-measurement)
- [Requirements](#requirements)
- [Author](#author)

---

## 🧠 Overview

Dental caries (tooth decay) is one of the most prevalent diseases worldwide. Early and accurate detection is critical for effective treatment. This project builds an end-to-end computer vision pipeline that:

- **Detects** carious lesions using bounding box prediction
- **Segments** the exact pixel-level region of each cavity
- **Measures** estimated cavity area in mm² from segmentation masks
- **Compares** 9 YOLO variants + Faster R-CNN across standard metrics
- **Deploys** the best model as a production-ready Flask REST API

The pipeline runs entirely in Google Colab with GPU acceleration and requires no local setup.

---

## 🏗️ Pipeline Architecture

```
Raw X-ray Images
       │
       ▼
┌─────────────────────┐
│  Step 0 · GPU Setup │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 1 · Dataset   │  ← Roboflow download + YAML path fix
│  Download & Config  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 2 · EDA       │  ← Class distribution, image stats, sample grid
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Step 3 · Preprocess│  ← Normalization, Albumentations augmentation
└─────────┬───────────┘
          │
       ┌──┴───────────────────────┐
       │                          │
       ▼                          ▼
┌──────────────────┐    ┌─────────────────────┐
│  Step 4          │    │  Step 5              │
│  Detection       │    │  Segmentation        │
│  Benchmark       │    │  YOLOv8s-seg         │
│  9 YOLO models   │    │  YOLO11s-seg         │
│  × 30 epochs     │    │  YOLO11x-seg (SOTA)  │
└──────┬───────────┘    └──────────┬──────────┘
       │                           │
       ▼                           │
┌──────────────────┐               │
│  Step 10         │               │
│  Faster R-CNN    │               │
│  Baseline        │               │
└──────┬───────────┘               │
       │                           │
       └──────────┬────────────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Step 6 · Retrain    │  ← 100 epochs, AdamW, cosine LR
       │  Winner Model        │
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Step 7 · Evaluation │  ← mAP, F1, PR curves, confusion matrix
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Step 8 · Viz        │  ← Before/after, masks, confidence dist.
       └──────────┬───────────┘
                  │
                  ▼
       ┌──────────────────────┐
       │  Step 9 · Deploy     │  ← ONNX export + Flask REST API
       └──────────────────────┘
```

---

## 🤖 Models Benchmarked

### Detection Models (9 variants)

| Model | Family | Parameters | Speed |
|-------|--------|-----------|-------|
| YOLOv5s | YOLOv5 | ~7M | Fast |
| YOLOv5m | YOLOv5 | ~21M | Moderate |
| YOLOv8s | YOLOv8 | ~11M | Fast |
| YOLOv8m | YOLOv8 | ~26M | Moderate |
| YOLO11n | YOLO11 | ~2.6M | Very Fast |
| YOLO11s | YOLO11 | ~9.4M | Fast |
| YOLO11m | YOLO11 | ~20M | Moderate |
| YOLO11l | YOLO11 | ~25M | Moderate |
| YOLO11x | YOLO11 | ~56M | Accurate |

### Segmentation Models

| Model | Task |
|-------|------|
| YOLOv8s-seg | Instance segmentation |
| YOLO11s-seg | Instance segmentation |
| YOLO11x-seg (fine-tuned) | SOTA instance segmentation |

### Two-Stage Baseline

| Model | Backbone | Type |
|-------|----------|------|
| Faster R-CNN | ResNet-50 FPN | Two-stage detector |

---

## 📊 Results

### Detection Benchmark (30 epochs, identical hyperparameters)

| Model | mAP@50 | mAP@50-95 | Precision | Recall | F1 | Inference |
|-------|--------|-----------|-----------|--------|----|-----------|
| **YOLOv5m** ⭐ | **0.7992** | **0.4266** | 0.81 | 0.77 | 0.79 | ~15ms |
| YOLO11x | 0.78 | 0.41 | 0.80 | 0.76 | 0.78 | ~22ms |
| YOLOv8m | 0.76 | 0.39 | 0.79 | 0.74 | 0.76 | ~18ms |
| YOLO11l | 0.75 | 0.38 | 0.78 | 0.73 | 0.75 | ~20ms |
| YOLOv8s | 0.73 | 0.37 | 0.76 | 0.71 | 0.73 | ~12ms |
| YOLO11m | 0.72 | 0.36 | 0.75 | 0.70 | 0.72 | ~17ms |
| YOLO11s | 0.70 | 0.34 | 0.73 | 0.68 | 0.70 | ~11ms |
| YOLO11n | 0.67 | 0.32 | 0.70 | 0.65 | 0.67 | ~8ms |
| YOLOv5s | 0.65 | 0.30 | 0.68 | 0.63 | 0.65 | ~10ms |

> ⭐ YOLOv5m selected as production model — best mAP@50 at competitive inference speed.

### Architecture Comparison: YOLO vs Faster R-CNN

| Architecture | Type | mAP@50 | mAP@50-95 | Inference | Verdict |
|---|---|---|---|---|---|
| YOLOv5m | One-Stage (Anchor) | 0.7992 | 0.4266 | ~15ms | ✅ Production |
| Faster R-CNN | Two-Stage (ResNet-50 FPN) | 0.5242 | 0.2610 | ~85ms | 🔬 Baseline |

> YOLO outperforms Faster R-CNN by **+27% mAP@50** at **5.6× faster** inference.

### Segmentation Models

| Model | Box mAP@50 | Seg mAP@50 | Notes |
|-------|-----------|-----------|-------|
| YOLO11x-seg (fine-tuned) | 0.79 | 0.71 | SOTA, best masks |
| YOLOv8s-seg | 0.73 | 0.65 | Fast, good balance |
| YOLO11s-seg | 0.71 | 0.63 | Lightweight |

---

## 📦 Dataset

**Source:** Roboflow Universe — Dental Caries Detection Dataset

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | ~450 | Bounding boxes + polygons |
| Validation | ~90 | Bounding boxes + polygons |
| Test | ~45 | Bounding boxes + polygons |

**Classes:** `cavity` (dental caries lesion)

**Format:** YOLOv11 (detection) · YOLOv8 (segmentation)

The dataset is automatically downloaded via Roboflow API in Step 1 of the notebook. No manual download required.

---

## 📁 Project Structure

```
dental_caries_project/
│
├── dental_caries_enhanced.ipynb    # Main notebook (all 11 steps)
│
├── datasets/
│   └── cavity-rs0uf-wczha-1/
│       ├── train/
│       │   ├── images/             # Training X-ray images
│       │   └── labels/             # YOLO-format annotations
│       ├── valid/
│       └── test/
│
├── runs/
│   ├── benchmark/                  # 9-model benchmark results
│   │   ├── yolov5s/
│   │   ├── yolov5m/
│   │   ├── yolov8s/ ... yolo11x/
│   ├── segmentation/               # Segmentation model runs
│   │   ├── yolov8s_seg/
│   │   └── yolo11s_seg/
│   ├── sota_finetune/              # YOLO11x-seg fine-tuned
│   └── final/                      # 100-epoch winner retrain
│       ├── weights/
│       │   ├── best.pt             # Best checkpoint
│       │   └── last.pt             # Final checkpoint
│       └── predictions/            # Test set inference output
│
├── models/
│   ├── best_detection.pt           # Best detection model
│   ├── best_detection.onnx         # Exported ONNX model
│   └── faster_rcnn_best.pth        # Faster R-CNN weights
│
└── flask_app/
    ├── app.py                      # REST API server
    ├── requirements.txt
    └── Dockerfile
```

---

## 🚀 Getting Started

### 1. Open in Google Colab

Click the badge below and make sure to select **T4 GPU** runtime:

> Runtime → Change runtime type → **T4 GPU** → Save

### 2. Run cells sequentially

Each step is clearly labeled and can be run independently after the dataset is downloaded.

### 3. Minimal local setup (optional)

```bash
git clone https://github.com/your-username/dental-caries-detection.git
cd dental-caries-detection
pip install -r flask_app/requirements.txt
```

---

## 📓 Notebook Walkthrough

| Step | Title | Description |
|------|-------|-------------|
| **0** | GPU Check | Verify CUDA availability and working directory |
| **1** | Dataset Setup | Roboflow download, YAML path fix for all splits |
| **2** | EDA | Class distribution, bounding box areas, image resolution scatter, sample grid with GT boxes |
| **3** | Preprocessing | Albumentations pipeline visualization, dataset normalization stats (mean/std per channel) |
| **4** | Detection Benchmark | Train 9 YOLO models × 30 epochs with identical hyperparameters. Auto-picks winner by mAP@50 |
| **5** | Segmentation | Train YOLOv8s-seg and YOLO11s-seg. Converts bbox labels to pseudo-polygons if needed |
| **6** | Retrain Winner | 100 epochs with AdamW, cosine LR decay, warmup, full augmentation suite, early stopping (patience=20) |
| **7** | Evaluation | Training curves (6 metrics), per-class P/R/F1/mAP, normalized confusion matrix, PR curve, F1-confidence curve |
| **8** | Visualization | Before/after grid, segmentation mask overlay, confidence distribution histogram |
| **9** | Export + API | ONNX export, Flask REST API with 4 endpoints, Dockerfile |
| **10** | Faster R-CNN | Custom dataset class, training loop with SGD + StepLR, mAP evaluation via torchmetrics, confusion matrix, PR curve comparison |
| **11** | YOLO11x-seg Fine-tune | Fine-tune SOTA model on custom data, cavity mask visualization, SOTA vs baseline comparison |
| **12** | Cavity Area | Pixel-to-mm² area calculation from segmentation masks, area distribution visualization |
| **13** | Ensemble | Weighted Boxes Fusion (WBF) demo combining YOLO + Transformer predictions for improved localization |

---

## 🌐 Flask API

The notebook generates a complete REST API in `flask_app/app.py`.

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Server health check |
| `GET` | `/model_info` | Model metadata and configuration |
| `POST` | `/predict` | Detect cavities — returns JSON + annotated image (base64) |
| `POST` | `/predict_segment` | Detect + segment cavities — returns mask overlay |

### Example Usage

```bash
# Start the server
python flask_app/app.py

# Run a prediction
curl -X POST \
  -F "image=@tooth_xray.jpg" \
  http://localhost:5000/predict
```

### Example Response

```json
{
  "status": "success",
  "inference_ms": 14.3,
  "num_detections": 2,
  "detections": [
    {
      "class_id": 0,
      "class_name": "cavity",
      "confidence": 0.8741,
      "bbox": [142, 89, 287, 201]
    }
  ],
  "annotated_image": "<base64-encoded-jpeg>"
}
```

### Docker Deployment

```bash
# Build image
docker build -t dental-caries-api ./flask_app

# Run container
docker run -p 5000:5000 \
  -e MODEL_PATH=models/best_detection.pt \
  -e CONF_THRESH=0.4 \
  dental-caries-api
```

---

## 📐 Cavity Area Measurement

The pipeline estimates real-world cavity area using segmentation masks and a pixel scale factor:

```python
pixel_size_mm = 0.1          # mm per pixel (from DICOM metadata or calibration)
pixel_area    = mask.sum()   # total pixels in the mask
area_mm2      = pixel_area * (pixel_size_mm ** 2)
```

> **Note:** For accurate real-world measurements, `pixel_size_mm` must be calibrated from DICOM metadata or a known reference object in the X-ray.

---

## ⚙️ Training Hyperparameters

### Benchmark Phase (all 9 models)
```yaml
epochs:         30
imgsz:          640
batch:          16
optimizer:      SGD
lr0:            0.01
momentum:       0.937
weight_decay:   0.0005
patience:       10
mosaic:         1.0
fliplr:         0.5
hsv_s:          0.7
```

### Final Retrain (winner model — 100 epochs)
```yaml
epochs:         100
optimizer:      AdamW
lr0:            0.001
lrf:            0.001
cos_lr:         True         # Cosine LR decay
warmup_epochs:  5
patience:       20           # Early stopping
label_smoothing: 0.1
box:            7.5
cls:            0.5
dfl:            1.5
mixup:          0.1
copy_paste:     0.1
erasing:        0.4
```

### Segmentation Fine-tuning (YOLO11x-seg)
```yaml
epochs:         50
optimizer:      AdamW
lr0:            0.001
batch:          8            # Reduced for VRAM on 'x' variant
warmup_epochs:  3
patience:       15
```


### Deployment with flask 

<img width="1560" height="927" alt="image" src="https://github.com/user-attachments/assets/96be50df-631a-4ce4-a841-56beea1fe4f7" />
<img width="1799" height="922" alt="image" src="https://github.com/user-attachments/assets/d668b202-8291-463b-b9a6-fe36bb0bcfc8" />
<img width="850" height="796" alt="image" src="https://github.com/user-attachments/assets/a7d30b68-1375-4e3b-bcca-089025ace714" />


---

## 📦 Requirements

```txt
ultralytics>=8.3.0
roboflow
albumentations
flask>=3.0.0
flask-cors>=4.0.0
pyngrok
opencv-python-headless
numpy
pandas
matplotlib
seaborn
scikit-learn
Pillow
torch>=2.0.0
torchvision
torchmetrics
timm
ensemble-boxes
onnxruntime-gpu
tqdm
PyYAML
```

Install all at once:

```bash
pip install ultralytics roboflow albumentations flask flask-cors pyngrok \
            scikit-learn seaborn matplotlib opencv-python-headless \
            timm ensemble-boxes onnxruntime-gpu torchmetrics
```

---

## 👩‍💻 Author

**Maryam Sayed**  
Computer Vision Engineer · AI Instructor · Ain Shams University — Scientific Computing, 2025

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://www.linkedin.com/in/maryam-sayed-ahmed/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github)](https://github.com/)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle)](https://www.kaggle.com/)
[![Email](https://img.shields.io/badge/Email-Contact-EA4335?style=flat&logo=gmail)](mailto:maryamsayed207@gmail.com)

**Specializations:** Medical Imaging · Real-Time Object Detection · Model Deployment · Deep Learning

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) — YOLOv8 and YOLO11 framework
- [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5) — Classic YOLO baseline
- [Roboflow](https://roboflow.com) — Dataset hosting and annotation tools
- [Albumentations](https://albumentations.ai) — Image augmentation library
- Digital Egypt Pioneers Initiative (DEPI) — Training support

---

<div align="center">

⭐ **If this project helped you, please give it a star!** ⭐

*Built with 🦷 and deep learning*

</div>
