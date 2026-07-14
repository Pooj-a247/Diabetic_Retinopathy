# Diabetic_Retinopathy
> **Identifying Diabetic Retinopathy with Deep Learning, Graph-Based Segmentation, and Immutable Blockchain Auditing.**

---

## -> Objective
Diabetic retinopathy is a major microvascular complication of diabetes and a leading cause of blindness in the working-age population worldwide, especially in rapidly expanding diabetic populations like India. 

The need for automated, offline screening is critical. In remote rural areas, time is lost between patients getting their eyes scanned, having images analyzed by off-site doctors, and scheduling follow-up care. By processing images on-site and in real-time, this project aims to allow people to receive a high-fidelity diagnosis and schedule treatment on the exact same day.

The core objective of this project is to build an edge-capable system that does not require cloud infrastructure. It achieves this by combining:
1. **Discrete Mathematical Algorithms** to maintain native pixel accuracy ($>4\text{K}$) without memory exhaustion.
2. **State-of-the-Art Vision Deep Learning** to classify disease severity.
3. **Decentralized Cryptographic Ledgers** to write immutable audit trails.

---

## -> Table of Contents
1. [Data & Preprocessing](#-data--preprocessing)
2. [Algorithms & CNN Architecture](#-algorithms--cnn-architecture)
3. [System Architecture](#-system-architecture)
4. [How to Run & Install](#-how-to-run--install)
5. [Expected Results](#-expected-results)
6. [Future Steps](#-future-steps)

---

## -> Data & Preprocessing

Retinal fundus photographs are highly variable, captured across different clinics with diverse camera hardware. To preserve diagnostic features without losing small microaneurysms, our preprocessing pipeline avoids standard aggressive downsampling:

1. **Quadtree Hierarchical Spatial Tiling:** Ultra-high-resolution images ($>4\text{K}$) are recursively partitioned into smaller quadrants based on local spatial variance. Smooth background regions are left at coarse tree levels, while complex areas with high vessel density or lesions are parsed down to high-resolution leaf nodes.
2. **Normalization & Contrast Enhancement:** Standardizes lighting variations using green-channel extraction and CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. **Class Imbalance Rectification:** To address the severe class imbalance typical of clinical datasets (where mild NPDR is heavily outnumbered by normal scans), we implement offline image augmentation and online **Focal Loss** tuning during model optimization.

---

## -> Algorithms & CNN Architecture

Unlike standard "black-box" pipelines, this project utilizes a hybrid design combining strict discrete math with deep neural networks:

                  ┌──────────────────────────────────────┐
                  │      Input Fundus Image (>4K)        │
                  └──────────────────┬───────────────────┘
                                     ▼
                    [ Quadtree Spatial Decomposition ]
                                     │
            ┌────────────────────────┴────────────────────────┐
            ▼                                                 ▼
[ Dijkstra Vessel Segmentation ]                  [ YOLOv10 Target Detection ]
(Traces vascular tree via graph paths)            (Localizes microaneurysms/exudates)
            │                                                 │
            └────────────────────────┬────────────────────────┘
                                     ▼
                        [ ViT-H/16 Classification ] ──> [ Hungarian Longitudinal Match ]
                        (Assigns 5-class ETDRS grade)     (Tracks lesion changes over time)

### 1. Graph-Based Vessel Segmentation (Dijkstra's Shortest Path)
Retinal blood vessels are mapped by constructing an undirected graph $G = (V, E)$ over the pixel array. Edge weights are defined using a customized cost field:
$$C(u, v) = \exp\left( \alpha \cdot (1 - F(v)) + \beta \cdot \|\nabla I(v)\|^2 \right)$$
where $F(v)$ is the Frangi filter output and $\|\nabla I(v)\|^2$ is the gradient magnitude. Dijkstra’s algorithm solves the single-source shortest path, maintaining clean, continuous vascular structures.

### 2. Longitudinal Tracking (Hungarian Algorithm)
To monitor patients over multiple screenings, detected lesion centroids are tracked between checkups (Time $T_{k-1} \to T_k$). We solve this as a bipartite matching task in $O(V^3)$ time, matching baseline and follow-up coordinates to pinpoint expanding, stagnant, or resolving pathologies.

### 3. Deep Learning Networks
* **Feature Extraction:** **ConvNeXt-XL** serves as our core CNN for spatial visual representations.
* **Classification:** **Vision Transformers (ViT-H/16)** pre-trained on ImageNet-22K capture long-range contextual relationships across the entire fundus.
* **Localization:** **YOLOv10** detects and outputs bounding boxes for localized lesions.

---

## -> System Architecture

* **Frontend:** Built with **React.js** and **Cornerstone.js** for high-performance, web-based DICOM image viewing.
* **Backend:** Scaled via **FastAPI** asynchronous microservices connected to **PostgreSQL** and **TimescaleDB** for transactional/temporal tracking.
* **Security & Trust:** Encrypted using **AES-GCM-256** at rest and **TLS 1.3** in transit. Cryptographic report hashes are written to a private **Hyperledger Fabric** network for secure audit trails.
* **Edge Deployment:** Models are compiled using **NVIDIA TensorRT** and deployed on an embedded **NVIDIA Jetson Orin Nano** module for low-power, zero-latency inference directly in the field.

---

## -> How to Run & Install

### System Prerequisites
Ensure your hardware environment has the following dependencies:
* Ubuntu 22.04 LTS or JetPack 6.x (for Jetson edge hardware)
* Python 3.10+
* CUDA 12.2 & TensorRT 10.x
* Docker & Docker Compose

### Installation
Clone the repository and set up your virtual environment:
```bash
git clone [https://github.com/Pooj-a247/Diabetic_Retinopathy.git](https://github.com/Pooj-a247/Diabetic_Retinopathy.git)
cd Diabetic_Retinopathy
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```
---

