# 🛰️ Multispectral Satellite Water Segmentation (U-Net)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📌 Project Overview
Automates the detection and segmentation of water bodies from **Sentinel-2 Multispectral imagery**. By leveraging the unique spectral signatures of water in the **Near-Infrared (NIR)** spectrum, the system achieves high-precision mapping that outperforms standard RGB-based analysis.



## 🚀 Key Features
* **Physics-Informed Labeling:** Utilizes the **Normalized Difference Water Index (NDWI)** for pixel-level ground truth generation.
* **Multispectral Inference:** Processes 4-channel tensors (Red, Green, Blue, and NIR) to distinguish between water and deep shadows.
* **Deep Learning Architecture:** Implements a **U-Net** with a **ResNet34 backbone**.
* **Interactive Deployment:** Includes a **Streamlit Dashboard** for real-time inference with dynamic thresholding.

## 📊 Performance Metrics
The model was evaluated on 100+ truly unseen geospatial samples:
* **Mean IoU (Intersection over Union):** `0.8024`
* **Validation Accuracy:** `99.45%`
* **Validation Loss:** `0.0138`

## 🛠️ Tech Stack
* **Backend:** TensorFlow, Keras
* **Geospatial:** Rasterio, NumPy
* **UI/UX:** Streamlit
* **Environment:** Google Colab (Training), Local Python (Deployment)

## 📁 Project Structure
```text
├── app.py                     # Streamlit UI Application
├── vssc_final_water_model.h5  # Pre-trained U-Net Model
├── water_detection.ipynb      # Training & Evaluation Pipeline
├── requirements.txt           # Dependency List
└── README.md                  # Project Documentation

<img width="1856" height="822" alt="image" src="https://github.com/user-attachments/assets/bcaf8442-81b1-4d8f-9a48-2c53aa18846c" />
<img width="1883" height="814" alt="image" src="https://github.com/user-attachments/assets/a6da82d0-6561-471d-b517-b80a3891c037" />
<img width="1911" height="282" alt="image" src="https://github.com/user-attachments/assets/a9b06b75-455c-4a9d-a8d4-42358e14b17f" />

Installation:

**Clone the Repository:**
git clone [https://github.com/your-username/satellite-water-segmentation.git](https://github.com/your-username/satellite-water-segmentation.git)
cd satellite-water-segmentation

**Install Dependencies**
pip install -r requirements.txt

**Run the Dashboard**
streamlit run app.py

🔬 Methodology
Data Preprocessing: 13-band EuroSAT MS data was filtered and balanced to mitigate class imbalance.
Spectral Indexing: Calculated NDWI using $NDWI = \frac{Green - NIR}{Green + NIR}$ to create high-fidelity masks.
Model Training: Employed Binary Cross-Entropy loss and Adam optimizer with 4-channel input normalization (value/10000).
Inference: The system extracts B4, B3, B2, and B8 to predict water probability maps.

🌟 Acknowledgments
Dataset provided by EuroSAT (Zenodo).
