# Cattle-Weight-Identification-System
Cattle-Weight-Identification-System

## Before all you need to install [Node JS](https://nodejs.org/en/download) and [Python 3.11](https://www.python.org/downloads/release/python-3119/)
## Than download the backend [models](https://www.kaggle.com/models/vitaliyblackhole/cattle-weight-identification) through Kaggle, no registration needed just scroll down and download three models or download archive of them

## After that clone the current repository and move the model files to the backend folder to run properly
* best_cattle_segmentation_model_6.keras
* best_enhanced_triple_cattle_weight_model_res18_100ep_2600p.pth
* best_keypoints_model_9pts_limited.keras

## 1. Backend Setup 🐍
### Install Python Dependencies
### Navigate to the backend directory in your terminal: 
* cd backend
  
### Install the required Python packages using pip:
* pip install -r requirements.txt

### Initialize database by following commands:
* set FLASK_APP=backend.py
* flask db init
* flask db migrate -m "Initial migration with user and history tables."
* flask db upgrade

### Start the backend server
* python backend.py

### Make sure you see the following lines:
* Segmentation model (PyTorch) loaded successfully
* Keypoint model (Keras) loaded successfully
* Weight prediction model loaded successfully
