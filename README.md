# Cattle-Weight-Identification-System

## Before all you need to install [Node JS](https://nodejs.org/en/download) and [Python 3.11](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe)
### Do not forget to add [Python](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe) to PATH

## Also install [Anaconda](https://drive.google.com/file/d/1FPkHBKS8DUgHBjUwH5iMxuTc7u5BwiIL/view?usp=sharing) to make and environment for backend server
## Then download the backend [models](https://www.kaggle.com/models/vitaliyblackhole/cattle-weight-identification) through Kaggle, no registration needed just scroll down and download three models or download archive

## After that clone the current repository and move the model files to the backend folder to run properly
* best_cattle_segmentation_model_6.keras
* best_enhanced_triple_cattle_weight_model_res18_100ep_2600p.pth
* best_keypoints_model_9pts_limited.keras

## 1. Backend Setup 
### Install Anaconda
### Open anaconda terminal and make following steps:

* conda create -n backend python=3.10.16
* conda activate backend

### Navigate to the backend directory in your terminal: 
* cd backend

### Install pip

* conda install pip
  
### Install the required Python packages using pip:
* pip install -r requirements.txt


### Initialize database by following commands:
* $env:FLASK_APP = "backend.py"
* flask --app backend.py db init
* flask --app backend.py db migrate -m "Initial migration with user and history tables."
* flask --app backend.py db upgrade

### Start the backend server
* python backend.py

### Make sure you see the following lines:
* Segmentation model (PyTorch) loaded successfully
* Keypoint model (Keras) loaded successfully
* Weight prediction model loaded successfully

## 2. Frontend Setup 

### Navigate to the backend directory in your terminal: 
* cd frontend

### Install packages used in project by command:

* npm install --legacy-peer-deps

### Start the frontend server:

* npm start
