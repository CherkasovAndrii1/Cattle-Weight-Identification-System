# Cattle-Weight-Identification-System

## Before all you need to install [Node JS](https://nodejs.org/dist/v22.17.0/node-v22.17.0-x64.msi) and [Python 3.11](https://www.python.org/ftp/python/3.11.9/python-3.11.9.exe)
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


## 3. After all if you made everything okay you will face the login form then click register and register an account, all your data will be stored on your device in local database
<img width="1876" height="959" alt="image" src="https://github.com/user-attachments/assets/94887014-1851-496d-94b8-12e8cf3a1b23" />


## 4. If you want to test out the system you can get some photos form test-set folder and choose the method you want it can be segmentation, extracting point or the weight estimation
<img width="1876" height="961" alt="image" src="https://github.com/user-attachments/assets/89ef132f-edcb-497b-b8df-8196c6ae7532" />


## 5. To upload the photo you can use drag and drop function or upload the photo from your device
<img width="1880" height="947" alt="image" src="https://github.com/user-attachments/assets/504b314f-9534-4072-8ed1-9d90498f4bc4" />

## 6. After successfull upload "Process" button will become activate and you can press it and in few seconds you will get the result
<img width="1868" height="967" alt="image" src="https://github.com/user-attachments/assets/68d90e13-8385-4bba-b97a-82eebbeefce4" />
<img width="1866" height="968" alt="image" src="https://github.com/user-attachments/assets/22136be1-d9b9-42c8-a2b8-81ebe7e49195" />
<img width="1881" height="967" alt="image" src="https://github.com/user-attachments/assets/d19a791b-f94d-46ab-9b5b-71a79c4c5c11" />

## 7. If you want you can check request history in the top menu also you can export it on your device in json format
<img width="1869" height="965" alt="image" src="https://github.com/user-attachments/assets/eede1575-e6a8-4035-9663-458ac28a2546" />


