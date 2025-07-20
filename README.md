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

## 8. To work with your own dataset
To work with your own cattle image dataset, you need to create it, load it, and retrain the neural models. The process is divided into three main parts, one for each model.
## Part A: Retraining the Keypoint Detection Model
This model's purpose is to find 9 specific anatomical points on the cattle.
### 1. Preparing Your Data
Image Folder: Place all your cattle images (preferably from a side-on view) into a single directory.
Annotation File: You must create annotations in the COCO JSON format.
* o	Create a single JSON file (e.g., my_coco_annotations.json).
* o	This file must contain lists for images, annotations, and categories.
* o	Image entries must include id, file_name, width, and height.
* o	Annotation entries must link to an image via image_id and contain a keypoints array. This array is a flat list of 27 numbers: [x1, y1, v1, x2, y2, v2, ...] for the 9 keypoints. v stands for visibility.
### 2. Configuring the Script (Cattle_keypoints.ipynb)
Update Paths: In the second code cell, change the values of these variables to your folder and file paths:
* o	new_images_dir = r"C:\path\to\your\images"
* o	new_annotations_file = r"C:\path\to\your\my_coco_annotations.json"
Enable Training: Find the training function call, which is commented out by default. Uncomment it to activate the training process:
history = train_model_func(X_train, y_train, X_val, y_val, model)
Adjust Parameters: You can modify EPOCHS, BATCH_SIZE, or DATA_LIMIT to suit your dataset and hardware.
### 3. Running the Training
* •	Execute all cells in the Cattle_keypoints.ipynb notebook from top to bottom.
* •	The script will train the model on your data and automatically save the best version as best_keypoints_model_9pts_limited.keras.
## Part B: Retraining the Image Segmentation Model
This model identifies the pixels belonging to the cattle, a reference sticker, and the background.
### 1. Preparing Your Data
* •	Folder Structure: Create a main dataset folder. Inside it, create two subfolders: images and annotations.
* •	Images: Place your original cattle images into the images subfolder.
* •	Masks: For each image, create a corresponding segmentation mask and save it in the annotations subfolder.
The mask filename must match the image filename exactly, with ___fuse.png added at the end (e.g., cow1.jpg has a mask named cow1.jpg___fuse.png).
Masks must be colored with specific RGB values:
* Cattle: (255, 30, 249)
* Sticker: (0, 117, 255)
* Background: (0, 255, 193)
### 2. Configuring the Script (Cattle_Segmentation.ipynb)
Update the Directory Path: In the Config class, modify the BASE_DIR variable to point to your main dataset folder.
* BASE_DIR = r'C:\path\to\your\segmentation\dataset'
Adjust Parameters: You can change BATCH_SIZE, LEARNING_RATE, or NUM_EPOCHS in the Config class.
### 3. Running the Training
Execute all cells in the Cattle_Segmentation.ipynb notebook. The main() function automates the process of loading data, training, and saving the best model to the path specified in SAVED_MODEL_PATH.
## Part C: Retraining the Final Weight Estimation Model
This model integrates the images and keypoints to predict the final weight.
### 1. Preparing Your Data
Image and Keypoint Data: Have the folders and the JSON file from Part A ready.
Weight Data File: Create a CSV file (e.g., weights.csv) containing two columns: filename (the image file name) and weight (the corresponding weight in kg).
Model Files: Ensure the trained models from Part A (.keras file) and Part B (.keras or .pth file) are available.
### 2. Configuring the Script (Final_model.ipynb)
Update Model Paths: At the top of the script, set the correct paths for the models you trained in the previous steps.
* KERAS_MODEL_PATH = r'path\to\your\best_keypoints_model_9pts_limited.keras'
* SEGMENTATION_MODEL_PATH = r'path\to\your\best_cattle_segmentation_model_6.keras'
Update Data Paths: In the main() function, update the directory and file paths to match your dataset.
* original_dir: Path to the folder with your original images.
* segmented_dir: Path to the folder with your segmentation masks.
* csv_file: Path to your weights.csv file.
* json_annotation_file: Path to your keypoint annotation JSON file.
Enable Training: Find the RUN_TRAINING variable and set it to True.
* RUN_TRAINING = True
### 3. Running the Training
* •	Execute all cells in the Final_model.ipynb notebook.
* •	The script will load all data sources, filter for complete samples, and begin training.
* •	The best-performing weight prediction model will be saved to the location specified by the BEST_MODEL_SAVE_PATH variable.

